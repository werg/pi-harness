import { type Context, copyJson, type JsonValue } from "@earendil-works/chord";
import { awaitWithContext } from "@earendil-works/chord/context";
import { overlap } from "@earendil-works/chord/delta";
import type { ImageContent, TextContent, ToolCall, ToolResultMessage } from "@earendil-works/pi-ai";
import { validateToolArguments } from "@earendil-works/pi-ai/utils/validation";
import { defineDoc } from "../documents.ts";
import { AssistantEntry, DirectToolCallEntry, DirectToolResultEntry, ToolResultEntry } from "../entries.ts";
import { defineTask } from "../tasks.ts";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, utf8ByteLength } from "../truncate.ts";
import type {
	ConversationId,
	EntryId,
	JsonObject,
	RunningTask,
	Task,
	TaskId,
	TaskOptions,
	TaskOutcomeError,
	TaskRuntime,
	Tx,
	TypedEntry,
} from "../types.ts";
import { toolMatches } from "./define.ts";
import { assignJson, snapshotError } from "./json.ts";
import { clearProgress, finishSlot, LiveDoc, type ToolSlot, toolSlot } from "./live.ts";
import {
	boundOutput,
	OutputBuffer,
	type OutputCheckpoint,
	type OutputLimits,
	PROGRESS_BYTES_PER_SECOND,
	Progress,
} from "./output.ts";
import type {
	ToolBinding,
	ToolControl,
	ToolDiagnostic,
	ToolExecutionApi,
	ToolExecutionResult,
	ToolExecutionWait,
	ToolHooks,
	ToolRegistration,
} from "./types.ts";
import { recordUsage } from "./usage.ts";

export type DirectToolCall = { readonly id: string; readonly name: string; readonly arguments: JsonObject };
export type ToolTaskInput = {
	source: { kind: "assistant" | "direct"; entryId: EntryId };
	callId: string;
	binding: ToolBinding;
};

/** Commit real direct-call source and its ordinary ToolTask atomically with product admission. */
export async function createDirectToolTask(
	tx: Tx,
	conversationId: ConversationId,
	call: DirectToolCall,
	binding: ToolBinding,
	options: Omit<TaskOptions, "conversationId">,
): Promise<TaskId<ToolTaskResult>> {
	if (!call.id || !call.name || call.name !== binding.name)
		throw new Error("Direct tool call conflicts with its selected definition");
	const source = await tx.appendEntry(DirectToolCallEntry, conversationId, { data: { call } });
	return tx.createTask(
		ToolTask,
		{ source: { kind: "direct" as const, entryId: source.id }, callId: call.id, binding },
		{ ...options, conversationId },
	);
}

export type ToolTaskCheckpoint =
	| { phase: "call" }
	/** Durable intent: the final arguments and the replay policy recorded before execution. */
	| {
			phase: "execute";
			arguments: JsonObject;
			replay: "safe" | "unsafe";
			continuation?: JsonValue;
			output?: OutputCheckpoint;
	  };

export type ToolTaskResult = { entryId: EntryId; control?: ToolControl };

type Runtime = TaskRuntime<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult, ToolHooks>;
type Content = (TextContent | ImageContent)[];

/** Canonical progress owned by the actual task; generation's live slot is its presentation projection. */
export const ToolProgressDoc = defineDoc<{
	output: OutputCheckpoint | null;
	droppedBytes: number;
	droppedLines: number;
	details?: JsonValue;
	diagnostics: ToolDiagnostic[];
}>({
	kind: "pi.tool-progress",
	version: 1,
	scope: "task",
	initial: () => ({ output: null, droppedBytes: 0, droppedLines: 0, diagnostics: [] }),
	checkpointWhen: () => true,
});

/**
 * Built-in tool task: resolves the called tool among its phase agent's tools, validates, runs `beforeTool`, records intent,
 * executes, runs `afterTool`, and appends the result, all in one `call` handler so nothing separates resolution from
 * settlement. `execute` is reached only by recovery and applies the replay rule.
 */
export const ToolTask = defineTask<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult, ToolHooks>({
	name: "pi.tool",
	version: 2,
	initial: () => ({ phase: "call" }),
	phases: {
		call: async (task, runtime, context) => {
			const call = await readCall(runtime, task.input, context);
			const tool = (await runtime.agent(context)).tools.find((each) => each.name === call.name);
			if (tool === undefined) {
				const error = harnessError("tool_unavailable", `Tool ${call.name} is not available`);
				return settle(runtime, call, COMPLETED, () => error, context);
			}
			if (!toolMatches(tool, task.input.binding))
				return parkBinding(runtime, task.state.checkpoint, task.input.binding, tool, context);
			const prepared = prepare(tool, call.arguments as JsonObject);
			const checked = "error" in prepared ? prepared : validate(tool, call, prepared.args);
			if ("error" in checked) return settle(runtime, call, COMPLETED, () => invalid(checked.error), context);
			let args = checked.args;
			let block: string | undefined;
			await runtime.hooks.each("beforeTool", async (hook) => {
				if (block !== undefined) return;
				try {
					const decision = await hook({ ...call, arguments: args }, runtime, context);
					if (decision?.block !== undefined) block = decision.block;
					else if (decision?.arguments !== undefined) args = decision.arguments;
				} catch (error) {
					if (runtime.signal.aborted) throw error;
					block = errorText(error);
				}
			});
			if (block !== undefined) {
				const blocked = harnessError("blocked", `Tool call blocked: ${block}`);
				return settle(runtime, call, COMPLETED, () => blocked, context);
			}
			const validated = validate(tool, call, args);
			if ("error" in validated) return settle(runtime, call, COMPLETED, () => invalid(validated.error), context);
			const final = validated.args;
			await runtime.commit(async (tx) => {
				await tx.doc(ToolProgressDoc, runtime.taskId);
				const slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId);
				if (slot !== undefined) slot.status = "running";
				const intent = { phase: "execute", arguments: final, replay: tool.replay ?? "unsafe" } as const;
				return { status: "running", checkpoint: intent };
			}, context);
			await run(runtime, call, tool, final, task.input.binding, context);
		},
		/** Recovery after intent: rerun only when the stored and the current policy both say `safe`. */
		execute: async (task, runtime, context) => {
			const { arguments: args, replay, continuation, output } = task.state.checkpoint;
			try {
				const call = await readCall(runtime, task.input, context);
				const tool = (await runtime.agent(context)).tools.find((each) => each.name === call.name);
				// An unsafe interrupted invocation cannot rerun, regardless of which code is now installed.
				if (continuation === undefined && (replay !== "safe" || tool?.replay !== "safe")) {
					const message = `Tool ${call.name} was interrupted and may have partially run`;
					await settle(
						runtime,
						call,
						{ status: "failed", error: { message } },
						(slot) => fromSlot(slot, "interrupted", message),
						context,
					);
					return;
				}
				if (!toolMatches(tool, task.input.binding))
					return await parkBinding(runtime, task.state.checkpoint, task.input.binding, tool, context);
				// A replay-safe invocation starts reporting from scratch; an owned continuation keeps its output.
				if (continuation === undefined)
					await runtime.commit(async (tx) => {
						const progress = await tx.doc(ToolProgressDoc, runtime.taskId);
						progress.output = null;
						delete progress.details;
						progress.diagnostics = [];
						progress.droppedBytes = 0;
						progress.droppedLines = 0;
						const slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId);
						if (slot !== undefined) clearProgress(slot);
						return undefined;
					}, context);
				await run(runtime, call, tool, args, task.input.binding, context, continuation, false, output);
			} catch (error) {
				if (continuation === undefined || runtime.signal.aborted) throw error;
				// Selection, restoration and rejected result commits also leave the admitted operation owned.
				await runtime.parkFailure(error, context);
			}
		},
	},
	abort: async (task, runtime, context) => {
		const call = await readCall(runtime, task.input, context);
		const checkpoint = task.state.checkpoint;
		const tool = (await runtime.agent(context)).tools.find((each) => each.name === call.name);
		if (checkpoint.phase === "execute" && !toolMatches(tool, task.input.binding))
			return parkBinding(runtime, checkpoint, task.input.binding, tool, context);
		if (checkpoint.phase === "execute" && tool?.cancel !== undefined) {
			return run(
				runtime,
				call,
				tool,
				checkpoint.arguments,
				task.input.binding,
				context,
				checkpoint.continuation,
				true,
				checkpoint.output,
			);
		}
		const message = `Tool ${call.name} was aborted`;
		await settle(runtime, call, { status: "aborted" }, (slot) => fromSlot(slot, "aborted", message), context);
	},
});

/** Keep the canonical operation and cleanup debt live; changed code never claims its outcome. */
async function parkBinding(
	runtime: Runtime,
	checkpoint: ToolTaskCheckpoint,
	expected: ToolBinding,
	tool: ToolRegistration | undefined,
	context: Context,
): Promise<void> {
	const after = runtime.registry.revision;
	await runtime.commit(
		() => ({
			status: "waiting",
			checkpoint,
			condition: {
				kind: "registry",
				after,
				reason: {
					code: "incompatible_tool",
					name: expected.name,
					expectedVersion: expected.version,
					currentVersion: tool?.version ?? (tool === undefined ? null : 1),
				},
			},
		}),
		context,
	);
}

/** Resolve the exact original source; direct calls never claim provider authorship. */
async function readCall(runtime: Runtime, input: ToolTaskInput, context: Context): Promise<ToolCall> {
	if (input.source.kind === "direct") {
		const entry = await runtime.entry(DirectToolCallEntry, input.source.entryId, context);
		const call = entry?.data?.call;
		if (!call || call.id !== input.callId || entry.model !== undefined)
			throw new Error(`Entry ${input.source.entryId} has no original direct tool call ${input.callId}`);
		return { type: "toolCall", ...call };
	}
	const entry = await runtime.entry(AssistantEntry, input.source.entryId, context);
	const message = entry?.model?.[0];
	const call =
		message?.role === "assistant"
			? message.content.find(
					(content): content is ToolCall => content.type === "toolCall" && content.id === input.callId,
				)
			: undefined;
	if (call === undefined) throw new Error(`Entry ${input.source.entryId} has no tool call ${input.callId}`);
	return call;
}

/** Arguments, or why they are invalid. */
type Checked = { readonly args: JsonObject } | { readonly error: string };

/** The call's arguments as repaired by the tool; a throwing repair makes them invalid. */
function prepare(tool: ToolRegistration, args: JsonObject): Checked {
	if (tool.prepareArguments === undefined) return { args };
	try {
		return { args: tool.prepareArguments(args) as JsonObject };
	} catch (error) {
		return { error: errorText(error) };
	}
}

/** Arguments validated and coerced against the implementation's schema. */
function validate(tool: ToolRegistration, call: ToolCall, args: JsonObject): Checked {
	try {
		return { args: validateToolArguments(tool, { ...call, arguments: args }) as JsonObject };
	} catch (error) {
		return { error: errorText(error) };
	}
}

function invalid(message: string): ToolExecutionResult {
	return harnessError("invalid_arguments", message);
}

/** What a running tool reported through its api: output, the last details, and diagnostics. */
type Reported = {
	readonly output: OutputBuffer;
	readonly limits: OutputLimits;
	readonly diagnostics: ToolDiagnostic[];
	details: JsonValue | undefined;
};

/** Execute with the resolved implementation, then settle its result. */
async function run(
	runtime: Runtime,
	call: ToolCall,
	tool: ToolRegistration,
	args: JsonObject,
	binding: ToolBinding,
	context: Context,
	continuation?: JsonValue,
	cancelling = false,
	output?: OutputCheckpoint,
): Promise<void> {
	const limits: OutputLimits = {
		maxBytes: tool.outputLimits?.maxBytes ?? DEFAULT_MAX_BYTES,
		maxLines: tool.outputLimits?.maxLines ?? DEFAULT_MAX_LINES,
		retain: tool.outputLimits?.retain ?? "head",
	};
	const retained =
		continuation === undefined ? undefined : await runtime.snapshot(ToolProgressDoc, runtime.taskId, context);
	const reported: Reported = {
		output: new OutputBuffer(limits, retained?.output ?? output),
		limits,
		diagnostics: [],
		details: undefined,
	};
	if (continuation !== undefined) {
		reported.details = retained?.details === undefined ? undefined : copyJson(retained.details);
		if (retained?.diagnostics) reported.diagnostics.push(...(copyJson(retained.diagnostics) as ToolDiagnostic[]));
	}
	const progress = publishProgress(runtime, reported, context);
	let retainedContinuation = continuation;
	let ended = false;
	const assertLive = (): void => {
		if (ended) throw new Error(`Tool call ${call.id} has settled`);
	};
	const api: Omit<ToolExecutionApi, "env"> = {
		taskId: runtime.taskId,
		conversationId: runtime.conversationId,
		callId: call.id,
		get executionData() {
			return binding.data === undefined ? undefined : immutableData(copyJson(binding.data));
		},
		continuation,
		registry: runtime.registry,
		agent: runtime.agent,
		models: runtime.models,
		output: (chunk, skipped) => {
			assertLive();
			if (reported.output.push(chunk, skipped)) progress.mark();
		},
		outputWindow:
			limits.retain === "tail"
				? {
						maxBytes: limits.maxBytes,
						maxLines: limits.maxLines,
						minIntervalMs: runtime.settings.progress.outputIntervalMs,
						bytesPerSecond: PROGRESS_BYTES_PER_SECOND,
					}
				: undefined,
		diagnostic: (diagnostic) => {
			assertLive();
			reported.diagnostics.push(copyJson(diagnostic, { omitUndefinedProperties: true }) as ToolDiagnostic);
			progress.mark();
		},
		details: async (value, detailsContext) => {
			assertLive();
			detailsContext.abortSignal?.throwIfAborted();
			reported.details = copyJson(value, { omitUndefinedProperties: true });
			const committed = progress.markAndWait();
			// Cancelling the wait leaves the update in place; the commit's own outcome stays observed.
			committed.catch(() => {});
			return awaitWithContext(committed, detailsContext);
		},
		commit: async (change, commitContext) => {
			let result: Awaited<ReturnType<typeof change>> | undefined;
			await runtime.commit(async (tx) => {
				result = await change(tx);
				return undefined;
			}, commitContext);
			return result as Awaited<ReturnType<typeof change>>;
		},
		retainContinuation: async (candidate, change, commitContext) => {
			assertLive();
			if (tool.cancel === undefined) throw new Error(`External tool ${call.name} has no cancellation contract`);
			const retained = copyJson(candidate);
			await runtime.commit(async (tx, current) => {
				await change(tx);
				const checkpoint = await checkpointProgress(runtime, tx, current, reported);
				return { status: "running", checkpoint: { ...checkpoint, continuation: retained } };
			}, commitContext);
			retainedContinuation = retained;
		},
		memo: runtime.memo,
		createTask: async <I, S extends { phase: string }, R, H extends object>(
			task: Task<I, S, R, H>,
			input: I,
			options: Omit<TaskOptions, "conversationId">,
			taskContext: Context,
		): Promise<TaskId<R>> => {
			let id: TaskId<R> | undefined;
			await runtime.commit(async (tx) => {
				id = await tx.createTask(task, input, options);
				return undefined;
			}, taskContext);
			return id!;
		},
		getTask: runtime.getTask,
		waitForTask: runtime.waitForTask,
		conversation: runtime.conversation,
		snapshot: runtime.snapshot,
		snapshotAsOf: runtime.snapshotAsOf,
		watchDoc: runtime.watchDoc,
	};

	let result: ToolExecutionResult | ToolExecutionWait;
	let ending = COMPLETED;
	// Execution time of this attempt; a rerun after recovery measures only itself.
	let durationMs: number | undefined;
	try {
		// Built for this call, so a rerun after recovery gets the conversation's environment at that time.
		const env = await runtime.env(context);
		const startedAt = performance.now();
		try {
			result = await (cancelling ? tool.cancel! : tool.execute)(
				args,
				{
					...api,
					env,
					get executionData() {
						return binding.data === undefined ? undefined : immutableData(copyJson(binding.data));
					},
				},
				context,
			);
			if ("wait" in result && tool.cancel === undefined)
				throw new Error(`External tool ${call.name} has no cancellation contract`);
		} finally {
			durationMs = Math.round(performance.now() - startedAt);
		}
	} catch (error) {
		// A failed observation or cleanup does not establish the admitted operation's terminal outcome.
		if (cancelling || retainedContinuation !== undefined || runtime.signal.aborted) {
			ended = true;
			reported.output.end();
			const pending = await progress.stop();
			try {
				if (runtime.signal.aborted) throw error;
				await runtime.parkFailure(error, context, (tx, current) =>
					checkpointProgress(runtime, tx, current, reported),
				);
			} catch (failure) {
				for (const waiter of pending) waiter.reject(failure);
				throw failure;
			}
			for (const waiter of pending) waiter.reject(error);
			return;
		}
		result = { isError: true, diagnostics: [toolDiagnostic("tool_error", errorText(error))] };
		// No external continuation was admitted. End failed with the original durable diagnostic.
		ending = { status: "failed", error: snapshotError(error) };
	}
	if ("wait" in result) {
		ended = true;
		reported.output.end();
		const pending = await progress.stop();
		try {
			const waiting = result;
			await runtime.commit(async (tx, current) => {
				const checkpoint = await checkpointProgress(runtime, tx, current, reported);
				return {
					status: "waiting",
					checkpoint: { ...checkpoint, continuation: waiting.continuation },
					condition: waiting.wait,
				};
			}, context);
			for (const waiter of pending) waiter.resolve();
		} catch (error) {
			for (const waiter of pending) waiter.reject(error);
			if (retainedContinuation !== undefined && !runtime.signal.aborted) {
				await runtime.parkFailure(error, context, (tx, current) =>
					checkpointProgress(runtime, tx, current, reported),
				);
				return;
			}
			throw error;
		}
		return;
	}
	ended = true;
	reported.output.end();
	// Details still waiting for a progress commit settle with the terminal commit, the final flush.
	const pending = await progress.stop();
	try {
		const settled = await finalResult(runtime, call, result, reported, context);
		await settle(runtime, call, cancelling ? { status: "aborted" } : ending, () => settled, context, durationMs);
	} catch (error) {
		for (const waiter of pending) waiter.reject(error);
		if (retainedContinuation !== undefined && !runtime.signal.aborted) {
			await runtime.parkFailure(error, context, (tx, current) => checkpointProgress(runtime, tx, current, reported));
			return;
		}
		throw error;
	}
	for (const waiter of pending) waiter.resolve();
}

/** The continuation's output counters and visible progress advance in the same transaction as its wait. */
async function checkpointProgress(
	runtime: Runtime,
	tx: Tx,
	current: RunningTask<ToolTaskInput, ToolTaskCheckpoint, ToolTaskResult>,
	reported: Reported,
): Promise<Extract<ToolTaskCheckpoint, { phase: "execute" }>> {
	const checkpoint = current.state.checkpoint;
	if (checkpoint.phase !== "execute") throw new Error("Tool continuation has no committed intent");
	const slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId);
	const output = reported.output.snapshot();
	const progress = await tx.doc(ToolProgressDoc, runtime.taskId);
	progress.output = reported.output.checkpoint();
	progress.droppedBytes = output.droppedBytes;
	progress.droppedLines = output.droppedLines;
	if (reported.details !== undefined) progress.details = reported.details;
	progress.diagnostics = [...reported.diagnostics];
	if (slot !== undefined) {
		slot.output = output.text;
		slot.droppedBytes = output.droppedBytes;
		slot.droppedLines = output.droppedLines;
		if (reported.details !== undefined) slot.details = reported.details;
		if (reported.diagnostics.length) slot.diagnostics = [...reported.diagnostics];
	}
	return { ...checkpoint, output: reported.output.checkpoint() };
}

/**
 * Throttled commits of what the tool reported into its `pi.live.tools` slot, each writing only what changed since the
 * last one.
 */
function publishProgress(runtime: Runtime, reported: Reported, context: Context): Progress {
	let written = { text: "", details: reported.details, diagnostics: reported.diagnostics.length };
	return new Progress(
		async () => {
			// Capture everything synchronously: the tool keeps reporting while the commit is in flight.
			const snapshot = reported.output.snapshot();
			const outputCheckpoint = reported.output.snapshotCheckpoint();
			const current = { text: snapshot.text, details: reported.details, diagnostics: reported.diagnostics.length };
			const added = reported.diagnostics.slice(written.diagnostics, current.diagnostics);
			const detailsChanged = current.details !== written.details;
			// What the commit writes, as Chord diffs the string: an append, a trim plus an append of what follows the shared
			// part, or the whole window when its bounded overlap search finds nothing.
			let bytes = 0;
			if (snapshot.text !== written.text) {
				const shared = snapshot.text.startsWith(written.text)
					? written.text.length
					: overlap(written.text, snapshot.text, 65_536);
				bytes += utf8ByteLength(snapshot.text.slice(shared));
			}
			if (detailsChanged) bytes += utf8ByteLength(JSON.stringify(current.details ?? null));
			if (added.length > 0) bytes += utf8ByteLength(JSON.stringify(added));
			await runtime.commit(async (tx) => {
				const progress = await tx.doc(ToolProgressDoc, runtime.taskId);
				assignJson(progress as unknown as Record<string, JsonValue>, "output", outputCheckpoint);
				progress.droppedBytes = snapshot.droppedBytes;
				progress.droppedLines = snapshot.droppedLines;
				if (detailsChanged && current.details !== undefined)
					assignJson(progress as unknown as Record<string, JsonValue>, "details", current.details);
				for (const diagnostic of added) progress.diagnostics.push(diagnostic);
				const slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId);
				if (slot === undefined) return undefined;
				// REMINDER: assign `output` as one string field. Chord then diffs it into an append, or a trim plus an
				// append for a sliding tail; replacing the slot object would record the whole window on every commit.
				if ((slot.output ?? "") !== snapshot.text) slot.output = snapshot.text;
				if (snapshot.droppedBytes > 0) slot.droppedBytes = snapshot.droppedBytes;
				if (snapshot.droppedLines > 0) slot.droppedLines = snapshot.droppedLines;
				// Diff details leaf by leaf and append new diagnostics, so each commit writes only what changed.
				if (detailsChanged && current.details !== undefined) {
					assignJson(slot as unknown as Record<string, JsonValue>, "details", current.details);
				}
				if (added.length > 0) {
					if (slot.diagnostics === undefined) slot.diagnostics = [];
					for (const diagnostic of added) slot.diagnostics.push(diagnostic);
				}
				return undefined;
			}, context);
			written = current;
			return bytes;
		},
		(error) => {
			// Rejections after an abort mark or close are expected; the committed state stays consistent.
			if (!runtime.signal.aborted) runtime.report(error);
		},
		runtime.settings.progress.outputIntervalMs,
	);
}

/**
 * The settled result: the tool's result with the retained output and last details as fallbacks, its diagnostics after
 * those reported through the api, `afterTool` applied, and explicit text bounded, with the Harness's truncation
 * diagnostic last.
 */
async function finalResult(
	runtime: Runtime,
	call: ToolCall,
	result: ToolExecutionResult,
	reported: Reported,
	context: Context,
): Promise<ToolExecutionResult> {
	const harness: ToolDiagnostic[] = [];
	const retained = result.content === undefined ? reported.output.snapshot() : undefined;
	const content: Content =
		retained === undefined ? result.content! : retained.text === "" ? [] : [{ type: "text", text: retained.text }];
	let final: ToolExecutionResult = {
		...result,
		content,
		details: result.details === undefined ? reported.details : result.details,
		diagnostics: [...reported.diagnostics, ...(result.diagnostics ?? [])],
	};
	await runtime.hooks.each("afterTool", async (hook) => {
		final = (await hook(call, final, runtime, context)) ?? final;
	});
	// The retained output's truncation applies only while afterTool kept that content.
	if (retained !== undefined && retained.droppedBytes > 0 && final.content === content) {
		harness.push(truncated(retained, reported.limits.retain));
	}
	const bounded = boundContent(final.content ?? [], reported.limits);
	if (bounded.droppedBytes > 0) harness.push(truncated(bounded, reported.limits.retain));
	return { ...final, content: bounded.content, diagnostics: [...(final.diagnostics ?? []), ...harness] };
}

/**
 * Commit the tool's terminal state: append its result entry, mark its slot done, and complete or end aborted with the
 * entry ID. `build` receives the slot so interruption and abort can report the durable partial output.
 */
async function settle(
	runtime: Runtime,
	call: ToolCall,
	ending: Ending,
	build: (slot: Readonly<ToolSlot> | undefined) => ToolExecutionResult,
	context: Context,
	durationMs?: number,
): Promise<void> {
	await runtime.commit(async (tx) => {
		const slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId);
		const progress = await tx.doc(ToolProgressDoc, runtime.taskId);
		const result = build({
			callId: call.id,
			name: call.name,
			taskId: runtime.taskId,
			status: "running",
			output: progress.output?.text ?? "",
			droppedBytes: progress.droppedBytes,
			droppedLines: progress.droppedLines,
			...(progress.details === undefined ? {} : { details: progress.details }),
			diagnostics: progress.diagnostics,
		});
		const task = await tx.task(runtime.taskId);
		if (!task) throw new Error("Tool settlement has no original task");
		const input = task.input as ToolTaskInput;
		const entry =
			input.source.kind === "direct"
				? await appendDirectToolResult(tx, runtime.conversationId, input.source.entryId, call, result, durationMs)
				: await appendToolResult(tx, runtime.conversationId, call, result, runtime.now(), durationMs);
		if (slot !== undefined) finishSlot(slot, entry.id);
		const entryId = entry.id;
		if (ending.status === "aborted")
			return { status: "terminal", outcome: { status: "aborted", result: { entryId } } };
		if (ending.status === "failed") {
			return { status: "terminal", outcome: { status: "failed", error: ending.error, result: { entryId } } };
		}
		// Tools build control objects freely; drop keys set to undefined so the task result is strict JSON.
		const control =
			result.control === undefined
				? {}
				: { control: copyJson(result.control as JsonValue, { omitUndefinedProperties: true }) as ToolControl };
		return { status: "terminal", outcome: { status: "completed", result: { entryId, ...control } } };
	}, context);
}

/**
 * How a tool task ends; the result entry is appended either way. `failed` (execution threw or was interrupted)
 * records cancellation intent for the conversations the call owns; a result with `isError` still completes.
 */
type Ending =
	| { readonly status: "completed" | "aborted" }
	| { readonly status: "failed"; readonly error: TaskOutcomeError };

const COMPLETED: Ending = { status: "completed" };

/** An error result from the slot's durable partial output, details, and diagnostics. */
function fromSlot(slot: Readonly<ToolSlot> | undefined, code: string, message: string): ToolExecutionResult {
	const diagnostics = [...(slot?.diagnostics ?? [])];
	const droppedBytes = slot?.droppedBytes ?? 0;
	if (droppedBytes > 0) diagnostics.push(truncated({ droppedBytes, droppedLines: slot?.droppedLines ?? 0 }));
	diagnostics.push(toolDiagnostic(code, message));
	return {
		content: slot?.output === undefined || slot.output === "" ? [] : [{ type: "text", text: slot.output }],
		isError: true,
		...(slot?.details === undefined ? {} : { details: slot.details }),
		diagnostics,
	};
}

/** An error result the Harness writes itself: no content and one `error` diagnostic with `code`. */
export function harnessError(code: string, message: string): ToolExecutionResult {
	return { content: [], isError: true, diagnostics: [toolDiagnostic(code, message)] };
}

function toolDiagnostic(code: string, message: string): ToolDiagnostic {
	return { severity: "error", code, message };
}

/** The Harness's truncation diagnostic; `retain` is unknown when rebuilt from a slot after recovery. */
function truncated(
	dropped: { readonly droppedLines: number; readonly droppedBytes: number },
	retain?: "head" | "tail",
): ToolDiagnostic {
	const kept = retain === undefined ? "" : ` to its ${retain === "head" ? "beginning" : "end"}`;
	return {
		severity: "warn",
		code: "truncated",
		message: `Output truncated${kept}: ${dropped.droppedLines} lines, ${dropped.droppedBytes} bytes dropped`,
	};
}

/**
 * Append a `pi.tool-result` entry. The content ends with the rendered diagnostics, so the stored message is exactly
 * what the model sees; `data` keeps the structured list. A result's usage is added to `pi.usage` in the same commit.
 */
export async function appendToolResult(
	tx: Tx,
	conversationId: ConversationId,
	call: ToolCall,
	result: ToolExecutionResult,
	timestamp: number,
	durationMs?: number,
): Promise<TypedEntry<{ diagnostics: ToolDiagnostic[] }>> {
	const diagnostics = [...(result.diagnostics ?? [])];
	const content: Content = [...(result.content ?? [])];
	if (diagnostics.length > 0) content.push({ type: "text", text: renderDiagnostics(diagnostics) });
	const message = {
		role: "toolResult",
		toolCallId: call.id,
		toolName: call.name,
		content,
		...(result.details === undefined ? {} : { details: result.details }),
		...(result.usage === undefined ? {} : { usage: result.usage }),
		isError: result.isError ?? false,
		...(durationMs === undefined ? {} : { durationMs }),
		timestamp,
	} as ToolResultMessage;
	if (result.usage !== undefined) await recordUsage(tx, conversationId, "tools", call.name, result.usage);
	return tx.appendEntry(ToolResultEntry, conversationId, { model: [message], data: { diagnostics } });
}

/** Direct results preserve full structured output without manufacturing provider context. */
async function appendDirectToolResult(
	tx: Tx,
	conversationId: ConversationId,
	sourceEntryId: EntryId,
	call: ToolCall,
	result: ToolExecutionResult,
	durationMs?: number,
) {
	if (result.usage !== undefined) await recordUsage(tx, conversationId, "tools", call.name, result.usage);
	return tx.appendEntry(DirectToolResultEntry, conversationId, {
		data: {
			sourceEntryId,
			callId: call.id,
			name: call.name,
			...(durationMs === undefined ? {} : { durationMs }),
			result: copyJson(result as JsonValue, { omitUndefinedProperties: true }),
		},
	});
}

function renderDiagnostics(diagnostics: readonly ToolDiagnostic[]): string {
	return `<harness>\n${diagnostics.map((diagnostic) => `[${diagnostic.severity}] ${diagnostic.message}`).join("\n")}\n</harness>`;
}

/**
 * Bound the text of result content. When the joined text exceeds the limits, the text items are replaced by one bounded
 * item at the position of the first (head) or last (tail) text item; other content is kept.
 */
function boundContent(
	content: Content,
	limits: OutputLimits,
): { content: Content; droppedBytes: number; droppedLines: number } {
	const texts = content.filter((item): item is TextContent => item.type === "text");
	const bounded = boundOutput(texts.map((item) => item.text).join(""), limits);
	if (bounded.droppedBytes === 0) return { content, droppedBytes: 0, droppedLines: 0 };
	const keep = limits.retain === "head" ? texts[0] : texts.at(-1);
	const result: Content = [];
	for (const item of content) {
		if (item.type !== "text") result.push(item);
		else if (item === keep) result.push({ ...item, text: bounded.text });
	}
	return { content: result, droppedBytes: bounded.droppedBytes, droppedLines: bounded.droppedLines };
}

function errorText(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/** A detached offer observation stays immutable even when an authoring wrapper spreads the API. */
function immutableData(value: JsonValue): JsonValue {
	if (value !== null && typeof value === "object") {
		for (const child of Object.values(value)) immutableData(child);
		Object.freeze(value);
	}
	return value;
}
