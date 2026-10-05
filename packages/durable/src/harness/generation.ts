import { type Context, copyJson, type Draft, type JsonValue } from "@earendil-works/chord";
import type {
	Api,
	AssistantMessage,
	DeferredHandle,
	Message,
	Model,
	ModelThinkingLevel,
	SimpleStreamOptions,
	ToolCall,
} from "@earendil-works/pi-ai";
import { isContextOverflow } from "@earendil-works/pi-ai/utils/overflow";
import { isRetryableAssistantError, retryDelayMs } from "@earendil-works/pi-ai/utils/retry";
import { AssistantEntry, ResetEntry, SystemEntry, UserEntry } from "../entries.ts";
import type { ExecutionEnv } from "../env/index.ts";
import { defineTask } from "../tasks.ts";
import type {
	ConversationId,
	EntryId,
	NextTaskState,
	SubmissionId,
	TaskId,
	TaskRuntime,
	Tx,
	TypedEntry,
} from "../types.ts";
import { addTools } from "./agent.ts";
import { createCompaction, estimateContext, selectCut } from "./compaction.ts";
import { bindTool } from "./define.ts";
import { applyBoundary, prepareBoundary } from "./inbox.ts";
import { assignJson, jsonEqual, type PinnedMessages, type PinnedModel, pinMessages, pinModel } from "./json.ts";
import { endRun, LiveDoc, type LiveState, type ToolSlot } from "./live.ts";
import { planSystemEntries, renderSections, replaySections } from "./prompt.ts";
import { ensureProviderSessionId } from "./provider.ts";
import { appendToolResult, harnessError, ToolTask, type ToolTaskResult } from "./tool.ts";
import type {
	CompactionPolicy,
	CompactionResult,
	ContextView,
	ConversationStreamOptions,
	GenerationHooks,
	GenerationResponseRequest,
	GenerationRetrySelection,
	ModelRef,
	ModelRequestResult,
	PromptInput,
	ToolBinding,
	ToolControl,
	UserInput,
} from "./types.ts";
import { recordUsage } from "./usage.ts";

export type GenerationInput = Record<string, never>;

/** Preparation fixes the endpoint, provider conversion policy, request options and offered executable bindings. */
type PreparedRequest = {
	attempt: number;
	compacted?: TaskId<CompactionResult>;
	model: PinnedModel;
	thinkingLevel: ModelThinkingLevel;
	streamOptions: ConversationStreamOptions;
	/** Newest entry included in the request. */
	cutoff: EntryId;
	offered: ToolBinding[];
};

export type GenerationCheckpoint =
	| {
			phase: "prepare";
			attempt: number;
			/** The blocking compaction this generation waited for; it starts no other compaction (spec §8.3). */
			compacted?: TaskId<CompactionResult>;
			/** Error text of the overflow that started `compacted`; checked once when `prepare` resumes. */
			overflow?: string;
	  }
	| ({ phase: "bind" } & PreparedRequest)
	| ({ phase: "request"; messages: PinnedMessages } & PreparedRequest)
	| { phase: "retry"; attempt: number; compacted?: TaskId<CompactionResult>; until: number }
	| ({
			phase: "poll";
			messages: PinnedMessages;
			handle: DeferredHandle;
			pollAt: number;
	  } & PreparedRequest)
	| {
			/** Waiting on the round's tool tasks, which the generation owns (spec §8.5). */
			phase: "tools";
			/** The tool-calling answer. */
			assistant: EntryId;
			/** Tool tasks created so far, in call order, across the committed waves. */
			tools: TaskId<ToolTaskResult>[];
			/** Ordered waves not yet admitted: parallel groups separated by singleton barriers. */
			pending: string[][];
			bindings: ToolBinding[];
	  };

export type GenerationResult = { entryId: EntryId };

type Runtime = TaskRuntime<GenerationInput, GenerationCheckpoint, GenerationResult, GenerationHooks>;
type Next = NextTaskState<GenerationCheckpoint, GenerationResult>;

/** What classification needs from the request that produced a message. */
type Request = PreparedRequest & {
	readonly messages: PinnedMessages;
	/** Set when the message came from polling, so a still deferred result polls strictly later. */
	readonly pollAt?: number;
};

const DEFAULT_POLL_AFTER_MS = 5000;

/**
 * Built-in generation task: prepares the positional system prompt and tool loadout, requests or polls the model,
 * retries, and classifies the response. The run's inputs live in `pi.live.run`.
 */
export const GenerationTask = defineTask<GenerationInput, GenerationCheckpoint, GenerationResult, GenerationHooks>({
	name: "pi.generation",
	version: 2,
	initial: () => ({ phase: "prepare", attempt: 1 }),
	phases: {
		/**
		 * Render the system prompt and tool loadout and append the positional `pi.system` entries they need, then move to
		 * `request`. The agent and settings resolved here are fixed for this request. Only the Harness writes to a busy
		 * conversation, so the transcript read here is still the tail at the commit.
		 */
		prepare: async (task, runtime, context) => {
			const { conversationId } = runtime;
			const agent = await runtime.agent(context);
			const settings = runtime.settings;
			const live = await runtime.snapshot(LiveDoc, conversationId, context);
			const selection = live?.run?.taskId === runtime.taskId ? live.run.requestSelection : undefined;
			const model = selection?.model ?? agent.model;
			const thinkingLevel = selection?.thinkingLevel ?? agent.thinkingLevel;
			const streamOptions = selection?.stream ?? agent.stream;
			const resolved = model === undefined ? undefined : runtime.models.getModel(model.provider, model.modelId);
			if (model === undefined || resolved === undefined) return failNoModel(runtime, model, context);
			const descriptor = pinModel(resolved);
			const { attempt, compacted, overflow } = task.state.checkpoint;
			if (compacted !== undefined && overflow !== undefined) {
				const [outcome] = await runtime.outcomes([compacted], context);
				if (outcome?.status !== "completed" || outcome.result.entryId === undefined) {
					return failModelError(runtime, overflow, context);
				}
			}
			const view = await runtime.context(conversationId, context);
			const shown = replaySections(view.messages);
			const report = (error: unknown) => runtime.report(error);
			let env: ExecutionEnv | undefined;
			try {
				env = await runtime.env(context);
			} catch (error) {
				if (context.abortSignal?.aborted) throw error;
				report(error);
			}
			// Execution selection remains intact for genuine direct calls. The descriptor fixes what this model can receive.
			const offeredTools = descriptor.capabilities?.tools === false ? [] : agent.tools;
			const input: PromptInput = {
				conversationId,
				agent: { ...agent, tools: offeredTools },
				env,
				shown: Object.fromEntries(shown),
				read: runtime,
			};
			const desired = await renderSections(agent.sections, input, shown, report, context);
			const entries = planSystemEntries(view, desired, offeredTools, runtime.now());
			const threshold =
				compacted === undefined
					? thresholdCompaction(view, entries, descriptor.contextWindow, settings.compaction)
					: undefined;
			if (threshold === "blocking") {
				// Compact first and prepare again; the transcript is unchanged until the compaction appends.
				await runtime.commit(async (tx): Promise<Next> => {
					const child = await createCompaction(tx, conversationId, { reason: "threshold" }, runtime.taskId);
					const checkpoint = { phase: "prepare", attempt, compacted: child } as const;
					return {
						status: "waiting",
						checkpoint,
						condition: { kind: "tasks", on: [child], policy: "allSettled" },
					};
				}, context);
				return;
			}
			await runtime.commit(async (tx) => {
				let cutoff = (await tx.scanEntries({ conversationId }, 1)).items[0]?.id;
				for (const entry of entries) cutoff = (await tx.appendEntry(SystemEntry, conversationId, entry)).id;
				if (cutoff === undefined) throw new Error(`Conversation ${conversationId} has no entries to send`);
				// Checked in this commit, so a compaction admitted during preparation counts.
				if (threshold === "background" && (await tx.doc(LiveDoc, conversationId)).compactions === undefined) {
					await createCompaction(tx, conversationId, { reason: "threshold" });
				}
				const request = {
					attempt,
					...(compacted === undefined ? {} : { compacted }),
					model: descriptor,
					thinkingLevel,
					streamOptions,
					offered: offeredTools.map((tool) => bindTool(tool, settings.toolExecution)),
					cutoff,
				};
				return { status: "running", checkpoint: { phase: "bind", ...request } };
			}, context);
		},
		bind: async (task, runtime, context) => {
			const { phase: _, ...request } = task.state.checkpoint;
			const view = await runtime.context(runtime.conversationId, context, request.cutoff);
			let messages = view.messages;
			await runtime.hooks.each("beforeRequest", async (hook) => {
				const replaced = await hook({ messages }, runtime, context);
				if (replaced !== undefined) messages = replaced.messages;
			});
			const input = pinMessages(messages);
			await runtime.commit(
				() => ({ status: "running", checkpoint: { phase: "request", ...request, messages: input } }),
				context,
			);
		},
		request: async (task, runtime, context) => {
			const { phase: _, ...request } = task.state.checkpoint;
			const { attempt, model, thinkingLevel, streamOptions, messages } = request;
			const conversationId = runtime.conversationId;
			await runtime.commit(async (tx) => {
				const live = await tx.doc(LiveDoc, conversationId);
				await convertPartial(tx, live, conversationId);
				live.generation = { attempt };
				return undefined;
			}, context);
			if (runtime.models.getProvider(model.provider) === undefined)
				return failNoModel(runtime, { provider: model.provider, modelId: model.id }, context);
			const options = {
				...streamOptions,
				sessionId: await ensureProviderSessionId(runtime, context),
				...(thinkingLevel === "off" ? {} : { reasoning: thinkingLevel }),
			} satisfies SimpleStreamOptions;
			const response = await runtime.withModelRequest(
				{ purpose: "generation", operation: "stream", attempt, model, messages, cutoff: request.cutoff, options },
				(capabilities, signal, prepared) =>
					streamResponse(
						runtime,
						prepared.model,
						messages,
						{ ...options, ...capabilities, signal },
						attempt,
						context,
					),
				context,
			);
			if (response.status === "waiting") {
				await runtime.commit(
					() => ({ status: "waiting", checkpoint: task.state.checkpoint, condition: response.condition }),
					context,
				);
				return;
			}
			await classify(runtime, request, response.result, context);
		},
		retry: async (task, runtime, context) => {
			const { attempt, compacted, until } = task.state.checkpoint;
			if (runtime.now() < until) {
				await runtime.commit(
					() => ({ status: "waiting", checkpoint: task.state.checkpoint, condition: { kind: "time", until } }),
					context,
				);
				return;
			}
			await runtime.commit(async (tx) => {
				(await tx.doc(LiveDoc, runtime.conversationId)).generation = { attempt: attempt + 1 };
				const checkpoint: GenerationCheckpoint = {
					phase: "prepare",
					attempt: attempt + 1,
					...(compacted === undefined ? {} : { compacted }),
				};
				return { status: "running", checkpoint };
			}, context);
		},
		poll: async (task, runtime, context) => {
			const { phase: _, ...request } = task.state.checkpoint;
			const { model, handle, pollAt, streamOptions } = request;
			if (runtime.now() < pollAt) {
				await runtime.commit(
					() => ({
						status: "waiting",
						checkpoint: task.state.checkpoint,
						condition: { kind: "time", until: pollAt },
					}),
					context,
				);
				return;
			}
			let response: ModelRequestResult<AssistantMessage>;
			try {
				response = await runtime.withModelRequest(
					{
						purpose: "generation",
						operation: "fetchDeferred",
						attempt: request.attempt,
						model,
						messages: request.messages,
						cutoff: request.cutoff,
						options: streamOptions,
						handle,
					},
					(capabilities, signal, prepared) =>
						runtime.models.fetchDeferred(prepared.model, handle, {
							...streamOptions,
							...capabilities,
							signal,
						}),
					context,
				);
			} catch (error) {
				if (runtime.signal.aborted) throw error;
				await runtime.parkFailure(error, context);
				return;
			}
			if (response.status === "waiting") {
				await runtime.commit(
					() => ({ status: "waiting", checkpoint: task.state.checkpoint, condition: response.condition }),
					context,
				);
				return;
			}
			const message = response.result;
			// A failed observation is not proof that the admitted remote operation ended. Retain its handle;
			// repair polls that same operation, while explicit cancellation still invokes its cancel contract.
			if (message.stopReason === "error" || message.stopReason === "aborted") {
				const error = new Error(message.errorMessage ?? `Deferred polling ended with ${message.stopReason}`);
				await runtime.parkFailure(error, context, async (tx) => {
					await appendAssistant(tx, runtime.conversationId, message);
				});
				return;
			}
			await classify(runtime, request, message, context);
		},
		tools: async (task, runtime, context) => {
			const { assistant, tools, pending, bindings } = task.state.checkpoint;
			const [next, ...rest] = pending;
			if (next === undefined) return finishToolRound(runtime, assistant, tools, context);
			// The prior wave is terminal; admit the next wave atomically before any call executes.
			await runtime.commit(async (tx): Promise<Next> => {
				const live = await tx.doc(LiveDoc, runtime.conversationId);
				const admitted = await admitToolWave(tx, runtime, assistant, next, live.tools ?? [], bindings);
				const checkpoint: GenerationCheckpoint = {
					phase: "tools",
					assistant,
					tools: [...tools, ...admitted],
					pending: rest,
					bindings,
				};
				return { status: "waiting", checkpoint, condition: { kind: "tasks", on: admitted, policy: "allSettled" } };
			}, context);
		},
	},
	abort: async (task, runtime, context) => {
		const checkpoint = task.state.checkpoint;
		if (checkpoint.phase === "poll") {
			const model = checkpoint.model;
			const response = await runtime.withModelRequest(
				{
					purpose: "generation",
					operation: "cancelDeferred",
					attempt: checkpoint.attempt,
					model,
					messages: checkpoint.messages,
					cutoff: checkpoint.cutoff,
					options: checkpoint.streamOptions,
					handle: checkpoint.handle,
				},
				(capabilities, signal, prepared) =>
					runtime.models.cancelDeferred(prepared.model, checkpoint.handle, {
						...checkpoint.streamOptions,
						...capabilities,
						signal,
					}),
				context,
			);
			if (response.status === "waiting") {
				await runtime.commit(() => ({ status: "waiting", checkpoint, condition: response.condition }), context);
				return;
			}
		}
		const conversationId = runtime.conversationId;
		// Runs after the round's tool tasks are terminal; calls never started get `aborted` results (spec §8.5).
		const unstarted =
			checkpoint.phase === "tools"
				? await readCalls(runtime, checkpoint.assistant, checkpoint.pending.flat(), context)
				: [];
		await runtime.commit(async (tx) => {
			const boundary = await prepareBoundary(tx, conversationId, {
				...runtime.settings,
				followUpMode: "one-at-a-time",
			});
			const live = await tx.doc(LiveDoc, conversationId);
			await convertPartial(tx, live, conversationId);
			for (const call of unstarted) {
				const result = harnessError("aborted", `Tool ${call.name} was aborted`);
				await appendToolResult(tx, conversationId, call, result, runtime.now());
			}
			if (live.run?.taskId === runtime.taskId && live.run.interruption?.kind === "flush") {
				const steering = boundary.inbox.items.some((item) => item.mode === "steer");
				const { users, reset } = await applyBoundary(tx, boundary, steering ? "postTools" : "final", runtime.now());
				if (steering && users.length > 0 && !reset) {
					live.run.inputs.push(...users);
					delete live.run.interruption;
					delete live.generation;
					delete live.tools;
					handOver(live, runtime.taskId, await createGeneration(tx, conversationId));
				} else {
					endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "aborted" });
					if (users.length > 0) await startRun(tx, conversationId, live, users);
				}
			} else endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "aborted" });
			return { status: "terminal", outcome: { status: "aborted" } };
		}, context);
	},
});

/** The calls `callIds` of the assistant entry, in the given order. */
async function readCalls(
	runtime: Runtime,
	assistant: EntryId,
	callIds: readonly string[],
	context: Context,
): Promise<ToolCall[]> {
	const message = (await runtime.entry(AssistantEntry, assistant, context))?.model?.[0];
	const calls = message?.role === "assistant" ? message.content.filter((content) => content.type === "toolCall") : [];
	return callIds.flatMap((id) => calls.filter((call) => call.id === id).slice(0, 1));
}

/** A tool task for call `callId`, owned by the generation. */
function createToolTask(
	tx: Tx,
	runtime: Runtime,
	assistant: EntryId,
	callId: string,
	binding: ToolBinding,
): Promise<TaskId<ToolTaskResult>> {
	return tx.createTask(
		ToolTask,
		{ source: { kind: "assistant" as const, entryId: assistant }, callId, binding },
		{ ownership: { kind: "task", taskId: runtime.taskId } },
	);
}

/**
 * Which threshold compaction preparation starts before its request (spec §8.3): `blocking` above
 * `contextWindow - reserveTokens`, `background` above the background threshold, and only when range selection finds a
 * cut. The caller starts a background one only while no compaction is listed.
 */
function thresholdCompaction(
	view: ContextView,
	planned: readonly { readonly model?: readonly Message[] }[],
	contextWindow: number,
	policy: CompactionPolicy,
): "blocking" | "background" | undefined {
	if (!policy.enabled || contextWindow <= 0) return undefined;
	const tokens = estimateContext(
		view,
		planned.flatMap((entry) => entry.model ?? []),
	);
	const blocking = contextWindow - policy.reserveTokens;
	const background = blocking - policy.backgroundTokens;
	const over =
		tokens > blocking ? "blocking" : policy.backgroundTokens > 0 && tokens > background ? "background" : undefined;
	if (over === undefined || selectCut(view, policy.keepRecentTokens) === undefined) return undefined;
	return over;
}

/** Settle the run's inputs `unanswered` with `model_error` and fail with `text`. */
async function failModelError(runtime: Runtime, text: string, context: Context): Promise<void> {
	await runtime.commit(async (tx) => {
		const live = await tx.doc(LiveDoc, runtime.conversationId);
		endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "model_error", detail: text });
		return {
			status: "terminal",
			outcome: { status: "failed", error: { message: text, detail: { reason: "model_error" } } },
		};
	}, context);
}

/** Settle the run's inputs `unanswered` with `no_model` and fail. */
async function failNoModel(runtime: Runtime, ref: ModelRef | undefined, context: Context): Promise<void> {
	const message =
		ref === undefined ? "No model is configured" : `Model ${ref.provider}/${ref.modelId} is not available`;
	await runtime.commit(async (tx) => {
		const live = await tx.doc(LiveDoc, runtime.conversationId);
		endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "no_model" });
		return { status: "terminal", outcome: { status: "failed", error: { message, detail: { reason: "no_model" } } } };
	}, context);
}

/**
 * Append a committed partial left by an interrupted, aborted, faulted, or orphaned attempt as an aborted assistant
 * entry; the caller replaces or removes `generation`.
 */
export async function convertPartial(tx: Tx, live: Draft<LiveState>, conversationId: ConversationId): Promise<void> {
	const partial = live.generation?.message;
	if (partial === undefined) return;
	const message = copyJson(partial) as unknown as AssistantMessage;
	await appendAssistant(tx, conversationId, { ...message, stopReason: "aborted" });
}

/**
 * Stream one request and return the terminal message. Partials commit as trailing writes at most every
 * `progress.partialIntervalMs` (default 100 ms) with one commit in flight; `finally` stops the throttle and awaits that
 * commit, so no stale partial lands after the outcome.
 */
async function streamResponse(
	runtime: Runtime,
	model: Model<Api>,
	messages: readonly Message[],
	options: SimpleStreamOptions,
	attempt: number,
	context: Context,
): Promise<AssistantMessage> {
	const interval = runtime.settings.progress.partialIntervalMs;
	let pending: AssistantMessage | undefined;
	let timer: ReturnType<typeof setTimeout> | undefined;
	let inFlight: Promise<void> | undefined;
	let stopped = false;
	const flush = (): void => {
		timer = undefined;
		const partial = pending;
		pending = undefined;
		if (partial === undefined || stopped) return;
		inFlight = (async () => {
			// Copy synchronously: the provider keeps mutating its partial.
			const message = copyJson(partial, { omitUndefinedProperties: true });
			await runtime.commit(async (tx) => {
				const live = await tx.doc(LiveDoc, runtime.conversationId);
				live.generation ??= { attempt };
				assignJson(live.generation as Draft<Record<string, JsonValue>>, "message", message);
				return undefined;
			}, context);
		})()
			.catch((error: unknown) => {
				// Rejections after an abort mark or close are expected; the committed state stays consistent.
				if (!runtime.signal.aborted) runtime.report(error);
			})
			.finally(() => {
				inFlight = undefined;
				if (pending !== undefined && !stopped) timer = setTimeout(flush, interval);
			});
	};
	try {
		const events = runtime.models.streamSimple(model, { messages: [...messages] }, options);
		for await (const event of events) {
			// A partial without content, such as pi-ai's opening `start` event, shows nothing; a deferred response
			// never gets past it, so it never leaves a partial.
			if (event.type === "done" || event.type === "error" || event.partial.content.length === 0) continue;
			pending = event.partial;
			if (timer === undefined && inFlight === undefined) timer = setTimeout(flush, interval);
		}
		return await events.result();
	} finally {
		stopped = true;
		clearTimeout(timer);
		await inFlight;
	}
}

/** Classify a terminal provider message in one commit that also clears the partial. */
async function classify(
	runtime: Runtime,
	request: Request,
	message: AssistantMessage,
	context: Context,
): Promise<void> {
	// An abort mark or close: the abort invocation or the reopened run handles the committed state.
	runtime.signal.throwIfAborted();
	const conversationId = runtime.conversationId;
	const { attempt, compacted, cutoff } = request;
	if (message.stopReason === "deferred" && message.deferred !== undefined) {
		const handle = message.deferred;
		const pollAt = Math.max(
			runtime.now() + (handle.pollAfterMs ?? DEFAULT_POLL_AFTER_MS),
			request.pollAt === undefined ? Number.NEGATIVE_INFINITY : request.pollAt + 1,
		);
		await runtime.commit(async (tx) => {
			(await tx.doc(LiveDoc, conversationId)).generation = { attempt, deferred: { pollAt } };
			const checkpoint = {
				phase: "poll",
				...request,
				handle,
				pollAt,
			} as const;
			return { status: "running", checkpoint };
		}, context);
		return;
	}
	let selected: GenerationRetrySelection | undefined;
	await runtime.hooks.each("afterResponse", async (hook) => {
		const decision = await hook(
			message,
			runtime,
			context,
			copyJson({
				attempt,
				model: request.model,
				thinkingLevel: request.thinkingLevel,
				streamOptions: request.streamOptions,
				cutoff,
			}) as GenerationResponseRequest,
		);
		if (selected === undefined && decision?.retry !== undefined) selected = decision.retry;
	});
	if (selected !== undefined) {
		const selection = copyJson(
			{
				model: selected.model,
				thinkingLevel: selected.thinkingLevel ?? request.thinkingLevel,
				stream:
					selected.stream === undefined ? request.streamOptions : { ...request.streamOptions, ...selected.stream },
			},
			{ omitUndefinedProperties: true },
		) as GenerationRetrySelection;
		const unchanged = jsonEqual(selection, {
			model: { provider: request.model.provider, modelId: request.model.id },
			thinkingLevel: request.thinkingLevel,
			stream: request.streamOptions,
		});
		const invalid =
			message.stopReason !== "error"
				? "Model retry selection requires an actual provider error"
				: unchanged
					? "Model retry selection must change the failed request policy"
					: undefined;
		await runtime.commit(async (tx): Promise<Next> => {
			const live = await tx.doc(LiveDoc, conversationId);
			if (live.run?.taskId !== runtime.taskId) throw new Error("Model retry selection has no owning input run");
			await appendAssistant(tx, conversationId, message);
			if (invalid !== undefined) {
				endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "model_error", detail: invalid });
				return {
					status: "terminal",
					outcome: {
						status: "failed",
						error: {
							message: invalid,
							detail: { reason: "model_policy_error", providerError: message.errorMessage ?? null },
						},
					},
				};
			}
			live.run.requestSelection = selection;
			delete live.generation;
			return {
				status: "running",
				checkpoint: { phase: "prepare", attempt: attempt + 1, ...(compacted === undefined ? {} : { compacted }) },
			};
		}, context);
		return;
	}
	const calls = message.content.filter((content): content is ToolCall => content.type === "toolCall");
	if (message.stopReason === "toolUse" && calls.length > 0) {
		return startToolRound(runtime, request, message, calls, context);
	}
	if (message.stopReason === "stop" || message.stopReason === "length" || message.stopReason === "toolUse") {
		return answer(runtime, message, context);
	}
	// The retry and compaction policies govern the next attempt, so they are read now rather than pinned at preparation.
	const settings = runtime.settings;
	const overflow = message.stopReason === "error" && isContextOverflow(message);
	if (overflow && compacted === undefined && settings.compaction.enabled) {
		const policy = settings.compaction;
		const view = await runtime.context(conversationId, context, cutoff);
		if (selectCut(view, policy.keepRecentTokens) !== undefined) {
			const text = message.errorMessage ?? "Context overflow";
			await runtime.commit(async (tx): Promise<Next> => {
				const live = await tx.doc(LiveDoc, conversationId);
				await appendAssistant(tx, conversationId, message);
				delete live.generation;
				const child = await createCompaction(tx, conversationId, { reason: "overflow" }, runtime.taskId);
				const checkpoint = { phase: "prepare", attempt, compacted: child, overflow: text } as const;
				return { status: "waiting", checkpoint, condition: { kind: "tasks", on: [child], policy: "allSettled" } };
			}, context);
			return;
		}
	}
	const policy = settings.retry;
	// An overflow is never retried: only a compaction can make the next request fit.
	const retry =
		message.stopReason === "error" &&
		!overflow &&
		isRetryableAssistantError(message) &&
		policy.enabled &&
		attempt <= policy.maxRetries;
	const until = retry ? runtime.now() + retryDelayMs(policy, attempt) : 0;
	await runtime.commit(async (tx): Promise<Next> => {
		const live = await tx.doc(LiveDoc, conversationId);
		await appendAssistant(tx, conversationId, message);
		if (retry) {
			live.generation = { attempt, retry: { at: until, error: message.errorMessage ?? "" } };
			const checkpoint = {
				phase: "retry",
				attempt,
				...(compacted === undefined ? {} : { compacted }),
				until,
			} as const;
			return { status: "running", checkpoint };
		}
		const text = message.errorMessage ?? `Model response ended with stop reason ${message.stopReason}`;
		endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "model_error", detail: text });
		return {
			status: "terminal",
			outcome: { status: "failed", error: { message: text, detail: { reason: "model_error" } } },
		};
	}, context);
}

/**
 * A final answer; the final boundary places queued items (spec §6). The first `onYield` continuation appends a user
 * message and hands the run to a successor generation, but only when the boundary selected no user item and no reset.
 * Otherwise the run's inputs settle `done`, and selected user items start the next run.
 */
async function answer(runtime: Runtime, message: AssistantMessage, context: Context): Promise<void> {
	let continuation: UserInput | undefined;
	await runtime.hooks.each("onYield", async (hook) => {
		if (continuation !== undefined) return;
		continuation = (await hook(message, runtime, context))?.continue;
	});
	const conversationId = runtime.conversationId;
	await runtime.commit(async (tx): Promise<Next> => {
		// Queue modes are read on the Session line, when the boundary is decided.
		const boundary = await prepareBoundary(tx, conversationId, runtime.settings);
		const live = await tx.doc(LiveDoc, conversationId);
		const entry = await appendAssistant(tx, conversationId, message);
		const result: Next = { status: "terminal", outcome: { status: "completed", result: { entryId: entry.id } } };
		const { users, reset } = await applyBoundary(tx, boundary, "final", runtime.now());
		if (continuation !== undefined && users.length === 0 && !reset) {
			const user = { role: "user", content: continuation, timestamp: runtime.now() } as const;
			await tx.appendEntry(UserEntry, conversationId, { model: [user] });
			handOver(live, runtime.taskId, await createGeneration(tx, conversationId));
			delete live.generation;
			return result;
		}
		endRun(tx, live, runtime.taskId, { status: "done", answer: entry.id });
		if (users.length > 0) await startRun(tx, conversationId, live, users);
		return result;
	}, context);
}

/**
 * Append the tool-calling answer and start its tool round in one commit (spec §8.3). A call to a tool the request did
 * not offer gets its `tool_unavailable` result here; every other call gets a tool task owned by the generation, only the
 * first wave now. Parallel-safe groups and singleton barriers are committed in call order; generation waits for each wave.
 */
async function startToolRound(
	runtime: Runtime,
	request: Request,
	message: AssistantMessage,
	calls: readonly ToolCall[],
	context: Context,
): Promise<void> {
	const conversationId = runtime.conversationId;
	const bindings = request.offered;
	const offered = new Map(bindings.map((binding) => [binding.name, binding]));
	const waves: string[][] = [];
	let parallel: string[] = [];
	for (const call of calls) {
		if (!offered.has(call.name)) continue;
		const sequential = offered.get(call.name)?.executionMode === "sequential";
		if (sequential) {
			if (parallel.length > 0) waves.push(parallel);
			parallel = [];
			waves.push([call.id]);
		} else parallel.push(call.id);
	}
	if (parallel.length > 0) waves.push(parallel);
	await runtime.commit(async (tx): Promise<Next> => {
		const live = await tx.doc(LiveDoc, conversationId);
		const entry = await appendAssistant(tx, conversationId, message);
		const slots: ToolSlot[] = [];

		for (const call of calls) {
			if (!offered.has(call.name)) {
				const unavailable = harnessError("tool_unavailable", `Tool ${call.name} is not available`);
				const result = await appendToolResult(tx, conversationId, call, unavailable, runtime.now());
				slots.push({ callId: call.id, name: call.name, status: "done", entry: result.id });
				continue;
			}
			slots.push({ callId: call.id, name: call.name, status: "pending" });
		}
		const [first = [], ...pending] = waves;
		const tools = await admitToolWave(tx, runtime, entry.id, first, slots, bindings);
		delete live.generation;
		live.tools = slots;
		const checkpoint = { phase: "tools", assistant: entry.id, tools, pending, bindings } as const;
		return { status: "waiting", checkpoint, condition: { kind: "tasks", on: tools, policy: "allSettled" } };
	}, context);
}

/** Create exactly one committed wave. The generation's checkpoint is the only admission cursor. */
async function admitToolWave(
	tx: Tx,
	runtime: Runtime,
	assistant: EntryId,
	calls: readonly string[],
	slots: Draft<ToolSlot>[],
	bindings: readonly ToolBinding[],
): Promise<TaskId<ToolTaskResult>[]> {
	const admitted: TaskId<ToolTaskResult>[] = [];
	for (const callId of calls) {
		const slot = slots.find((slot) => slot.callId === callId);
		if (slot === undefined || slot.taskId !== undefined)
			throw new Error(`Tool wave has invalid admission slot ${callId}`);
		const binding = bindings.find((binding) => binding.name === slot.name);
		if (binding === undefined) throw new Error(`Tool wave has no offered binding for ${slot.name}`);
		const taskId = await createToolTask(tx, runtime, assistant, callId, binding);
		slot.taskId = taskId;
		admitted.push(taskId);
	}
	return admitted;
}

/**
 * The round's tools are terminal: apply their controls and either end the run at the final boundary (`terminate`,
 * `handoff`, or a queued reset) or hand it to the next generation at the `postTools` boundary (spec §8.5).
 */
async function finishToolRound(
	runtime: Runtime,
	assistant: EntryId,
	tools: readonly TaskId<ToolTaskResult>[],
	context: Context,
): Promise<void> {
	const conversationId = runtime.conversationId;
	const controls = new Map<TaskId, ToolControl | undefined>();
	const outcomes = await runtime.outcomes(tools, context);
	tools.forEach((id, index) => {
		const outcome = outcomes[index]!;
		controls.set(id, outcome.status === "completed" ? outcome.result.control : undefined);
	});
	const slots = (await runtime.snapshot(LiveDoc, conversationId, context))?.tools ?? [];
	const results = slots.flatMap((slot) => (slot.entry === undefined ? [] : [slot.entry]));
	await runtime.hooks.each("afterTools", (hook) => hook(assistant, results, runtime, context));
	// Every call of the round, including those answered without a task, must ask to terminate.
	const terminate =
		slots.length > 0 &&
		slots.every((slot) => slot.taskId !== undefined && controls.get(slot.taskId)?.terminate === true);
	const added = [...controls.values()].flatMap((control) => control?.addTools ?? []);
	// The last handoff in call order wins.
	const handoff = [...controls.values()].findLast((control) => control?.handoff !== undefined)?.handoff;
	await runtime.commit(async (tx): Promise<Next> => {
		const boundary = await prepareBoundary(tx, conversationId, runtime.settings);
		if (added.length > 0) await addTools(tx, conversationId, added);
		const live = await tx.doc(LiveDoc, conversationId);
		const now = runtime.now();
		if (terminate || handoff !== undefined) {
			if (handoff !== undefined) {
				const message = { role: "user", content: handoff, timestamp: now } as const;
				const entry = await tx.appendEntry(ResetEntry, conversationId, { head: "self", model: [message] });
				boundary.head = entry.id;
			}
			const { users } = await applyBoundary(tx, boundary, "final", now);
			endRun(tx, live, runtime.taskId, { status: "done", answer: assistant });
			if (users.length > 0) await startRun(tx, conversationId, live, users);
		} else {
			const { users, reset } = await applyBoundary(tx, boundary, "postTools", now);
			if (reset) {
				// The queued reset cut the run's context before an answer.
				endRun(tx, live, runtime.taskId, { status: "unanswered", reason: "reset" });
				if (users.length > 0) await startRun(tx, conversationId, live, users);
			} else {
				delete live.tools;
				if (live.run?.taskId === runtime.taskId) live.run.inputs.push(...users);
				handOver(live, runtime.taskId, await createGeneration(tx, conversationId));
			}
		}
		return { status: "terminal", outcome: { status: "completed", result: { entryId: assistant } } };
	}, context);
}

/**
 * Append a provider result and add its usage to `pi.usage` in the same commit.
 * REMINDER: every built-in writer of assistant entries goes through here, so the usage ledger stays complete.
 */
async function appendAssistant(
	tx: Tx,
	conversationId: ConversationId,
	message: AssistantMessage,
): Promise<TypedEntry<never>> {
	await recordUsage(tx, conversationId, "models", `${message.provider}/${message.model}`, message.usage);
	return tx.appendEntry(AssistantEntry, conversationId, { model: [message] });
}

/** Start a run for `inputs`, placed input submissions: a new generation takes `pi.live.run`. */
export async function startRun(
	tx: Tx,
	conversationId: ConversationId,
	live: Draft<LiveState>,
	inputs: SubmissionId[],
): Promise<void> {
	live.run = { taskId: await createGeneration(tx, conversationId), inputs };
}

/** A generation owned by its conversation. */
function createGeneration(tx: Tx, conversationId: ConversationId): Promise<TaskId<GenerationResult>> {
	return tx.createTask(GenerationTask, {}, { ownership: { kind: "conversation" }, conversationId });
}

/** Hand run control from `from` to `to`; the run's inputs move with it. */
export function handOver(live: Draft<LiveState>, from: TaskId, to: TaskId): void {
	if (live.run?.taskId === from) live.run.taskId = to;
}
