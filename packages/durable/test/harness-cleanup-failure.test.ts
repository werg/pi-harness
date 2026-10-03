import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@earendil-works/chord";
import { createModels, fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	bindReceipt,
	defineExtension,
	defineTask,
	defineTool,
	type EntryId,
	Harness,
	type Id,
	LiveDoc,
	MemoryStorage,
	type ModelRequestPort,
	StorageRejected,
	type StorageWrite,
	type TaskId,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { type ChatSetup, chatSetup, toolsNamed } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";
import { openTasks } from "./task-support.ts";

const harnesses = new Set<Harness>();
const directories = new Set<string>();

afterEach(async () => {
	const closed = await Promise.allSettled([...harnesses].map((harness) => harness.close(context)));
	harnesses.clear();
	for (const directory of directories) await rm(directory, { recursive: true, force: true });
	directories.clear();
	for (const result of closed) if (result.status === "rejected") throw result.reason;
});

async function path() {
	const directory = await mkdtemp(join(tmpdir(), "pi-cleanup-failure-"));
	directories.add(directory);
	return join(directory, "session.sqlite");
}

async function open(databasePath: string, setup: ChatSetup, modelRequests?: ModelRequestPort) {
	const harness = await Harness.open(
		await openNodeSqliteStorage(databasePath),
		{
			models: setup.models,
			registry: setup.registry,
			settings: setup.settings,
			now: () => 1_000,
			modelRequests,
			onReport: (error) => setup.reports.push(error),
		},
		context,
	);
	harnesses.add(harness);
	const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
	return { harness, root };
}

async function cleanupIncident(harness: Harness, taskId: TaskId): Promise<EntryId> {
	const task = await harness.getTask(taskId, context);
	expect(task).toMatchObject({ abortRequested: true, state: { status: "waiting", mode: "abort" } });
	if (task?.state.status !== "waiting" || task.state.condition.kind !== "failure")
		throw new Error("Expected owned cleanup failure");
	return task.state.condition.incident;
}

describe("failed cleanup ownership", () => {
	it("retains a failed tool cancellation and its operation continuation until exact repair after replacement", async () => {
		const databasePath = await path();
		const setup = chatSetup();
		const original = new Error("external operation cancellation refused");
		let repaired = false;
		const cancellations: unknown[] = [];
		const tool = defineTool({
			name: "owned-operation",
			description: "An external operation with durable cleanup",
			parameters: Type.Object({ value: Type.String() }),
			execute: async (_args, api, ctx) => {
				await api.commit((tx) => bindReceipt(tx, "operation", "accepted-operation"), ctx);
				api.output("retained operation output\n");
				return {
					wait: { kind: "receipt" as const, key: "operation", binding: "accepted-operation" },
					continuation: { operation: "accepted-operation" },
				};
			},
			cancel: async (args, api) => {
				cancellations.push(structuredClone({ args, continuation: api.continuation }));
				if (!repaired) throw original;
				return { content: [{ type: "text" as const, text: "cancelled operation" }] };
			},
		});
		setup.registry.install(defineExtension({ name: "cleanup-tool", tools: [tool] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall(tool.name, { value: "pinned" }, { id: "call" })], {
				stopReason: "toolUse",
			}),
		]);
		let opened = await open(databasePath, setup);
		await opened.root.configure({ tools: toolsNamed(setup, tool.name) }, context);
		const submission = await opened.root.submit({ type: "input", content: "run operation" }, context);
		await opened.harness.runPass(context);
		const inspection = await opened.harness.inspect(context);
		const task = inspection.tasks.find((item) => item.record.kind === "pi.tool")!.record;
		if (task.state.status !== "waiting") throw new Error("Expected retained operation wait");
		const checkpoint = task.state.checkpoint;
		await expect(opened.root.abort(context)).rejects.toBe(original);
		await opened.harness.runPass(context);
		const incident = await cleanupIncident(opened.harness, task.id);
		expect((await opened.harness.getTask(task.id, context))!.state).toMatchObject({ checkpoint });
		expect(await submission.status(context)).toMatchObject({ status: "placed" });
		expect((await opened.harness.snapshot(LiveDoc, opened.root.id, context))?.tools).toEqual([
			expect.objectContaining({ taskId: task.id, output: "retained operation output\n" }),
		]);
		expect(
			(await opened.root.entries({}, 100, undefined, context)).items.some(
				(entry) => entry.kind === "pi.tool-result",
			),
		).toBe(false);
		await opened.harness.close(context);
		opened = await open(databasePath, setup);
		await opened.harness.runPass(context);
		expect(await cleanupIncident(opened.harness, task.id)).toBe(incident);
		expect(cancellations).toHaveLength(1);
		await expect(opened.root.waitForIdle(context)).rejects.toThrow(original.message);
		repaired = true;
		expect(await opened.harness.retryTask(task.id, incident, context)).toBe("queued");
		await opened.harness.runPass(context);
		expect((await opened.harness.waitForTask(task.id, context)).state.outcome.status).toBe("aborted");
		expect(cancellations).toHaveLength(2);
		expect(cancellations[1]).toEqual(cancellations[0]);
		expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
			status: "unanswered",
			reason: "aborted",
		});
		const results = (await opened.root.entries({}, 100, undefined, context)).items.filter(
			(entry) => entry.kind === "pi.tool-result",
		);
		expect(results).toHaveLength(1);
		expect(results[0]!.model?.[0]).toMatchObject({
			role: "toolResult",
			content: [{ type: "text", text: "cancelled operation" }],
		});
	});

	it.each(["commit", "entry"] as const)(
		"stops dispatch and joins the caller when cleanup failure %s fails",
		async (stage) => {
			const original = new Error("resource cleanup failed");
			const writeFailure =
				stage === "commit"
					? new StorageRejected("cleanup failure storage rejected")
					: new Error("cleanup failure identity refused");
			let attempts = 0;
			const storage = new (class extends ControlledStorage {
				override mintId<I extends Id<string>>(): Promise<I> {
					if (stage === "entry" && attempts > 0) return Promise.reject(writeFailure);
					return super.mintId<I>();
				}
				override async commit(writes: readonly StorageWrite[], ctx: Context) {
					if (
						stage === "commit" &&
						writes.some((write) => write.type === "entry" && write.value.kind === "pi.failure")
					)
						throw writeFailure;
					return super.commit(writes, ctx);
				}
			})();
			const task = defineTask<null, { phase: "work" }, null>({
				name: "test.cleanup-write-failure",
				version: 1,
				initial: () => ({ phase: "work" }),
				phases: {
					work: (task, runtime, ctx) =>
						runtime.commit(
							() => ({
								status: "waiting",
								checkpoint: task.state.checkpoint,
								condition: { kind: "time", until: 100_000 },
							}),
							ctx,
						),
				},
				abort: async () => {
					attempts++;
					throw original;
				},
			});
			const { harness, reports } = await openTasks(storage, [task], { now: () => 1_000 });
			harnesses.add(harness);
			const root = await harness.root(context);
			const id = await root.commit(
				(tx) => tx.createTask(task, null, { ownership: { kind: "conversation" } }),
				context,
			);
			await harness.runPass(context);
			const caller = root.abort(context);
			await expect(caller).rejects.toMatchObject({ errors: [original, writeFailure] });
			await expect(harness.runPass(context)).rejects.toMatchObject({ errors: [original, writeFailure] });
			expect(await harness.getTask(id, context)).toMatchObject({
				abortRequested: true,
				state: { status: "running", checkpoint: { phase: "work" } },
			});
			expect(attempts).toBe(1);
			expect(reports).toContainEqual(expect.objectContaining({ errors: [original, writeFailure] }));
		},
	);

	it("keeps unrelated and background failures outside an ordinary conversation's idle scope", async () => {
		const failures = {
			foreign: new Error("foreign cleanup failed"),
			background: new Error("background cleanup failed"),
		};
		const task = defineTask<"foreign" | "background" | "complete", { phase: "work" }, null>({
			name: "test.scoped-cleanup",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async (task, runtime, ctx) => {
					await runtime.commit(
						() =>
							task.input === "complete"
								? { status: "terminal", outcome: { status: "completed", result: null } }
								: {
										status: "waiting",
										checkpoint: task.state.checkpoint,
										condition: { kind: "time", until: 100_000 },
									},
						ctx,
					);
				},
			},
			abort: async (task) => {
				if (task.input === "complete") throw new Error("Completed work must not be aborted");
				throw failures[task.input];
			},
		});
		const { harness } = await openTasks(new MemoryStorage(), [task], { now: () => 1_000 });
		harnesses.add(harness);
		const root = await harness.root(context);
		const other = await harness.createConversation({ ownership: { kind: "ownerless" } }, context);
		const backgroundId = await root.commit(
			(tx) => tx.createTask(task, "background", { ownership: { kind: "conversation" }, background: true }),
			context,
		);
		const foreignId = await other.commit(
			(tx) => tx.createTask(task, "foreign", { ownership: { kind: "conversation" } }),
			context,
		);
		await harness.runPass(context);
		await harness.abortTask(backgroundId, context);
		await harness.abortTask(foreignId, context);
		await harness.runPass(context);
		await expect(root.waitForIdle(context)).resolves.toBeUndefined();
		await expect(other.waitForIdle(context)).rejects.toBe(failures.foreign);
		await expect(harness.waitForIdle(context)).rejects.toBe(failures.foreign);
		await expect(harness.waitForTask(backgroundId, context)).rejects.toBe(failures.background);
		const incident = await cleanupIncident(harness, foreignId);
		await harness.abortTask(foreignId, context);
		await harness.runPass(context);
		expect(await cleanupIncident(harness, foreignId)).toBe(incident);
		await expect(other.waitForIdle(context)).rejects.toBe(failures.foreign);
		const completedId = await root.commit(
			(tx) => tx.createTask(task, "complete", { ownership: { kind: "conversation" } }),
			context,
		);
		await harness.runPass(context);
		expect((await harness.waitForTask(completedId, context)).state.outcome.status).toBe("completed");
		const entries = await other.entries({}, 100, undefined, context);
		expect(entries.items).toHaveLength(1);
		expect(entries.items[0]).toMatchObject({
			id: incident,
			kind: "pi.failure",
			data: { taskId: foreignId, error: { message: failures.foreign.message } },
		});
		expect(entries.items[0]!.model).toBeUndefined();
	});

	it("retains a rejected deferred cancellation across replacement and settles only after explicit repair", async () => {
		const databasePath = await path();
		const models = createModels();
		const setup = { ...chatSetup({ deferred: { pollAfterMs: 60_000 } }), models };
		setup.settings.stream = { deferred: true };
		setup.faux.setResponses([fauxAssistantMessage("unused")]);
		const original = new Error("provider refused cancellation");
		let repaired = false;
		const cancelled: unknown[] = [];
		models.setProvider({
			...setup.faux.provider,
			cancelDeferred: async (model, handle, options) => {
				cancelled.push(structuredClone({ model, handle }));
				if (!repaired) throw original;
				await setup.faux.provider.cancelDeferred!(model, handle, options);
			},
		});
		let opened = await open(databasePath, setup);
		const submission = await opened.root.submit({ type: "input", content: "go" }, context);
		await opened.harness.runPass(context);
		const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
		const before = (await opened.harness.getTask(taskId, context))!;
		if (before.state.status !== "waiting") throw new Error("Expected deferred polling wait");
		const checkpoint = before.state.checkpoint;
		await expect(opened.root.abort(context)).rejects.toBe(original);
		await opened.harness.runPass(context);
		const incident = await cleanupIncident(opened.harness, taskId);
		expect((await opened.harness.getTask(taskId, context))!.state).toMatchObject({ checkpoint });
		expect(await submission.status(context)).toMatchObject({ status: "placed" });
		expect(cancelled).toHaveLength(1);
		expect(setup.reports).toContain(original);
		await expect(opened.harness.waitForTask(taskId, context)).rejects.toBe(original);
		await opened.harness.close(context);

		opened = await open(databasePath, setup);
		await opened.harness.runPass(context);
		expect(await cleanupIncident(opened.harness, taskId)).toBe(incident);
		expect(cancelled).toHaveLength(1);
		await expect(opened.root.waitForIdle(context)).rejects.toThrow(original.message);
		repaired = true;
		expect(await opened.harness.retryTask(taskId, incident, context)).toBe("queued");
		await opened.harness.runPass(context);
		expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
			status: "unanswered",
			reason: "aborted",
		});
		expect(cancelled).toHaveLength(2);
		expect(cancelled[1]).toEqual(cancelled[0]);
		expect(await opened.harness.retryTask(taskId, incident, context)).toBe("terminal");
	});

	it("does not declare an absent provider cancelled or lose the deferred handle", async () => {
		const databasePath = await path();
		const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
		setup.settings.stream = { deferred: true };
		setup.faux.setResponses([fauxAssistantMessage("unused")]);
		let opened = await open(databasePath, setup);
		const submission = await opened.root.submit({ type: "input", content: "go" }, context);
		await opened.harness.runPass(context);
		const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
		const before = await opened.harness.getTask(taskId, context);
		if (before?.state.status !== "waiting") throw new Error("Expected deferred polling wait");
		const checkpoint = before.state.checkpoint;
		await opened.harness.close(context);
		opened = await open(databasePath, { ...setup, models: createModels() });
		await expect(opened.root.abort(context)).rejects.toThrow("faux");
		await opened.harness.runPass(context);
		const incident = await cleanupIncident(opened.harness, taskId);
		expect((await opened.harness.getTask(taskId, context))!.state).toMatchObject({
			checkpoint,
		});
		await opened.harness.close(context);
		opened = await open(databasePath, setup);
		expect(await opened.harness.retryTask(taskId, incident, context)).toBe("queued");
		await opened.harness.runPass(context);
		expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
			status: "unanswered",
			reason: "aborted",
		});
		expect(setup.faux.state.cancelledDeferred).toHaveLength(1);
	});

	it("retains connection-close failures after cancellation and rejects duplicate or stale repair", async () => {
		const databasePath = await path();
		const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
		setup.settings.stream = { deferred: true };
		setup.faux.setResponses([fauxAssistantMessage("unused")]);
		let repaired = false;
		let closes = 0;
		const original = new Error("transport close not confirmed");
		const opened = await open(databasePath, setup, async (request) => ({
			status: "ready",
			options: {},
			close: async () => {
				if (request.operation === "cancelDeferred") {
					closes++;
					if (!repaired) throw original;
				}
			},
		}));
		await opened.root.submit({ type: "input", content: "go" }, context);
		await opened.harness.runPass(context);
		const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
		await expect(opened.root.abort(context)).rejects.toBe(original);
		const first = await cleanupIncident(opened.harness, taskId);
		const retried = await Promise.all([
			opened.harness.retryTask(taskId, first, context),
			opened.harness.retryTask(taskId, first, context),
		]);
		expect(retried.sort()).toEqual(["queued", "stale"]);
		await opened.harness.runPass(context);
		const second = await cleanupIncident(opened.harness, taskId);
		expect(second).not.toBe(first);
		expect(closes).toBe(2);
		expect(await opened.harness.retryTask(taskId, first, context)).toBe("stale");
		await opened.harness.runPass(context);
		expect(closes).toBe(2);
		repaired = true;
		expect(await opened.harness.retryTask(taskId, second, context)).toBe("queued");
		await opened.harness.runPass(context);
		expect((await opened.harness.waitForTask(taskId, context)).state.outcome.status).toBe("aborted");
		expect(closes).toBe(3);
	});

	it("keeps the parent unfinished and propagates a child's cleanup failure without re-running it", async () => {
		const original = new Error("child process still owns its resource");
		let repaired = false;
		let parentAborts = 0;
		let childId!: TaskId;
		const child = defineTask<Record<string, never>, { phase: "work" }, null>({
			name: "test.cleanup-child",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async (task, runtime, ctx) => {
					await runtime.commit(
						() => ({
							status: "waiting",
							checkpoint: task.state.checkpoint,
							condition: { kind: "time", until: 100_000 },
						}),
						ctx,
					);
				},
			},
			abort: async (_task, runtime, ctx) => {
				if (!repaired) throw original;
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		const parent = defineTask<Record<string, never>, { phase: "work" }, null>({
			name: "test.cleanup-parent",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async (task, runtime, ctx) => {
					await runtime.commit(async (tx) => {
						childId = await tx.createTask(child, {}, { ownership: { kind: "task", taskId: runtime.taskId } });
						return {
							status: "waiting",
							checkpoint: task.state.checkpoint,
							condition: { kind: "tasks", on: [childId], policy: "allSettled" },
						};
					}, ctx);
				},
			},
			abort: async (_task, runtime, ctx) => {
				parentAborts++;
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		const { harness } = await openTasks(new MemoryStorage(), [child, parent], { now: () => 1_000 });
		harnesses.add(harness);
		const root = await harness.root(context);
		const parentId = await root.commit(
			(tx) => tx.createTask(parent, {}, { ownership: { kind: "conversation" } }),
			context,
		);
		await harness.runPass(context);
		await expect(root.abort(context)).rejects.toBe(original);
		await harness.runPass(context);
		const incident = await cleanupIncident(harness, childId);
		expect(parentAborts).toBe(0);
		expect((await harness.getTask(parentId, context))!.state.status).not.toBe("terminal");
		await expect(harness.waitForTask(parentId, context)).rejects.toBe(original);
		repaired = true;
		await harness.retryTask(childId, incident, context);
		await harness.runPass(context);
		expect((await harness.waitForTask(parentId, context)).state.outcome.status).toBe("aborted");
		expect(parentAborts).toBe(1);
	});

	it("preserves an unavailable cancellation definition until compatible publication", async () => {
		const task = defineTask<Record<string, never>, { phase: "work" }, null>({
			name: "test.unavailable-cleanup",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: {
				work: async (task, runtime, ctx) => {
					await runtime.commit(
						() => ({
							status: "waiting",
							checkpoint: task.state.checkpoint,
							condition: { kind: "time", until: 100_000 },
						}),
						ctx,
					);
				},
			},
			abort: async (_task, runtime, ctx) => {
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		const { harness, registry } = await openTasks(new MemoryStorage(), [task], { now: () => 1_000 });
		harnesses.add(harness);
		const root = await harness.root(context);
		const taskId = await root.commit(
			(tx) => tx.createTask(task, {}, { ownership: { kind: "conversation" } }),
			context,
		);
		await harness.runPass(context);
		registry.uninstall(registry.snapshot().extension("tasks")!);
		await harness.abortTask(taskId, context);
		await harness.runPass(context);
		expect(await harness.getTask(taskId, context)).toMatchObject({
			abortRequested: true,
			state: { status: "waiting", mode: "abort", condition: { kind: "registry", reason: { code: "missing_task" } } },
		});
		registry.install(defineExtension({ name: "tasks", tasks: [task] }));
		await harness.runPass(context);
		expect((await harness.waitForTask(taskId, context)).state.outcome.status).toBe("aborted");
	});
});
