import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createModels, fauxAssistantMessage, fauxProvider, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	acceptReceipt,
	bindReceipt,
	createRegistry,
	defineExtension,
	defineTask,
	defineTool,
	type EntryId,
	Harness,
	ReceiptDoc,
	UserEntry,
	WakeDoc,
	type WakeSchedule,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OutputBuffer } from "../src/harness/output.ts";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { waitFor } from "./chat-support.ts";
import { context } from "./session-support.ts";

const directories: string[] = [];
const handles: Harness[] = [];
afterEach(async () => {
	await Promise.all(handles.splice(0).map((harness) => harness.close(context)));
	await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});
async function database() {
	const directory = await mkdtemp(join(tmpdir(), "pi-dormant-"));
	directories.push(directory);
	return join(directory, "state.sqlite");
}
const done = (result: string) => ({ status: "terminal", outcome: { status: "completed", result } }) as const;
const abort = async (
	_task: unknown,
	runtime: { commit: (change: () => ReturnType<typeof done>, ctx: typeof context) => Promise<void> },
) => runtime.commit(() => done("cancelled"), context);
const Timed = defineTask<{ until: number }, { phase: "wait" | "finish" }, string>({
	name: "test.dormant-timer",
	version: 1,
	initial: () => ({ phase: "wait" }),
	abort,
	phases: {
		wait: async (task, runtime, ctx) =>
			runtime.commit(
				() => ({
					status: "waiting",
					checkpoint: { phase: "finish" },
					condition: { kind: "time", until: task.input.until },
				}),
				ctx,
			),
		finish: async (_task, runtime, ctx) => runtime.commit(() => done("timed"), ctx),
	},
});
const External = defineTask<{ key: string; binding: string }, { phase: "wait" | "finish" | "cleanup" }, string>({
	name: "test.dormant-receipt",
	version: 1,
	initial: () => ({ phase: "wait" }),
	phases: {
		wait: async (task, runtime, ctx) =>
			runtime.commit(async (tx) => {
				await bindReceipt(tx, task.input.key, task.input.binding);
				return {
					status: "waiting",
					checkpoint: { phase: "finish" },
					condition: { kind: "receipt", ...task.input },
				};
			}, ctx),
		finish: async (task, runtime, ctx) => {
			const receipt = await runtime.snapshot(ReceiptDoc, task.input.key, ctx);
			await runtime.commit(() => done(JSON.stringify(receipt?.result)), ctx);
		},
		cleanup: async () => {
			throw new Error("Cleanup resumes the abort handler, not a run phase");
		},
	},
	abort: async (task, runtime, ctx) => {
		if (task.state.checkpoint.phase !== "cleanup") {
			await runtime.commit(async (tx) => {
				await bindReceipt(tx, "cleanup", "owner-cancel");
				return {
					status: "waiting",
					checkpoint: { phase: "cleanup" },
					condition: { kind: "receipt", key: "cleanup", binding: "owner-cancel" },
				};
			}, ctx);
			return;
		}
		await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
	},
});
async function open(
	path: string,
	options: { now?: () => number; publishWake?: (schedule: WakeSchedule) => Promise<void> } = {},
) {
	const registry = createRegistry();
	registry.install(defineExtension({ name: "test", tasks: [Timed, External] }));
	const harness = await Harness.open(
		await openNodeSqliteStorage(path),
		{ models: createModels(), registry, publishWake: async () => {}, ...options },
		context,
	);
	handles.push(harness);
	return harness;
}

describe("durable dormant scheduling", () => {
	it.each([1000, 2_147_484_747])("keeps an early resident timer armed until its deadline %i", async (until) => {
		const timers: { fire: () => void; handle: ReturnType<typeof setTimeout> }[] = [];
		const expectedDelay = Math.min(2_147_483_647, until - 100);
		const nativeSetTimeout = globalThis.setTimeout;
		const timerSpy = vi.spyOn(globalThis, "setTimeout").mockImplementation((callback, delay, ...args) => {
			const handle = nativeSetTimeout(callback, delay, ...args);
			if (delay === expectedDelay) timers.push({ fire: () => callback(...args), handle });
			return handle;
		});
		let now = 100;
		let harness: Harness | undefined;
		try {
			harness = await open(await database(), { now: () => now, publishWake: undefined });
			const root = await harness.root(context);
			const id = await root.commit(
				(tx) => tx.createTask(Timed, { until }, { ownership: { kind: "conversation" } }),
				context,
			);
			await harness.runPass(context);
			const early = timers.at(-1)!;
			const before = timers.length;
			clearTimeout(early.handle);
			early.fire();
			await waitFor(() => timers.length > before);
			expect((await harness.getTask(id, context))?.state.status).toBe("waiting");
			now = until;
			const due = timers.at(-1)!;
			clearTimeout(due.handle);
			due.fire();
			expect((await harness.waitForTask(id, context)).state).toEqual(done("timed"));
		} finally {
			try {
				await harness?.close(context);
			} finally {
				timerSpy.mockRestore();
				for (const timer of timers) clearTimeout(timer.handle);
			}
		}
	});

	it("bounds a pass and gives later independent work its turn without duplicate task execution", async () => {
		const path = await database();
		const executed = new Set<number>();
		const Short = defineTask<number, { phase: "run" }, string>({
			name: "test.finite-pass",
			version: 1,
			initial: () => ({ phase: "run" }),
			abort,
			phases: {
				run: async (task, runtime, ctx) => {
					expect(executed.has(task.input)).toBe(false);
					executed.add(task.input);
					await runtime.commit(() => done(String(task.input)), ctx);
				},
			},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "finite", tasks: [Short] }));
		const harness = await Harness.open(
			await openNodeSqliteStorage(path),
			{ models: createModels(), registry, publishWake: async () => {} },
			context,
		);
		handles.push(harness);
		const root = await harness.root(context);
		await root.commit(async (tx) => {
			for (let i = 0; i < 300; i++) await tx.createTask(Short, i, { ownership: { kind: "conversation" } });
		}, context);
		const independent = await harness.createConversation({ ownership: { kind: "ownerless" } }, context);
		const last = await independent.commit(
			(tx) => tx.createTask(Short, 300, { ownership: { kind: "conversation" } }),
			context,
		);
		expect((await harness.runPass(context)).wakeAt).toBe(0);
		expect(executed.size).toBe(256);
		expect((await harness.getTask(last, context))?.state.status).toBe("pending");
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		expect(executed.size).toBe(301);
		expect((await harness.getTask(last, context))?.state).toEqual(done("300"));
	});
	it("releases a timed invocation, reopens before its deadline and completes on a quiet host wake", async () => {
		const path = await database();
		let now = 100;
		let harness = await open(path, { now: () => now });
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Timed, { until: 1000 }, { ownership: { kind: "conversation" } }),
			context,
		);
		expect((await harness.runPass(context)).wakeAt).toBe(1000);
		expect((await harness.inspect(context)).tasks.find((task) => task.record.id === id)?.state.kind).toBe("waiting");
		await harness.close(context);
		harness = await open(path, { now: () => now });
		expect((await harness.runPass(context)).wakeAt).toBe(1000);
		expect((await harness.getTask(id, context))?.state.status).toBe("waiting");
		now = 1000;
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		expect((await harness.getTask(id, context))?.state).toEqual(done("timed"));
	});

	it("binds early receipts, rejects unknown identities and conflicts, accepts duplicate JSON regardless of key order", async () => {
		const harness = await open(await database());
		const root = await harness.root(context);
		await expect(harness.commit((tx) => acceptReceipt(tx, "unknown", "wrong", null), context)).rejects.toThrow(
			/admitted/,
		);
		await harness.commit(async (tx) => {
			await bindReceipt(tx, "op", "owner");
			await acceptReceipt(tx, "op", "owner", { b: 2, a: 1 });
		}, context);
		const id = await root.commit(
			(tx) => tx.createTask(External, { key: "op", binding: "owner" }, { ownership: { kind: "conversation" } }),
			context,
		);
		await harness.runPass(context);
		expect((await harness.getTask(id, context))?.state.status).toBe("terminal");
		await harness.commit((tx) => acceptReceipt(tx, "op", "owner", { a: 1, b: 2 }), context);
		await expect(harness.commit((tx) => acceptReceipt(tx, "op", "owner", { a: 3 }), context)).rejects.toThrow(
			/conflicts/,
		);
		await expect(harness.commit((tx) => acceptReceipt(tx, "op", "other", null), context)).rejects.toThrow(/admitted/);
	});

	it("parks abort cleanup across replacement, without admitting a run handler or returning an outcome early", async () => {
		const path = await database();
		let harness = await open(path);
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(External, { key: "op", binding: "owner" }, { ownership: { kind: "conversation" } }),
			context,
		);
		await harness.runPass(context);
		await harness.abortTask(id, context);
		await harness.runPass(context);
		expect((await harness.getTask(id, context))?.state).toMatchObject({
			status: "waiting",
			mode: "abort",
			checkpoint: { phase: "cleanup" },
		});
		await harness.close(context);
		harness = await open(path);
		await harness.runPass(context);
		expect((await harness.getTask(id, context))?.state.status).toBe("waiting");
		await harness.commit((tx) => acceptReceipt(tx, "cleanup", "owner-cancel", { cancelled: true }), context);
		await harness.runPass(context);
		expect((await harness.getTask(id, context))?.state).toEqual({
			status: "terminal",
			outcome: { status: "aborted" },
		});
	});

	it("retains an unacknowledged schedule through publication loss and acknowledges only the revision actually installed", async () => {
		const path = await database();
		let lose = true;
		const schedules: WakeSchedule[] = [];
		let harness = await open(path, {
			publishWake: async (schedule) => {
				if (lose) throw new Error("lost transport");
				schedules.push(schedule);
			},
		});
		const root = await harness.root(context);
		await root.commit(
			(tx) => tx.createTask(Timed, { until: Date.now() + 60_000 }, { ownership: { kind: "conversation" } }),
			context,
		);
		await expect(harness.runPass(context)).rejects.toThrow("lost transport");
		const before = (await harness.snapshot(WakeDoc, context))!;
		expect(before.publishedRevision).toBeLessThan(before.revision);
		await harness.close(context);
		lose = false;
		harness = await open(path, {
			publishWake: async (schedule) => {
				schedules.push(schedule);
			},
		});
		const after = await harness.runPass(context);
		expect(after).toMatchObject({ revision: before.revision, wakeAt: before.wakeAt });
		expect((await harness.snapshot(WakeDoc, context))!.publishedRevision).toBe(after.revision);
		expect(schedules.at(-1)).toEqual(after);
	});

	it("derives input readiness from staged eligible entries in the same atomic commit", async () => {
		const path = await database();
		const Input = defineTask<{ conversation: number }, { phase: "wait" | "finish" }, string>({
			name: "test.input-wait",
			version: 1,
			initial: () => ({ phase: "wait" }),
			abort,
			phases: {
				wait: async (_task, runtime, ctx) =>
					runtime.commit(
						() => ({
							status: "waiting",
							checkpoint: { phase: "finish" },
							condition: {
								kind: "input",
								conversationId: runtime.conversationId,
								after: 0 as EntryId,
								kinds: [UserEntry.kind],
							},
						}),
						ctx,
					),
				finish: async (_task, runtime, ctx) => runtime.commit(() => done("input"), ctx),
			},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "input", tasks: [Input] }));
		const harness = await Harness.open(
			await openNodeSqliteStorage(path),
			{ models: createModels(), registry, publishWake: async () => {} },
			context,
		);
		handles.push(harness);
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Input, { conversation: root.id }, { ownership: { kind: "conversation" } }),
			context,
		);
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		await root.commit((tx) => tx.appendEntry(root.id, { kind: "ineligible" }), context);
		expect((await harness.flushWake(context)).wakeAt).toBeNull();
		await root.commit((tx) => tx.appendEntry(root.id, { kind: UserEntry.kind }), context);
		expect((await harness.snapshot(WakeDoc, context))!.wakeAt).toBe(0);
		await harness.runPass(context);
		expect((await harness.getTask(id, context))?.state).toEqual(done("input"));
	});

	it("releases a scripted model tool invocation and recovers its exact continuation and output after reopen", async () => {
		const path = await database();
		const faux = fauxProvider();
		const models = createModels();
		models.setProvider(faux.provider);
		let admissions = 0;
		let attaches = 0;
		const tool = defineTool({
			name: "external",
			description: "durable operation",
			parameters: Type.Object({}),
			replay: "unsafe",
			execute: async (_args, api, ctx) => {
				const key = `eval:${api.taskId}`;
				if (api.continuation === undefined) {
					await api.commit((tx) => bindReceipt(tx, key, "eval-owner"), ctx);
					admissions++;
					api.output("retained output\n");
					await api.details({ identity: key }, ctx);
					return { wait: { kind: "receipt" as const, key, binding: "eval-owner" }, continuation: key };
				}
				attaches++;
				const receipt = await api.snapshot(ReceiptDoc, key, ctx);
				expect(api.continuation).toBe(key);
				return { details: receipt?.result };
			},
			cancel: async () => ({ content: [] }),
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "external", tools: [tool] }));
		const openHarness = async () => {
			const harness = await Harness.open(
				await openNodeSqliteStorage(path),
				{ models, registry, publishWake: async () => {} },
				context,
			);
			handles.push(harness);
			return harness;
		};
		faux.setResponses([
			fauxAssistantMessage([fauxToolCall("external", {}, { id: "call" })], { stopReason: "toolUse" }),
		]);
		let harness = await openHarness();
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		const submission = await root.submit({ type: "input", content: "run durable eval" }, context);
		await harness.runPass(context);
		const external = (await harness.inspect(context)).tasks.find((task) => task.record.kind === "pi.tool")!;
		expect(external.record.state.status).toBe("waiting");
		expect(admissions).toBe(1);
		await harness.close(context);
		harness = await openHarness();
		const key = `eval:${external.record.id}`;
		await harness.commit((tx) => acceptReceipt(tx, key, "eval-owner", { returnValue: 42 }), context);
		faux.setResponses([fauxAssistantMessage("result is 42")]);
		await harness.runPass(context);
		expect(admissions).toBe(1);
		expect(attaches).toBe(1);
		const resumed = (await harness.conversation(root.id, context))!;
		const entries = await resumed.entries({}, 100, undefined, context);
		const result = entries.items.find((entry) => entry.kind === "pi.tool-result");
		expect(result?.model?.[0]).toMatchObject({ content: [{ type: "text", text: "retained output\n" }] });
		expect((await (await harness.submission(submission.id, context))?.status(context))?.status).toBe("done");
	});

	it("restores bounded stream counters across a continuation without inflating retention", () => {
		for (const retain of ["head", "tail"] as const) {
			const limits = { maxBytes: 8, maxLines: 2, retain };
			const continuous = new OutputBuffer(limits);
			continuous.push("a\nb\nc\nd\n");
			const resumed = new OutputBuffer(limits, continuous.checkpoint());
			continuous.push("last\n");
			resumed.push("last\n");
			expect(resumed.snapshot()).toEqual(continuous.snapshot());
			expect(resumed.storedBytes).toBeLessThanOrEqual(8);
		}
	});
});
