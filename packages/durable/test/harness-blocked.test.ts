import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createModels } from "@earendil-works/pi-ai";
import {
	createRegistry,
	defineExtension,
	defineTask,
	Harness,
	type Registry,
	type ReportInfo,
	type Storage,
	type TaskId,
	type WakeSchedule,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { context, flush } from "./session-support.ts";
import { completed, deferred, eventually, openTasks } from "./task-support.ts";

const directories = new Set<string>();

/** Storages of killed Harnesses, never closed by them: closed after each test. */
const killed = new Set<Storage>();

afterEach(async () => {
	for (const storage of killed) await storage.close(context).catch(() => {});
	killed.clear();
	for (const directory of directories) await rm(directory, { recursive: true, force: true });
	directories.clear();
});

async function sqlitePath(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-durable-blocked-"));
	directories.add(directory);
	return join(directory, "session.sqlite");
}

/** A task that hangs in its phase while `crashing`, as one killed every time would, and completes otherwise. */
function crashLoop(state: { crashing: boolean; started: number }) {
	return defineTask<null, { phase: "run" }, null>({
		name: "test.crash-loop",
		version: 1,
		initial: () => ({ phase: "run" }),
		phases: {
			run: async (_task, runtime, ctx) => {
				state.started++;
				if (state.crashing)
					await new Promise<void>((resolve) => runtime.signal.addEventListener("abort", () => resolve()));
				else await runtime.commit(() => completed(null), ctx);
			},
		},
		abort: async () => {},
	});
}

/**
 * Open a Harness over `path`. `kill()` simulates a crash: every later call of its storage hangs, so it writes nothing
 * more, not even the `pending` records a close writes for the tasks it stops.
 */
async function openWith(path: string, registry: Registry) {
	const wakes: WakeSchedule[] = [];
	const inner = await openNodeSqliteStorage(path);
	let dead = false;
	const storage = new Proxy(inner, {
		get(target, property) {
			const value = Reflect.get(target, property, target);
			if (typeof value !== "function") return value;
			return (...args: unknown[]) => (dead ? new Promise(() => {}) : value.apply(target, args));
		},
	});
	const infos: ReportInfo[] = [];
	const harness = await Harness.open(
		storage,
		{
			models: createModels(),
			registry,
			publishWake: async (wake) => {
				wakes.push(wake);
			},
			onReport: (_error, info) => {
				infos.push(info);
			},
		},
		context,
	);
	const kill = (): void => {
		dead = true;
		killed.add(inner);
	};
	return { harness, wakes, infos, kill };
}

describe("recovery at open", () => {
	it("propagates a failed recovery commit and retains work for a later open", async () => {
		const path = await sqlitePath();
		const Held = defineTask<null, { phase: "run" }, null>({
			name: "test.held",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: { run: async () => {} },
			abort: async () => {},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "held", tasks: [Held] }));
		const first = await openWith(path, registry);
		const id = await (await first.harness.root(context)).commit(
			(tx) => tx.createTask(Held, null, { ownership: { kind: "conversation" } }),
			context,
		);
		await first.harness.close(context);
		// A crash left the task holding its outcome: the next open's recovery pass makes it terminal.
		const raw = await openNodeSqliteStorage(path);
		const record = (await raw.task(id, context))!;
		const { memos: _memos, ...rest } = record;
		const completing = {
			...rest,
			state: { status: "completing", outcome: { status: "completed", result: null } },
		} as const;
		await raw.commit([{ type: "task", value: completing }], () => {}, context);
		await raw.close(context);

		// That pass's commit fails, as on a platform reset.
		const inner = await openNodeSqliteStorage(path);
		let failNext = true;
		const failing = new Proxy(inner, {
			get(target, property) {
				if (property === "commit") {
					return (...args: Parameters<typeof inner.commit>) => {
						if (!failNext) return target.commit(...args);
						failNext = false;
						return Promise.reject(new Error("reset"));
					};
				}
				const value = Reflect.get(target, property, target);
				return typeof value === "function" ? value.bind(target) : value;
			},
		});
		await expect(openTasks(failing, [], { registry })).rejects.toThrow("reset");
		// The failed attempt did not write a terminal outcome. A later healthy open owns recovery.
		const reopened = await openWith(path, registry);
		await eventually(async () => (await reopened.harness.getTask(id, context))?.state.status === "terminal");
		expect((await reopened.harness.getTask(id, context))?.state).toMatchObject({
			status: "terminal",
			outcome: { status: "completed" },
		});
		await reopened.harness.close(context);
	});
});

describe("tasks that cannot run", () => {
	it("recovers repeatedly interrupted work from its checkpoint without manufacturing a fault", async () => {
		const path = await sqlitePath();
		const state = { crashing: true, started: 0 };
		const registry = createRegistry();
		const Loop = crashLoop(state);
		registry.install(defineExtension({ name: "loop", tasks: [Loop] }));
		let opened = await openWith(path, registry);
		const root = await opened.harness.root(context);
		const id = await root.commit((tx) => tx.createTask(Loop, null, { ownership: { kind: "conversation" } }), context);
		for (let run = 1; run <= 3; run++) {
			opened.harness.resume();
			await eventually(() => state.started === run);
			opened.kill();
			opened = await openWith(path, registry);
			expect(await opened.harness.getTask(id, context)).toMatchObject({
				state: { status: "pending", checkpoint: { phase: "run" } },
			});
		}
		expect(opened.infos).toEqual([]);
		state.crashing = false;
		expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		expect(state.started).toBe(4);
		await opened.harness.close(context);
	});

	it("recovers a task stopped by close from its checkpoint", async () => {
		const path = await sqlitePath();
		const state = { crashing: true, started: 0 };
		const registry = createRegistry();
		const Loop = crashLoop(state);
		registry.install(defineExtension({ name: "loop", tasks: [Loop] }));
		let opened = await openWith(path, registry);
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Loop, null, { ownership: { kind: "conversation" } }),
			context,
		);
		opened.harness.resume();
		await eventually(() => state.started === 1);
		await opened.harness.close(context);
		opened = await openWith(path, registry);
		expect(await opened.harness.getTask(id, context)).toMatchObject({ state: { status: "pending" } });
		state.crashing = false;
		expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		await opened.harness.close(context);
	});

	it("keeps an abort mark admitted before close through recovery", async () => {
		const path = await sqlitePath();
		const state = { crashing: true, started: 0 };
		const registry = createRegistry();
		const Loop = crashLoop(state);
		registry.install(defineExtension({ name: "loop", tasks: [Loop] }));
		const { harness } = await openWith(path, registry);
		const root = await harness.root(context);
		const id = await root.commit((tx) => tx.createTask(Loop, null, { ownership: { kind: "conversation" } }), context);
		harness.resume();
		await eventually(() => state.started === 1);
		// Hold the line; the abort mark is admitted behind it, then close begins.
		const gate = deferred();
		const holding = root.commit(() => gate.promise, context);
		const aborting = harness.abortTask(id, context);
		const closing = harness.close(context);
		// The admitted abort mark settles even though close has sealed further execution.
		await flush();
		gate.resolve();
		await holding;
		expect(await aborting).toBe("marked");
		await closing;
		const raw = await openNodeSqliteStorage(path);
		expect(await raw.task(id, context)).toMatchObject({ abortRequested: true, state: { status: "running" } });
		await raw.close(context);
		const reopened = await openWith(path, registry);
		expect(await reopened.harness.getTask(id, context)).toMatchObject({
			abortRequested: true,
			state: { status: "pending" },
		});
		await reopened.harness.close(context);
	});

	it("recovers interrupted abort cleanup without manufacturing a fault", async () => {
		const path = await sqlitePath();
		let aborts = 0;
		let recover = false;
		const Stuck = defineTask<null, { phase: "run" }, null>({
			name: "test.stuck-abort",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: {
				run: (_task, runtime) =>
					new Promise((resolve) => runtime.signal.addEventListener("abort", () => resolve())),
			},
			abort: async (_task, runtime, ctx) => {
				aborts++;
				if (recover) await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
				else await new Promise<void>((resolve) => runtime.signal.addEventListener("abort", () => resolve()));
			},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "stuck", tasks: [Stuck] }));
		let opened = await openWith(path, registry);
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Stuck, null, { ownership: { kind: "conversation" } }),
			context,
		);
		opened.harness.resume();
		void opened.harness.abortTask(id, context);
		await eventually(() => aborts === 1);
		opened.kill();
		opened = await openWith(path, registry);
		expect(await opened.harness.getTask(id, context)).toMatchObject({
			abortRequested: true,
			state: { status: "pending" },
		});
		recover = true;
		expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("aborted");
		expect(aborts).toBe(2);
		expect(opened.infos).toEqual([]);
		await opened.harness.close(context);
	});

	it("does not duplicate an active invocation on a registry publication", async () => {
		const path = await sqlitePath();
		const state = { crashing: true, started: 0 };
		const registry = createRegistry();
		const Loop = crashLoop(state);
		registry.install(defineExtension({ name: "loop", tasks: [Loop] }));
		let opened = await openWith(path, registry);
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Loop, null, { ownership: { kind: "conversation" } }),
			context,
		);
		opened.harness.resume();
		await eventually(() => state.started === 1);
		registry.install(defineExtension({ name: "unrelated" }));
		await flush();
		expect(state.started).toBe(1);
		opened.kill();
		state.crashing = false;
		opened = await openWith(path, registry);
		expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		expect(state.started).toBe(2);
		await opened.harness.close(context);
	});

	it("runs the abort handler of a recovered task", async () => {
		const path = await sqlitePath();
		const state = { crashing: true, started: 0 };
		const aborted: TaskId[] = [];
		const registry = createRegistry();
		const Loop = defineTask<null, { phase: "run" }, null>({
			...crashLoop(state).definition,
			abort: async (task, runtime, ctx) => {
				aborted.push(task.id);
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		registry.install(defineExtension({ name: "loop", tasks: [Loop] }));
		let opened = await openWith(path, registry);
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Loop, null, { ownership: { kind: "conversation" } }),
			context,
		);
		opened.harness.resume();
		await eventually(() => state.started === 1);
		opened.kill(); // a crash: a close would not count
		opened = await openWith(path, registry);
		await opened.harness.abortTask(id, context);
		expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("aborted");
		expect(aborted).toEqual([id]);
		expect(state.started).toBe(1);
		await opened.harness.close(context);
	});

	it("recovers the same checkpoint after repeated unchanged commits", async () => {
		const path = await sqlitePath();
		let started = 0;
		const Intent = defineTask<null, { phase: "run" }, null>({
			name: "test.intent",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: {
				run: async (task, runtime, ctx) => {
					started++;
					// Intent before an effect, then the effect never returns: the checkpoint is unchanged.
					await runtime.commit(() => ({ status: "running", checkpoint: task.state.checkpoint }), ctx);
					await new Promise<void>((resolve) => runtime.signal.addEventListener("abort", () => resolve()));
				},
			},
			abort: async () => {},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "intent", tasks: [Intent] }));
		let opened = await openWith(path, registry);
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Intent, null, { ownership: { kind: "conversation" } }),
			context,
		);
		for (let run = 1; run <= 2; run++) {
			opened.harness.resume();
			await eventually(() => started === run);
			await eventually(async () => (await opened.harness.getTask(id, context))?.state.status === "running");
			opened.kill(); // a crash: a close would not count
			opened = await openWith(path, registry);
		}
		expect((await opened.harness.getTask(id, context))?.state).toMatchObject({
			status: "pending",
			checkpoint: { phase: "run" },
		});
		expect((await opened.harness.inspect(context)).tasks[0]?.state.kind).not.toBe("blocked");
		await opened.harness.close(context);
	});

	it("recovers the latest committed checkpoint after progress", async () => {
		const path = await sqlitePath();
		let crashing = true;
		let started = 0;
		const Steps = defineTask<null, { phase: "one" } | { phase: "two" }, null>({
			name: "test.steps",
			version: 1,
			initial: () => ({ phase: "one" }),
			phases: {
				one: async (_task, runtime, ctx) => {
					started++;
					if (crashing)
						await new Promise<void>((resolve) => runtime.signal.addEventListener("abort", () => resolve()));
					else await runtime.commit(() => ({ status: "running", checkpoint: { phase: "two" } }), ctx);
				},
				two: async (_task, runtime) => {
					started++;
					await new Promise<void>((resolve) => runtime.signal.addEventListener("abort", () => resolve()));
				},
			},
			abort: async () => {},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "steps", tasks: [Steps] }));
		let opened = await openWith(path, registry);
		const id: TaskId = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Steps, null, { ownership: { kind: "conversation" } }),
			context,
		);
		opened.harness.resume();
		await eventually(() => started === 1);
		opened.kill(); // a crash: a close would not count
		opened = await openWith(path, registry);
		expect((await opened.harness.getTask(id, context))?.state).toMatchObject({
			status: "pending",
			checkpoint: { phase: "one" },
		});
		// Progress commits phase two; a subsequent crash must recover that checkpoint.
		crashing = false;
		opened.harness.resume();
		await eventually(() => started === 3);
		opened.kill();
		opened = await openWith(path, registry);
		expect((await opened.harness.getTask(id, context))?.state).toMatchObject({
			status: "pending",
			checkpoint: { phase: "two" },
		});
		await opened.harness.close(context);
	});

	it("tells the graph and the host when a migration fails", async () => {
		const path = await sqlitePath();
		const run = { phases: { run: async () => {} }, abort: async () => {} } as const;
		const V1 = defineTask<null, { phase: "run" }, null>({
			name: "test.migrating",
			version: 1,
			initial: () => ({ phase: "run" }),
			...run,
		});
		const V2 = defineTask<null, { phase: "run" }, null>({
			name: "test.migrating",
			version: 2,
			initial: () => ({ phase: "run" }),
			...run,
			migrate: () => {
				throw new Error("cannot migrate");
			},
		});
		const first = createRegistry();
		first.install(defineExtension({ name: "v1", tasks: [V1] }));
		let opened = await openWith(path, first);
		// Created, never run: scheduling stays off.
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(V1, null, { ownership: { kind: "conversation" } }),
			context,
		);
		await opened.harness.close(context);
		const second = createRegistry();
		second.install(defineExtension({ name: "v2", tasks: [V2] }));
		opened = await openWith(path, second);
		const graph = await opened.harness.watchTaskGraph(context);
		// A started watch keeps its value current.
		graph.start(async () => {});
		opened.harness.resume();
		await eventually(() => graph.value.tasks[String(id)]?.blocked === "migration_failed");
		await eventually(() => opened.wakes.at(-1)?.wakeAt === null);
		expect((await opened.harness.flushWake(context)).wakeAt).toBeNull();
		await graph.stop();
		await opened.harness.close(context);
	});

	it("shows an abandoned task waiting for its definition as blocked", async () => {
		const path = await sqlitePath();
		const Abandoned = defineTask<null, { phase: "run" }, null>({
			name: "test.abandoned",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: { run: async () => {} },
			abort: async () => {},
		});
		const registry = createRegistry();
		registry.install(defineExtension({ name: "abandoned", tasks: [Abandoned] }));
		let opened = await openWith(path, registry);
		const id = await (await opened.harness.root(context)).commit(
			(tx) => tx.createTask(Abandoned, null, { ownership: { kind: "conversation" }, abandonOnRestart: true }),
			context,
		);
		await opened.harness.close(context);
		// The next open marks it `restart`; without its definition, it waits for one to clean up.
		opened = await openWith(path, createRegistry());
		const graph = await opened.harness.watchTaskGraph(context);
		// A started watch keeps its value current.
		graph.start(async () => {});
		opened.harness.resume();
		await eventually(async () => (await opened.harness.getTask(id, context))?.abortReason === "restart");
		await eventually(() => graph.value.tasks[String(id)]?.blocked === "missing_task");
		await graph.stop();
		await opened.harness.close(context);
	});

	it("shows a task without a definition as blocked, and wakes the host once a registry change unblocks it", async () => {
		const Missing = defineTask<null, { phase: "run" }, null>({
			name: "test.missing",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: { run: (_task, runtime, ctx) => runtime.commit(() => completed(null), ctx) },
			abort: async () => {},
		});
		const registry = createRegistry();
		const path = await sqlitePath();
		const { harness, wakes } = await openWith(path, registry);
		const graph = await harness.watchTaskGraph(context);
		// A started watch keeps its value current.
		graph.start(async () => {});
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Missing, null, { ownership: { kind: "conversation" } }),
			context,
		);
		harness.resume();
		await eventually(() => graph.value.tasks[String(id)]?.blocked === "missing_task");
		await eventually(() => wakes.at(-1)?.wakeAt === null);
		registry.install(defineExtension({ name: "late", tasks: [Missing] }));
		// The graph follows the registry; the host hears it can run work now.
		await eventually(() => graph.value.tasks[String(id)]?.blocked === undefined);
		await eventually(() => wakes.some((wake) => wake.wakeAt === 0));
		expect((await harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		await graph.stop();
		await harness.close(context);
	});
});
