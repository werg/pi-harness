import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	defineExtension,
	defineTask,
	MemoryStorage,
	type SetTimer,
	type Storage,
	type TaskId,
	WakeDoc,
	type WakeSchedule,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { context, flush } from "./session-support.ts";
import { aborted, abortedWith, completed, eventually, openTasks } from "./task-support.ts";

const directories = new Set<string>();

afterEach(async () => {
	for (const directory of directories) await rm(directory, { recursive: true, force: true });
	directories.clear();
});

async function sqlitePath(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-durable-sleep-"));
	directories.add(directory);
	return join(directory, "session.sqlite");
}

type NapState = { phase: "start" } | { phase: "after"; slept: number };

/** Sleeps `delay` ms on the Harness clock, then completes with the time it resumed at. */
function napTask(delay: number, runs: { after: number } = { after: 0 }) {
	return defineTask<null, NapState, number>({
		name: "test.nap",
		version: 1,
		initial: () => ({ phase: "start" }),
		phases: {
			start: async (_task, runtime, ctx) => {
				await runtime.sleepUntil(runtime.now() + delay, { phase: "after", slept: delay }, ctx);
			},
			after: async (_task, runtime, ctx) => {
				runs.after++;
				const resumed = runtime.now();
				await runtime.commit(() => completed(resumed), ctx);
			},
		},
		abort: async (_task, runtime, ctx) => {
			await runtime.commit(() => abortedWith("test"), ctx);
		},
	});
}

async function startNap(harness: Awaited<ReturnType<typeof openTasks>>["harness"], task: ReturnType<typeof napTask>) {
	const root = await harness.root(context);
	return root.commit((tx) => tx.createTask(task, null, { ownership: { kind: "conversation" } }), context);
}

async function stateOf(harness: Awaited<ReturnType<typeof openTasks>>["harness"], id: TaskId) {
	return (await harness.getTask(id, context))?.state;
}

type FakeTimer = { readonly delayMs: number; readonly callback: () => void; disarmed: boolean };

/** A host `setTimer` that records the timers that pace work and declines background ones. */
function fakeTimers() {
	const armed: FakeTimer[] = [];
	const setTimer: SetTimer = (callback, delayMs, options) => {
		if (options.background) return undefined;
		const timer: FakeTimer = { delayMs, callback, disarmed: false };
		armed.push(timer);
		return () => {
			timer.disarmed = true;
		};
	};
	return { armed, setTimer, live: () => armed.filter((timer) => !timer.disarmed) };
}

/** Let microtasks and a few timer turns run. */
async function settle(): Promise<void> {
	await flush();
	await new Promise((resolve) => setTimeout(resolve, 20));
	await flush();
}

describe("sleeping tasks", () => {
	it("sleeps without an invocation and resumes at its checkpoint once due", async () => {
		let clock = 1_000;
		const wakes: WakeSchedule[] = [];
		const runs = { after: 0 };
		const Nap = napTask(1_000, runs);
		const { harness } = await openTasks(new MemoryStorage(), [Nap], {
			now: () => clock,
			publishWake: async (wake) => {
				wakes.push(wake);
			},
		});
		const graph = await harness.watchTaskGraph(context);
		// A started watch keeps its value current.
		graph.start(async () => {});
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(async () => (await stateOf(harness, id))?.status === "waiting");
		expect(await stateOf(harness, id)).toEqual({
			status: "waiting",
			checkpoint: { phase: "after", slept: 1_000 },
			condition: { kind: "time", until: 2_000 },
			mode: "run",
		});
		const inspection = await harness.inspect(context);
		expect(inspection.tasks.find((task) => task.record.id === id)?.state).toEqual({
			kind: "waiting",
			on: [],
			condition: { kind: "time", until: 2_000 },
		});
		expect(graph.value.tasks[String(id)]?.state).toEqual({
			status: "waiting",
			phase: "after",
			condition: { kind: "time", until: 2_000 },
		});
		expect(await harness.runPass(context)).toMatchObject({ wakeAt: 2_000 });
		await eventually(() => wakes.at(-1)?.wakeAt === 2_000);
		expect(await harness.snapshot(WakeDoc, context)).toMatchObject({ wakeAt: 2_000 });

		// Early and repeated wakes run nothing.
		clock = 1_999;
		harness.resume();
		harness.resume();
		await settle();
		expect(runs.after).toBe(0);
		expect((await stateOf(harness, id))?.status).toBe("waiting");

		clock = 2_000;
		harness.resume();
		harness.resume();
		const receipt = await harness.waitForTask(id, context);
		expect(receipt.state.outcome).toEqual({ status: "completed", result: 2_000 });
		expect(runs.after).toBe(1);
		await settle();
		expect(await harness.runPass(context)).toMatchObject({ wakeAt: null });
		await eventually(() => wakes.at(-1)?.wakeAt === null);
		expect(wakes.some((wake) => wake.wakeAt === 2_000)).toBe(true);
		for (let index = 1; index < wakes.length; index++)
			expect(wakes[index].revision).toBeGreaterThanOrEqual(wakes[index - 1].revision);
		await graph.stop();
		await harness.close(context);
	});

	it("keeps a sleep across reopen", async () => {
		const path = await sqlitePath();
		let clock = 1_000;
		const Nap = napTask(60_000);
		const open = async (storage: Storage) =>
			openTasks(storage, [Nap], { now: () => clock, publishWake: async () => {} });
		let { harness } = await open(await openNodeSqliteStorage(path));
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(async () => (await stateOf(harness, id))?.status === "waiting");
		await harness.close(context);

		({ harness } = await open(await openNodeSqliteStorage(path)));
		// Known from storage before scheduling starts, once open's recovery pass is done.
		expect(await harness.runPass(context)).toMatchObject({ wakeAt: 61_000 });
		harness.resume();
		await settle();
		expect((await stateOf(harness, id))?.status).toBe("waiting");
		await harness.close(context);

		clock = 61_000;
		({ harness } = await open(await openNodeSqliteStorage(path)));
		harness.resume();
		expect((await harness.waitForTask(id, context)).state.outcome).toEqual({ status: "completed", result: 61_000 });
		await harness.close(context);
	});

	it("resumes a due sleep with its timer without publishWake, with background timers declined", async () => {
		let clock = 1_000;
		const timers = fakeTimers();
		const Nap = napTask(500);
		const { harness } = await openTasks(new MemoryStorage(), [Nap], { now: () => clock, setTimer: timers.setTimer });
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(() => timers.live().some((timer) => timer.delayMs === 500));
		expect((await stateOf(harness, id))?.status).toBe("waiting");
		clock = 1_500;
		timers
			.live()
			.find((timer) => timer.delayMs === 500)!
			.callback();
		// Polled: `waitForTask()` would call `resume()`, which wakes due sleeps itself.
		await eventually(async () => (await stateOf(harness, id))?.status === "terminal");
		expect(await stateOf(harness, id)).toMatchObject({ outcome: { status: "completed", result: 1_500 } });
		await harness.close(context);
	});

	it("runs a due sleep at any scheduling pass, before its timer fires", async () => {
		let clock = 1_000;
		const timers = fakeTimers();
		const Nap = napTask(500);
		const { harness, registry } = await openTasks(new MemoryStorage(), [Nap], {
			now: () => clock,
			setTimer: timers.setTimer,
		});
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(async () => (await stateOf(harness, id))?.status === "waiting");
		// The deadline passes, and an unrelated change starts a pass before the timer fires.
		clock = 1_500;
		registry.install(defineExtension({ name: "unrelated" }));
		// Polled: `waitForTask()` would call `resume()`, which wakes due sleeps itself.
		await eventually(async () => (await stateOf(harness, id))?.status === "terminal");
		expect(await stateOf(harness, id)).toMatchObject({ outcome: { status: "completed", result: 1_500 } });
		await harness.close(context);
	});

	it("disarms the sleep timer when the sleep is aborted and when the Harness closes", async () => {
		const clock = 1_000;
		const timers = fakeTimers();
		const Short = napTask(500);
		const { harness } = await openTasks(new MemoryStorage(), [Short], {
			now: () => clock,
			setTimer: timers.setTimer,
		});
		const aborted = await startNap(harness, Short);
		harness.resume();
		await eventually(() => timers.live().some((timer) => timer.delayMs === 500));
		await harness.abortTask(aborted, context);
		await harness.waitForTask(aborted, context);
		await eventually(() => timers.live().length === 0);
		const kept = await startNap(harness, Short);
		await eventually(async () => (await stateOf(harness, kept))?.status === "waiting");
		expect(timers.live().map((timer) => timer.delayMs)).toEqual([500]);
		await harness.close(context);
		expect(timers.live()).toEqual([]);
	});

	it("disarms an invocation's timers when it ends, and a late fire runs nothing", async () => {
		const fired: string[] = [];
		const Timed = defineTask<null, { phase: "run" }, null>({
			name: "test.timed",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: {
				run: async (_task, runtime, ctx) => {
					runtime.setTimer(() => fired.push("late"), 60_000);
					await runtime.commit(() => completed(null), ctx);
				},
			},
			abort: async () => {},
		});
		const timers = fakeTimers();
		const { harness } = await openTasks(new MemoryStorage(), [Timed], { setTimer: timers.setTimer });
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Timed, null, { ownership: { kind: "conversation" } }),
			context,
		);
		harness.resume();
		await harness.waitForTask(id, context);
		const timer = timers.armed.find((armed) => armed.delayMs === 60_000)!;
		await eventually(() => timer.disarmed);
		// A host timer that fires anyway: the ended invocation's callback does not run.
		timer.callback();
		expect(fired).toEqual([]);
		await harness.close(context);
	});

	it("arms no timer for sleeps with publishWake", async () => {
		let clock = 1_000;
		const timers = fakeTimers();
		const Nap = napTask(500);
		const { harness } = await openTasks(new MemoryStorage(), [Nap], {
			now: () => clock,
			setTimer: timers.setTimer,
			publishWake: async () => {},
		});
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(async () => (await stateOf(harness, id))?.status === "waiting");
		await settle();
		expect(timers.armed).toEqual([]);
		clock = 1_500;
		harness.resume();
		expect((await harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		await harness.close(context);
	});

	it("arms a foreground timer the host declines with setTimeout, reporting it once", async () => {
		const Nap = napTask(10);
		const { harness, reports } = await openTasks(new MemoryStorage(), [Nap], { setTimer: () => undefined });
		const first = await startNap(harness, Nap);
		const second = await startNap(harness, Nap);
		harness.resume();
		expect((await harness.waitForTask(first, context)).state.outcome.status).toBe("completed");
		expect((await harness.waitForTask(second, context)).state.outcome.status).toBe("completed");
		expect(reports.map((error) => (error as Error).message)).toEqual([
			"HarnessOptions.setTimer declined a timer that paces work; using setTimeout",
		]);
		await harness.close(context);
	});

	it("counts neither a wait on a sleeping task nor an outcome held over one as active", async () => {
		const clock = 1_000;
		const Child = napTask(500);
		type ParentState = { phase: "start" } | { phase: "joined"; child: TaskId<number> };
		const Parent = defineTask<{ hold: boolean }, ParentState, null>({
			name: "test.parent",
			version: 1,
			initial: () => ({ phase: "start" }),
			phases: {
				start: async (task, runtime, ctx) => {
					await runtime.commit(async (tx) => {
						const child = await tx.createTask(Child, null, { ownership: { kind: "task", taskId: task.id } });
						// Held: completes next, holding its outcome over the sleeping child. Otherwise waits for it.
						return task.input.hold
							? { status: "running", checkpoint: { phase: "joined", child } }
							: {
									status: "waiting",
									checkpoint: { phase: "joined", child },
									condition: { kind: "tasks", on: [child], policy: "allSettled" },
								};
					}, ctx);
				},
				joined: async (_task, runtime, ctx) => {
					await runtime.commit(() => completed(null), ctx);
				},
			},
			abort: async () => {},
		});
		for (const hold of [false, true]) {
			const { harness } = await openTasks(new MemoryStorage(), [Parent, Child], {
				now: () => clock,
				publishWake: async () => {},
			});
			const root = await harness.root(context);
			const id = await root.commit(
				(tx) => tx.createTask(Parent, { hold }, { ownership: { kind: "conversation" } }),
				context,
			);
			harness.resume();
			const parked = hold ? "completing" : "waiting";
			await eventually(async () => (await stateOf(harness, id))?.status === parked);
			expect(await harness.runPass(context)).toMatchObject({ wakeAt: 1_500 });
			await harness.close(context);
		}
	});

	it("runs the abort handler of a sleeping task without waiting for its time", async () => {
		const clock = 1_000;
		const Nap = napTask(60_000);
		const { harness } = await openTasks(new MemoryStorage(), [Nap], {
			now: () => clock,
			publishWake: async () => {},
		});
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(async () => (await stateOf(harness, id))?.status === "waiting");
		await harness.abortTask(id, context);
		expect((await harness.waitForTask(id, context)).state.outcome).toEqual({ status: "aborted", reason: "test" });
		await harness.close(context);
	});

	it("rejects invalid times and resumes an abort handler after its committed wait", async () => {
		let clock = 1_000;
		const errors: string[] = [];
		const Invalid = defineTask<null, { phase: "run" } | { phase: "next" }, null>({
			name: "test.invalid-sleep",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: {
				run: async (_task, runtime, ctx) => {
					for (const until of [Number.NaN, -1, Number.POSITIVE_INFINITY]) {
						await runtime
							.sleepUntil(until, { phase: "next" }, ctx)
							.catch((error: unknown) => errors.push((error as Error).message));
					}
					await runtime.commit(() => completed(null), ctx);
				},
				next: async () => {},
			},
			abort: async () => {},
		});
		const Aborting = defineTask<null, { phase: "run" } | { phase: "cleanup" }, null>({
			name: "test.sleep-in-abort",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: { run: (_task, runtime) => aborted(runtime.signal), cleanup: async () => {} },
			abort: async (task, runtime, ctx) => {
				if (task.state.checkpoint.phase === "run") {
					await runtime.sleepUntil(runtime.now() + 10, { phase: "cleanup" }, ctx);
				} else {
					await runtime.commit(() => abortedWith("test"), ctx);
				}
			},
		});
		const { harness } = await openTasks(new MemoryStorage(), [Invalid, Aborting], {
			now: () => clock,
			publishWake: async () => {},
		});
		const root = await harness.root(context);
		const id = await root.commit(
			(tx) => tx.createTask(Invalid, null, { ownership: { kind: "conversation" } }),
			context,
		);
		const aborting = await root.commit(
			(tx) => tx.createTask(Aborting, null, { ownership: { kind: "conversation" } }),
			context,
		);
		harness.resume();
		expect((await harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		await harness.abortTask(aborting, context);
		expect(await harness.runPass(context)).toMatchObject({ wakeAt: 1_010 });
		expect(await stateOf(harness, aborting)).toMatchObject({
			status: "waiting",
			mode: "abort",
			condition: { kind: "time", until: 1_010 },
		});
		clock = 1_010;
		await harness.runPass(context);
		expect((await harness.waitForTask(aborting, context)).state.outcome.status).toBe("aborted");
		expect(errors).toEqual([
			`Task ${id} cannot sleep until NaN: not a time in milliseconds`,
			`Task ${id} cannot sleep until -1: not a time in milliseconds`,
			`Task ${id} cannot sleep until Infinity: not a time in milliseconds`,
		]);
		await harness.close(context);
	});

	it("does not count a task without a definition as active", async () => {
		const Missing = napTask(10);
		const { harness } = await openTasks(new MemoryStorage(), []);
		const root = await harness.root(context);
		// Created through a token the registry does not hold: nothing can run it.
		await root.commit((tx) => tx.createTask(Missing, null, { ownership: { kind: "conversation" } }), context);
		expect(await harness.runPass(context)).toMatchObject({ wakeAt: null });
		harness.resume();
		await settle();
		expect(await harness.runPass(context)).toMatchObject({ wakeAt: null });
		await harness.close(context);
	});

	it("reports a throwing publishWake and offers the same wake again at the next pass", async () => {
		let clock = 1_000;
		const wakes: WakeSchedule[] = [];
		const failure = new Error("alarm failed");
		let failing = true;
		const Nap = napTask(1_000);
		const { harness, registry, reports } = await openTasks(new MemoryStorage(), [Nap], {
			now: () => clock,
			publishWake: async (wake) => {
				wakes.push(wake);
				if (failing && wake.wakeAt === 2_000) throw failure;
			},
		});
		const id = await startNap(harness, Nap);
		harness.resume();
		await eventually(() => wakes.some((wake) => wake.wakeAt === 2_000));
		expect(reports).toContain(failure);
		const pending = await harness.snapshot(WakeDoc, context);
		expect(pending?.wakeAt).toBe(2_000);
		expect(pending?.publishedRevision).not.toBe(pending?.revision);
		// The host recovers; an unrelated change offers the same, undelivered wake again.
		failing = false;
		const offered = wakes.length;
		registry.install(defineExtension({ name: "unrelated" }));
		await harness.runPass(context);
		await eventually(() => wakes.length > offered);
		expect(wakes.at(-1)).toMatchObject({ wakeAt: 2_000 });
		const acknowledged = await harness.snapshot(WakeDoc, context);
		expect(acknowledged?.publishedRevision).toBe(acknowledged?.revision);
		clock = 2_000;
		harness.resume();
		await harness.waitForTask(id, context);
		await eventually(() => wakes.at(-1)?.wakeAt === null);
		await harness.close(context);
	});
});
