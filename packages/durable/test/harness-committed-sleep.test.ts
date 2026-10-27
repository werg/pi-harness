import { createModels } from "@earendil-works/pi-ai";
import {
	createRegistry,
	defineExtension,
	defineTask,
	Harness,
	MemoryStorage,
	WakeDoc,
	type WakeSchedule,
} from "@earendil-works/pi-durable";
import { expect, it } from "vitest";
import { context } from "./session-support.ts";
import { completed } from "./task-support.ts";

it("persists one time wait and acknowledged host wake until due, then resumes once", async () => {
	let now = 1_000;
	let runs = 0;
	const wakes: WakeSchedule[] = [];
	const Task = defineTask<null, { phase: "before" | "after" }, null>({
		name: "test.committed-sleep",
		version: 1,
		initial: () => ({ phase: "before" }),
		phases: {
			before: (_task, runtime, ctx) => runtime.sleepUntil(2_000, { phase: "after" }, ctx),
			after: async (_task, runtime, ctx) => {
				runs++;
				await runtime.commit(() => completed(null), ctx);
			},
		},
		abort: async (_task, runtime, ctx) => {
			await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
		},
	});
	const registry = createRegistry();
	registry.install(defineExtension({ name: "sleeper", tasks: [Task] }));
	const harness = await Harness.open(
		new MemoryStorage(),
		{
			models: createModels(),
			registry,
			now: () => now,
			publishWake: async (wake) => {
				wakes.push(wake);
			},
		},
		context,
	);
	try {
		const root = await harness.root(context);
		const id = await root.commit((tx) => tx.createTask(Task, null, { ownership: { kind: "conversation" } }), context);
		expect((await harness.runPass(context)).wakeAt).toBe(2_000);
		expect((await harness.getTask(id, context))?.state).toMatchObject({
			status: "waiting",
			condition: { kind: "time", until: 2_000 },
			checkpoint: { phase: "after" },
		});
		const stored = await harness.snapshot(WakeDoc, context);
		expect(stored?.wakeAt).toBe(2_000);
		expect(stored?.publishedRevision).toBe(stored?.revision);
		expect(wakes.at(-1)?.wakeAt).toBe(2_000);
		now = 1_999;
		await harness.runPass(context);
		expect(runs).toBe(0);
		now = 2_000;
		await harness.runPass(context);
		expect(runs).toBe(1);
		expect((await harness.getTask(id, context))?.state.status).toBe("terminal");
		expect((await harness.snapshot(WakeDoc, context))?.wakeAt).toBeNull();
	} finally {
		await harness.close(context);
	}
});
