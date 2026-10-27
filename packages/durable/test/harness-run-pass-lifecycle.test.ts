import { withAbortSignal } from "@earendil-works/chord/context";
import { defineTask, MemoryStorage } from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { context } from "./session-support.ts";
import { aborted, deferred, openTasks, settled } from "./task-support.ts";

describe("runPass lifecycle", () => {
	it("joins admitted work and the resulting completion before returning", async () => {
		const entered = deferred();
		const release = deferred();
		const Task = defineTask<null, { phase: "run" }, null>({
			name: "test.pass-lifecycle",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: {
				run: async (_task, runtime, ctx) => {
					entered.resolve();
					await release.promise;
					await runtime.commit(
						() => ({ status: "terminal", outcome: { status: "completed", result: null } }),
						ctx,
					);
				},
			},
			abort: async () => {},
		});
		const { harness } = await openTasks(new MemoryStorage(), [Task]);
		try {
			const root = await harness.root(context);
			const id = await root.commit(
				(tx) => tx.createTask(Task, null, { ownership: { kind: "conversation" } }),
				context,
			);
			const pass = harness.runPass(context);
			await entered.promise;
			expect(await settled(pass)).toBe(false);
			const controller = new AbortController();
			const cancelled = harness.runPass(withAbortSignal(controller.signal, context));
			const reason = new Error("caller cancelled");
			controller.abort(reason);
			await expect(cancelled).rejects.toBe(reason);
			expect(await settled(pass)).toBe(false);
			release.resolve();
			await pass;
			expect((await harness.getTask(id, context))?.state.status).toBe("terminal");
		} finally {
			release.resolve();
			await harness.close(context);
		}
	});
	it("wakes a pass when the Harness closes and joins its interrupted handler", async () => {
		const entered = deferred();
		const exited = deferred();
		const Task = defineTask<null, { phase: "run" }, null>({
			name: "test.pass-close",
			version: 1,
			initial: () => ({ phase: "run" }),
			phases: {
				run: async (_task, runtime) => {
					entered.resolve();
					try {
						await aborted(runtime.signal);
					} finally {
						exited.resolve();
					}
				},
			},
			abort: async () => {},
		});
		const { harness } = await openTasks(new MemoryStorage(), [Task]);
		const root = await harness.root(context);
		await root.commit((tx) => tx.createTask(Task, null, { ownership: { kind: "conversation" } }), context);
		const pass = harness.runPass(context);
		await entered.promise;
		const rejected = expect(pass).rejects.toThrow();
		await harness.close(context);
		await rejected;
		await exited.promise;
	});
});
