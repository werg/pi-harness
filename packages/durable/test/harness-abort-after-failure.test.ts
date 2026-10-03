import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineTask, type Harness, MemoryStorage, type Storage } from "@earendil-works/pi-durable";
import { expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { context } from "./session-support.ts";
import { deferred, openTasks } from "./task-support.ts";

it.each(["memory", "sqlite"] as const)(
	"joins authoritative cancellation after a retained run failure in %s",
	async (backend) => {
		const directory = backend === "sqlite" ? await mkdtemp(join(tmpdir(), "pi-abort-after-failure-")) : undefined;
		const original = new Error("original operation acknowledgement lost");
		const cleanup = new Error("original cancellation readback failed");
		const started = deferred();
		const release = deferred();
		let repaired = false;
		let attempts = 0;
		const task = defineTask<null, { phase: "work" }, null>({
			name: "test.abort-after-failure",
			version: 1,
			initial: () => ({ phase: "work" }),
			phases: { work: (_record, runtime, ctx) => runtime.parkFailure(original, ctx) },
			abort: async (_record, runtime, ctx) => {
				attempts++;
				started.resolve();
				await release.promise;
				if (!repaired) throw cleanup;
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		const memory = new MemoryStorage();
		const storage = (): Promise<Storage> =>
			directory ? openNodeSqliteStorage(join(directory, "session.sqlite")) : Promise.resolve(memory);
		let harness: Harness | undefined;
		try {
			harness = (await openTasks(await storage(), [task])).harness;
			let root = await harness.root(context);
			const id = await root.commit(
				(tx) => tx.createTask(task, null, { ownership: { kind: "conversation" } }),
				context,
			);
			await harness.runPass(context);
			const failed = await harness.getTask(id, context);
			if (failed?.state.status !== "waiting" || failed.state.condition.kind !== "failure")
				throw Error("No run incident");
			const runIncident = failed.state.condition.incident;
			await expect(harness.waitForTask(id, context)).rejects.toBe(original);
			if (directory) {
				await harness.close(context);
				harness = (await openTasks(await storage(), [task])).harness;
				root = await harness.root(context);
			}
			expect(await harness.abortTask(id, context)).toBe("marked");
			const observers = [harness.waitForTask(id, context), root.waitForIdle(context), harness.waitForIdle(context)];
			const settled = observers.map(() => false);
			const joined = Promise.allSettled(
				observers.map((observer, index) =>
					observer.finally(() => {
						settled[index] = true;
					}),
				),
			);
			await started.promise;
			expect(settled).toEqual([false, false, false]);
			expect(await harness.retryTask(id, runIncident, context)).toBe("stale");
			release.resolve();
			const results = await joined;
			for (const result of results) {
				expect(result.status).toBe("rejected");
				if (result.status === "rejected") expect(result.reason).toBe(cleanup);
			}
			const cancelled = await harness.getTask(id, context);
			if (cancelled?.state.status !== "waiting" || cancelled.state.condition.kind !== "failure")
				throw Error("No cleanup incident");
			expect(cancelled.state.mode).toBe("abort");
			await expect(harness.waitForTask(id, context)).rejects.toBe(cleanup);
			repaired = true;
			expect(await harness.retryTask(id, cancelled.state.condition.incident, context)).toBe("queued");
			expect((await harness.waitForTask(id, context)).state.outcome.status).toBe("aborted");
			await root.waitForIdle(context);
			expect(attempts).toBe(2);
			const entries = (await root.entries({}, 100, undefined, context)).items;
			expect(
				entries.filter((entry) => entry.kind === "pi.failure").sort((left, right) => left.id - right.id),
			).toMatchObject([
				{ id: runIncident, data: { error: { message: original.message } } },
				{ id: cancelled.state.condition.incident, data: { error: { message: cleanup.message } } },
			]);
		} finally {
			release.resolve();
			await harness?.close(context);
			if (directory) await rm(directory, { recursive: true, force: true });
		}
	},
);
