import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createModels, fauxAssistantMessage } from "@earendil-works/pi-ai";
import {
	type EntryId,
	Harness,
	LiveDoc,
	type ModelRequestPort,
	StorageRejected,
	type StorageWrite,
	type TaskId,
	UsageDoc,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { chatSetup } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";

const harnesses = new Set<Harness>();
const directories = new Set<string>();

afterEach(async () => {
	const closed = await Promise.allSettled([...harnesses].map((harness) => harness.close(context)));
	harnesses.clear();
	await Promise.all([...directories].map((directory) => rm(directory, { recursive: true, force: true })));
	directories.clear();
	for (const result of closed) if (result.status === "rejected") throw result.reason;
});

async function failureIncident(harness: Harness, id: TaskId): Promise<EntryId> {
	const task = await harness.getTask(id, context);
	expect(task).toMatchObject({ abortRequested: false, state: { status: "waiting", mode: "run" } });
	if (task?.state.status !== "waiting" || task.state.condition.kind !== "failure")
		throw new Error("Expected retained polling failure");
	return task.state.condition.incident;
}

describe("deferred polling retains operation ownership", () => {
	it("cancels the same operation when the user cancels a failed poll instead of repairing it", async () => {
		const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
		setup.settings.stream = { deferred: true };
		setup.faux.setResponses([fauxAssistantMessage("unused remote answer")]);
		let now = 1000;
		const original = new Error("poll admission refused");
		const harness = await Harness.open(
			new ControlledStorage(),
			{
				models: setup.models,
				registry: setup.registry,
				settings: setup.settings,
				now: () => now,
				modelRequests: async (request) => {
					if (request.operation === "fetchDeferred") throw original;
					return { status: "ready", options: {}, close: async () => {} };
				},
			},
			context,
		);
		harnesses.add(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		const submission = await root.submit({ type: "input", content: "start remote work" }, context);
		await harness.runPass(context);
		const taskId = (await harness.snapshot(LiveDoc, root.id, context))!.run!.taskId;
		const before = (await harness.getTask(taskId, context))!;
		if (before.state.status !== "waiting") throw new Error("Expected deferred wait");
		const checkpoint = before.state.checkpoint;
		if (checkpoint === null || typeof checkpoint !== "object" || !("handle" in checkpoint))
			throw new Error("Expected deferred handle");
		const handle = checkpoint.handle;
		now = 61_000;
		await harness.runPass(context);
		await failureIncident(harness, taskId);
		await expect(root.waitForIdle(context)).rejects.toBe(original);
		await root.abort(context);
		expect(await submission.wait(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		expect(setup.faux.state.cancelledDeferred).toEqual([handle]);
		expect(setup.faux.state.deferredFetchCount).toBe(0);
		expect(setup.faux.state.callCount).toBe(1);
	});

	it("seals dispatch and rolls back poll evidence when recording the retained failure is rejected", async () => {
		const rejection = new StorageRejected("failure transaction refused");
		const storage = new (class extends ControlledStorage {
			override async commit(writes: readonly StorageWrite[], ctx: typeof context) {
				if (writes.some((write) => write.type === "entry" && write.value.kind === "pi.failure")) throw rejection;
				return super.commit(writes, ctx);
			}
		})();
		const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
		setup.settings.stream = { deferred: true };
		setup.faux.setResponses([fauxAssistantMessage("unused remote answer")]);
		let now = 1000;
		let polls = 0;
		const message = fauxAssistantMessage("", { stopReason: "error", errorMessage: "503 poll unavailable" });
		setup.models.fetchDeferred = async () => {
			polls++;
			return { ...message, usage: { ...message.usage, input: 5, totalTokens: 5 } };
		};
		const harness = await Harness.open(
			storage,
			{
				models: setup.models,
				registry: setup.registry,
				settings: setup.settings,
				now: () => now,
			},
			context,
		);
		harnesses.add(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		const submission = await root.submit({ type: "input", content: "start remote work" }, context);
		await harness.runPass(context);
		const taskId = (await harness.snapshot(LiveDoc, root.id, context))!.run!.taskId;
		now = 61_000;
		await expect(harness.runPass(context)).rejects.toMatchObject({
			errors: [expect.objectContaining({ message: "503 poll unavailable" }), rejection],
		});
		expect(await harness.getTask(taskId, context)).toMatchObject({
			state: { status: "running", checkpoint: { phase: "poll" } },
		});
		expect(await submission.status(context)).toMatchObject({ status: "placed" });
		expect(
			(await root.entries({}, 100, undefined, context)).items.some(
				(entry) => entry.kind === "pi.failure" || entry.kind === "pi.assistant",
			),
		).toBe(false);
		expect((await harness.snapshot(UsageDoc, root.id, context))?.models["faux/faux-1"]?.input ?? 0).toBe(0);
		await expect(harness.runPass(context)).rejects.toMatchObject({
			errors: [expect.objectContaining({ message: "503 poll unavailable" }), rejection],
		});
		expect(polls).toBe(1);
	});

	it.each(["provider", "acquisition", "poll", "close", "error response"] as const)(
		"reattaches the same handle after explicit repair of %s failure and replacement",
		async (stage) => {
			const directory = await mkdtemp(join(tmpdir(), "pi-poll-failure-"));
			directories.add(directory);
			const path = join(directory, "session.sqlite");
			const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
			setup.settings.stream = { deferred: true };
			setup.settings.retry = { enabled: true, maxRetries: 3, baseDelayMs: 1 };
			setup.faux.setResponses([fauxAssistantMessage("recovered response")]);
			let now = 1000;
			let repaired = false;
			let polls = 0;
			const original = new Error(`original ${stage} failure`);
			const models = createModels();
			models.setProvider(setup.faux.provider);
			const fetch = models.fetchDeferred.bind(models);
			models.fetchDeferred = async (model, handle, options) => {
				polls++;
				if (!repaired && stage === "poll") throw original;
				if (!repaired && stage === "error response") {
					const message = fauxAssistantMessage("", { stopReason: "error", errorMessage: "503 poll unavailable" });
					return { ...message, usage: { ...message.usage, input: 5, totalTokens: 5 } };
				}
				return fetch(model, handle, options);
			};
			const port: ModelRequestPort = async (request) => {
				if (!repaired && request.operation === "fetchDeferred" && stage === "acquisition") throw original;
				return {
					status: "ready",
					options: {},
					close: async () => {
						if (!repaired && request.operation === "fetchDeferred" && stage === "close") throw original;
					},
				};
			};
			const open = async () => {
				const harness = await Harness.open(
					await openNodeSqliteStorage(path),
					{
						models,
						registry: setup.registry,
						settings: setup.settings,
						now: () => now,
						modelRequests: port,
						onReport: (error) => setup.reports.push(error),
					},
					context,
				);
				harnesses.add(harness);
				const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
				return { harness, root };
			};
			let opened = await open();
			const submission = await opened.root.submit({ type: "input", content: "start remote work" }, context);
			await opened.harness.runPass(context);
			const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
			const before = await opened.harness.getTask(taskId, context);
			if (before?.state.status !== "waiting") throw new Error("Expected deferred operation wait");
			const checkpoint = before.state.checkpoint;
			if (stage === "provider") models.deleteProvider("faux");
			now = 61_000;
			await opened.harness.runPass(context);
			const incident = await failureIncident(opened.harness, taskId);
			expect((await opened.harness.getTask(taskId, context))!.state).toMatchObject({ checkpoint });
			expect(await submission.status(context)).toMatchObject({ status: "placed" });
			if (stage === "provider") await expect(opened.root.waitForIdle(context)).rejects.toThrow("faux");
			else if (stage === "error response")
				await expect(opened.root.waitForIdle(context)).rejects.toThrow("503 poll unavailable");
			else await expect(opened.root.waitForIdle(context)).rejects.toBe(original);
			const failedPolls = polls;
			await opened.harness.runPass(context);
			expect(polls).toBe(failedPolls);
			expect(setup.faux.state.callCount).toBe(1);
			await opened.harness.close(context);
			opened = await open();
			await opened.harness.runPass(context);
			expect(await failureIncident(opened.harness, taskId)).toBe(incident);
			expect(polls).toBe(failedPolls);
			repaired = true;
			models.setProvider(setup.faux.provider);
			expect(await opened.harness.retryTask(taskId, incident, context)).toBe("queued");
			expect(await opened.harness.retryTask(taskId, incident, context)).toBe("stale");
			await opened.harness.runPass(context);
			expect((await (await opened.harness.submission(submission.id, context))!.wait(context)).status).toBe("done");
			expect(setup.faux.state.callCount).toBe(1);
			expect(setup.faux.state.cancelledDeferred).toHaveLength(0);
			if (stage === "error response") {
				const entries = (await opened.root.entries({}, 100, undefined, context)).items;
				expect(
					entries.filter(
						(entry) => entry.model?.[0]?.role === "assistant" && entry.model[0].stopReason === "error",
					),
				).toHaveLength(1);
				expect(
					(await opened.harness.snapshot(UsageDoc, opened.root.id, context))?.models["faux/faux-1"]?.input,
				).toBeGreaterThanOrEqual(5);
			}
		},
	);
});
