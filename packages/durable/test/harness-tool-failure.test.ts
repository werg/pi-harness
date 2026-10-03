import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	bindReceipt,
	defineExtension,
	defineTool,
	type EntryId,
	Harness,
	LiveDoc,
	type Storage,
	StorageRejected,
	type StorageWrite,
	type TaskId,
	type ToolDiagnostic,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { chatSetup, openChat, toolsNamed } from "./chat-support.ts";
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
		throw new Error("Expected retained tool observation failure");
	return task.state.condition.incident;
}

function scenario(stage: "execute" | "environment" | "agent" | "cancel", replay: "safe" | "unsafe" = "unsafe") {
	const setup = chatSetup();
	const original = new Error(`original ${stage} failure`);
	const admissions: unknown[] = [];
	const observations: unknown[] = [];
	const cancellations: unknown[] = [];
	const details: Promise<unknown>[] = [];
	const state = { now: 1000, observing: false, repaired: false };
	const tool = defineTool({
		name: "owned-operation",
		description: "Attach to an admitted external operation",
		parameters: Type.Object({ value: Type.String() }),
		replay,
		outputLimits: { maxBytes: 64, maxLines: 2, retain: "tail" },
		execute: async (args, api, ctx) => {
			if (api.continuation === undefined) {
				await api.commit((tx) => bindReceipt(tx, "operation", "accepted-operation"), ctx);
				admissions.push(structuredClone(args));
				api.output("admitted\n");
				await api.details({ operation: "accepted-operation", stage: "admitted" }, ctx);
				return { wait: { kind: "time" as const, until: 2000 }, continuation: { operation: "accepted-operation" } };
			}
			observations.push(structuredClone({ args, continuation: api.continuation }));
			if (!state.repaired) {
				api.output("observed\n");
				api.output("failed observation\n");
				details.push(
					api.details({ operation: "accepted-operation", stage: "observation failed" }, ctx).then(
						() => "committed",
						(error: unknown) => error,
					),
				);
				api.diagnostic({ severity: "warn", code: "observation", message: "The operation is still owned" });
				throw original;
			}
			api.output("completed\n");
			return { isError: false };
		},
		cancel: async (args, api, ctx) => {
			cancellations.push(structuredClone({ args, continuation: api.continuation }));
			if (stage === "cancel" && !state.repaired) {
				api.output("release attempted\n");
				api.output("release failed\n");
				details.push(
					api.details({ operation: "accepted-operation", stage: "release failed" }, ctx).then(
						() => "committed",
						(error: unknown) => error,
					),
				);
				api.diagnostic({ severity: "error", code: "release", message: "Operation release is not confirmed" });
				throw original;
			}
			api.output("released\n");
			return { isError: false };
		},
	});
	setup.registry.install(defineExtension({ name: "owned-tool", tools: [tool] }));
	const snapshot = setup.registry.snapshot();
	const extension = snapshot.extension.bind(snapshot);
	snapshot.extension = (name) => {
		if (stage === "agent" && state.observing && !state.repaired) throw original;
		return extension(name);
	};
	setup.faux.setResponses([
		fauxAssistantMessage([fauxToolCall(tool.name, { value: "pinned" }, { id: "call" })], { stopReason: "toolUse" }),
		fauxAssistantMessage("operation completed"),
	]);
	const open = async (storage: Storage) => {
		const harness = await Harness.open(
			storage,
			{
				models: setup.models,
				registry: setup.registry,
				settings: setup.settings,
				now: () => state.now,
				env: () => {
					if (stage === "environment" && state.observing && !state.repaired) throw original;
					return undefined;
				},
				onReport: (error) => setup.reports.push(error),
			},
			context,
		);
		harnesses.add(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		await root.configure({ tools: toolsNamed(setup, tool.name) }, context);
		return { harness, root };
	};
	return { setup, original, admissions, observations, cancellations, details, state, open };
}

describe("tool continuation failures retain operation ownership", () => {
	it("does not append restored diagnostics again when a resumed tool publishes progress", async () => {
		const setup = chatSetup();
		let now = 1000;
		setup.now = () => now;
		let observed: readonly ToolDiagnostic[] | undefined;
		const diagnostic = { severity: "info" as const, code: "accepted", message: "Operation accepted" };
		const tool = defineTool({
			name: "diagnostic-operation",
			description: "Observe one owned operation",
			parameters: Type.Object({}),
			execute: async (_args, api, ctx) => {
				if (api.continuation === undefined) {
					api.diagnostic(diagnostic);
					return {
						wait: { kind: "time" as const, until: 2000 },
						continuation: { operation: "accepted-operation" },
					};
				}
				await api.details({ stage: "resumed" }, ctx);
				const live = await api.snapshot(LiveDoc, api.conversationId, ctx);
				observed = live!.tools!.find((slot) => slot.taskId === api.taskId)!.diagnostics;
				return { isError: false };
			},
			cancel: async () => ({ isError: false }),
		});
		setup.registry.install(defineExtension({ name: "diagnostics", tools: [tool] }));
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall(tool.name, {}, { id: "call" })], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		const { harness, root } = await openChat(new ControlledStorage(), setup);
		harnesses.add(harness);
		await root.configure({ tools: toolsNamed(setup, tool.name) }, context);
		const submission = await root.submit({ type: "input", content: "run operation" }, context);
		await harness.runPass(context);
		now = 2000;
		await harness.runPass(context);
		expect(observed).toEqual([diagnostic]);
		expect(await submission.wait(context)).toMatchObject({ status: "done" });
	});

	it.each([
		{ stage: "execute", replay: "unsafe" },
		{ stage: "execute", replay: "safe" },
		{ stage: "environment", replay: "unsafe" },
		{ stage: "agent", replay: "unsafe" },
	] as const)(
		"reattaches after $stage failure with $replay replay policy and SQLite replacement",
		async ({ stage, replay }) => {
			const directory = await mkdtemp(join(tmpdir(), "pi-tool-failure-"));
			directories.add(directory);
			const path = join(directory, "session.sqlite");
			const fixture = scenario(stage, replay);
			let opened = await fixture.open(await openNodeSqliteStorage(path));
			const submission = await opened.root.submit({ type: "input", content: "run operation" }, context);
			await opened.harness.runPass(context);
			const live = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!;
			const taskId = live.tools![0]!.taskId!;
			const ownerId = live.run!.taskId;
			fixture.state.now = 2000;
			fixture.state.observing = true;
			await opened.harness.runPass(context);
			const incident = await failureIncident(opened.harness, taskId);
			if (stage === "agent" || stage === "environment") expect(fixture.observations).toHaveLength(0);
			await expect(opened.root.waitForIdle(context)).rejects.toBe(fixture.original);
			await expect(opened.harness.waitForTask(ownerId, context)).rejects.toBe(fixture.original);
			expect(await submission.status(context)).toMatchObject({ status: "placed" });
			expect(await opened.harness.getTask(ownerId, context)).toMatchObject({
				abortRequested: false,
				state: { status: "waiting" },
			});
			const output = stage === "execute" ? "observed\nfailed observation\n" : "admitted\n";
			const droppedBytes = stage === "execute" ? 9 : 0;
			expect(await opened.harness.getTask(taskId, context)).toMatchObject({
				state: {
					checkpoint: {
						phase: "execute",
						arguments: { value: "pinned" },
						replay,
						continuation: { operation: "accepted-operation" },
						output: { text: output, totalBytes: new TextEncoder().encode(output).length + droppedBytes },
					},
				},
			});
			expect((await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.tools![0]).toMatchObject({
				status: "running",
				output,
				droppedBytes,
				details: { stage: stage === "execute" ? "observation failed" : "admitted" },
			});
			if (stage === "execute") expect(await fixture.details[0]).toBe(fixture.original);
			expect(
				(await opened.root.entries({}, 100, undefined, context)).items.some(
					(entry) => entry.kind === "pi.tool-result",
				),
			).toBe(false);
			const attempts = fixture.observations.length;
			await opened.harness.runPass(context);
			expect(fixture.observations).toHaveLength(attempts);
			await opened.harness.close(context);
			opened = await fixture.open(await openNodeSqliteStorage(path));
			await opened.harness.runPass(context);
			expect(await failureIncident(opened.harness, taskId)).toBe(incident);
			await expect(opened.root.waitForIdle(context)).rejects.toThrow(fixture.original.message);
			expect(fixture.observations).toHaveLength(attempts);
			fixture.state.repaired = true;
			expect(await opened.harness.retryTask(taskId, incident, context)).toBe("queued");
			expect(await opened.harness.retryTask(taskId, incident, context)).toBe("stale");
			await opened.harness.runPass(context);
			expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
				status: "done",
			});
			expect(fixture.admissions).toEqual([{ value: "pinned" }]);
			expect(fixture.observations.at(-1)).toEqual({
				args: { value: "pinned" },
				continuation: { operation: "accepted-operation" },
			});
			expect(fixture.cancellations).toHaveLength(0);
			const entries = (await opened.root.entries({}, 100, undefined, context)).items;
			expect(entries.filter((entry) => entry.kind === "pi.failure")).toHaveLength(1);
			const results = entries.filter((entry) => entry.kind === "pi.tool-result");
			expect(results).toHaveLength(1);
			expect(results[0]!.model?.[0]).toMatchObject({
				role: "toolResult",
				isError: false,
				content:
					stage === "execute"
						? [
								{ type: "text", text: "failed observation\ncompleted\n" },
								{
									type: "text",
									text: "<harness>\n[warn] The operation is still owned\n[warn] Output truncated to its end: 2 lines, 18 bytes dropped\n</harness>",
								},
							]
						: [
								expect.objectContaining({
									text: "admitted\ncompleted\n",
								}),
							],
				details: { stage: stage === "execute" ? "observation failed" : "admitted" },
			});
			if (stage === "execute")
				expect(results[0]!.data).toMatchObject({
					diagnostics: [
						expect.objectContaining({ code: "observation" }),
						expect.objectContaining({ code: "truncated" }),
					],
				});
		},
	);

	it("cancels the retained operation instead of resubmitting after a failed continuation", async () => {
		const fixture = scenario("execute");
		const { harness, root } = await fixture.open(new ControlledStorage());
		const submission = await root.submit({ type: "input", content: "run operation" }, context);
		await harness.runPass(context);
		const taskId = (await harness.snapshot(LiveDoc, root.id, context))!.tools![0]!.taskId!;
		fixture.state.now = 2000;
		await harness.runPass(context);
		const incident = await failureIncident(harness, taskId);
		await root.abort(context);
		expect(await submission.wait(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		expect(fixture.admissions).toHaveLength(1);
		expect(fixture.observations).toHaveLength(1);
		expect(fixture.cancellations).toEqual([
			{ args: { value: "pinned" }, continuation: { operation: "accepted-operation" } },
		]);
		expect(await harness.retryTask(taskId, incident, context)).toBe("terminal");
		expect(
			(await root.entries({}, 100, undefined, context)).items.filter((entry) => entry.kind === "pi.tool-result"),
		).toHaveLength(1);
	});

	it("retains failed cancellation progress and rejoins cleanup after SQLite replacement", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-tool-failure-"));
		directories.add(directory);
		const path = join(directory, "session.sqlite");
		const fixture = scenario("cancel");
		let opened = await fixture.open(await openNodeSqliteStorage(path));
		const submission = await opened.root.submit({ type: "input", content: "run operation" }, context);
		await opened.harness.runPass(context);
		const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.tools![0]!.taskId!;
		await expect(opened.root.abort(context)).rejects.toBe(fixture.original);
		await opened.harness.runPass(context);
		const failed = await opened.harness.getTask(taskId, context);
		expect(failed).toMatchObject({
			abortRequested: true,
			state: {
				status: "waiting",
				mode: "abort",
				checkpoint: {
					continuation: { operation: "accepted-operation" },
					output: { text: "release attempted\nrelease failed\n", totalBytes: 42 },
				},
			},
		});
		if (failed?.state.status !== "waiting" || failed.state.condition.kind !== "failure")
			throw new Error("Expected retained cancellation failure");
		const incident = failed.state.condition.incident;
		expect((await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.tools![0]).toMatchObject({
			status: "running",
			output: "release attempted\nrelease failed\n",
			droppedBytes: 9,
			details: { stage: "release failed" },
			diagnostics: [expect.objectContaining({ code: "release" })],
		});
		expect(await fixture.details[0]).toBe(fixture.original);
		await opened.harness.close(context);
		opened = await fixture.open(await openNodeSqliteStorage(path));
		await opened.harness.runPass(context);
		expect(fixture.cancellations).toHaveLength(1);
		fixture.state.repaired = true;
		expect(await opened.harness.retryTask(taskId, incident, context)).toBe("queued");
		await opened.harness.runPass(context);
		expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
			status: "unanswered",
			reason: "aborted",
		});
		expect(fixture.cancellations).toHaveLength(2);
		expect(fixture.cancellations[1]).toEqual(fixture.cancellations[0]);
		expect(fixture.admissions).toHaveLength(1);
		const results = (await opened.root.entries({}, 100, undefined, context)).items.filter(
			(entry) => entry.kind === "pi.tool-result",
		);
		expect(results).toHaveLength(1);
		expect(results[0]!.model?.[0]).toMatchObject({
			content: expect.arrayContaining([{ type: "text", text: "release failed\nreleased\n" }]),
			details: { stage: "release failed" },
		});
	});

	it("retains the admitted operation when its terminal result commit is rejected", async () => {
		const rejection = new StorageRejected("result transaction refused");
		let rejectResult = true;
		const storage = new (class extends ControlledStorage {
			override async commit(writes: readonly StorageWrite[], ctx: typeof context) {
				if (
					rejectResult &&
					writes.some((write) => write.type === "entry" && write.value.kind === "pi.tool-result")
				) {
					rejectResult = false;
					throw rejection;
				}
				return super.commit(writes, ctx);
			}
		})();
		const fixture = scenario("execute");
		const { harness, root } = await fixture.open(storage);
		const submission = await root.submit({ type: "input", content: "run operation" }, context);
		await harness.runPass(context);
		const taskId = (await harness.snapshot(LiveDoc, root.id, context))!.tools![0]!.taskId!;
		fixture.state.repaired = true;
		fixture.state.now = 2000;
		await harness.runPass(context);
		const incident = await failureIncident(harness, taskId);
		await expect(root.waitForIdle(context)).rejects.toBe(rejection);
		expect(await harness.getTask(taskId, context)).toMatchObject({
			state: { checkpoint: { continuation: { operation: "accepted-operation" } } },
		});
		expect(await submission.status(context)).toMatchObject({ status: "placed" });
		expect(
			(await root.entries({}, 100, undefined, context)).items.some((entry) => entry.kind === "pi.tool-result"),
		).toBe(false);
		expect(await harness.retryTask(taskId, incident, context)).toBe("queued");
		await harness.runPass(context);
		expect(await submission.wait(context)).toMatchObject({ status: "done" });
		expect(fixture.admissions).toHaveLength(1);
		expect(fixture.observations).toHaveLength(2);
		expect(fixture.observations[1]).toEqual(fixture.observations[0]);
		expect(
			(await root.entries({}, 100, undefined, context)).items.filter((entry) => entry.kind === "pi.tool-result"),
		).toHaveLength(1);
	});

	it("rolls back continuation progress with the rejected incident and seals dispatch", async () => {
		const rejection = new StorageRejected("failure transaction refused");
		let attempted: readonly StorageWrite[] | undefined;
		const storage = new (class extends ControlledStorage {
			override async commit(writes: readonly StorageWrite[], ctx: typeof context) {
				if (writes.some((write) => write.type === "entry" && write.value.kind === "pi.failure")) {
					attempted = structuredClone(writes);
					throw rejection;
				}
				return super.commit(writes, ctx);
			}
		})();
		const fixture = scenario("execute");
		const { harness, root } = await fixture.open(storage);
		const submission = await root.submit({ type: "input", content: "run operation" }, context);
		await harness.runPass(context);
		const taskId = (await harness.snapshot(LiveDoc, root.id, context))!.tools![0]!.taskId!;
		const before = (await harness.getTask(taskId, context))!;
		if (before.state.status !== "waiting") throw new Error("Expected admitted operation wait");
		fixture.state.now = 2000;
		await expect(harness.runPass(context)).rejects.toMatchObject({ errors: [fixture.original, rejection] });
		expect(await harness.getTask(taskId, context)).toMatchObject({
			state: { status: "running", checkpoint: before.state.checkpoint },
		});
		expect((await harness.snapshot(LiveDoc, root.id, context))!.tools![0]!.details).toEqual({
			operation: "accepted-operation",
			stage: "admitted",
		});
		expect(await submission.status(context)).toMatchObject({ status: "placed" });
		expect(attempted).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ type: "document.change" }),
				expect.objectContaining({
					type: "task",
					value: expect.objectContaining({
						id: taskId,
						state: expect.objectContaining({
							status: "waiting",
							checkpoint: expect.objectContaining({
								output: expect.objectContaining({ text: "observed\nfailed observation\n" }),
							}),
						}),
					}),
				}),
				expect.objectContaining({ type: "entry", value: expect.objectContaining({ kind: "pi.failure" }) }),
			]),
		);
		expect(
			(await root.entries({}, 100, undefined, context)).items.some(
				(entry) => entry.kind === "pi.failure" || entry.kind === "pi.tool-result",
			),
		).toBe(false);
		await expect(harness.runPass(context)).rejects.toMatchObject({ errors: [fixture.original, rejection] });
		expect(fixture.admissions).toHaveLength(1);
		expect(fixture.observations).toHaveLength(1);
	});
});
