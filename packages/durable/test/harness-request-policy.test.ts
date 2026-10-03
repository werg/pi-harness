import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	acceptReceipt,
	bindReceipt,
	defineExtension,
	defineTool,
	GenerationTask,
	Harness,
	type HarnessOptions,
	hook,
	LiveDoc,
	MemoryStorage,
	type ModelRequestTarget,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, chatSetup, waitFor } from "./chat-support.ts";
import { context } from "./session-support.ts";

const sessions: Harness[] = [];
const directories: string[] = [];
afterEach(async () => {
	await Promise.all(sessions.splice(0).map((harness) => harness.close(context)));
	await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

function setupPolicy() {
	const setup = chatSetup({ models: [{ id: "primary" }, { id: "fallback" }] });
	setup.settings.retry = { enabled: false };
	setup.registry.install(
		defineExtension({
			name: "policy",
			hooks: [
				hook(GenerationTask, {
					afterResponse: (message, _api, _ctx, request) => {
						if (
							message.stopReason === "error" &&
							request.model.id === "primary" &&
							message.errorMessage === "usage_limit_terminal"
						)
							return {
								retry: {
									model: { provider: "faux", modelId: "fallback" },
									thinkingLevel: "low",
									stream: { serviceTier: "default" },
								},
							};
					},
				}),
			],
		}),
	);
	return setup;
}

describe("native conversation request policy", () => {
	it("refuses an identical retry policy while retaining the actual failed provider response", async () => {
		const setup = setupPolicy();
		setup.registry.install(
			defineExtension({
				name: "policy",
				hooks: [
					hook(GenerationTask, {
						afterResponse: (_message, _api, _ctx, request) => ({
							retry: { model: { provider: request.model.provider, modelId: request.model.id } },
						}),
					}),
				],
			}),
		);
		setup.faux.setResponses([
			fauxAssistantMessage([], { stopReason: "error", errorMessage: "usage_limit_terminal" }),
		]);
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "primary" } } });
		expect(await (await root.submit({ type: "input", content: "original" }, context)).wait(context)).toMatchObject({
			status: "unanswered",
			detail: "Model retry selection must change the failed request policy",
		});
		expect((await allEntries(root)).filter((entry) => entry.model?.[0]?.role === "assistant")).toMatchObject([
			{ model: [{ stopReason: "error", errorMessage: "usage_limit_terminal" }] },
		]);
		expect(setup.faux.state.callCount).toBe(1);
	});
	it("pins separate channel stream preferences and clears them without changing an admitted request", async () => {
		const setup = setupPolicy();
		setup.settings.stream = { serviceTier: "auto", cacheRetention: "none" };
		const requests: ModelRequestTarget[] = [];
		const harness = await Harness.open(
			new MemoryStorage(),
			{
				...setup,
				modelRequests: async (request, api, ctx) => {
					requests.push(request);
					const key = `request:${request.taskId}`;
					await api.commit((tx) => bindReceipt(tx, key, "test"), ctx);
					return { status: "waiting", condition: { kind: "receipt", key, binding: "test" } };
				},
			},
			context,
		);
		sessions.push(harness);
		const first = await harness.root(context, {
			agent: { model: { provider: "faux", modelId: "primary" }, stream: { serviceTier: "priority" } },
		});
		const second = await harness.createConversation(
			{
				ownership: { kind: "ownerless" },
				agent: { model: { provider: "faux", modelId: "primary" }, stream: { serviceTier: "default" } },
			},
			context,
		);
		await first.submit({ type: "input", content: "first" }, context);
		await harness.runPass(context);
		await second.submit({ type: "input", content: "second" }, context);
		await harness.runPass(context);
		expect(requests.map((request) => request.options.serviceTier)).toEqual(["priority", "default"]);
		expect(requests.every((request) => request.options.cacheRetention === "none")).toBe(true);
		await first.configure({ stream: null }, context);
		expect(requests[0]!.options.serviceTier).toBe("priority");
		await first.abort(context);
		await first.submit({ type: "input", content: "new first" }, context);
		await harness.runPass(context);
		expect(requests[2]!.options.serviceTier).toBe("auto");
	});

	it("keeps a real failed assistant and fallback through the same run's tool handover, then starts fresh input on primary", async () => {
		const setup = setupPolicy();
		let calls = 0;
		const tool = defineTool({
			name: "work",
			description: "Real work",
			parameters: Type.Object({}),
			execute: async () => {
				calls++;
				return {};
			},
		});
		setup.registry.install(defineExtension({ name: "tools", tools: [tool] }));
		const requests: unknown[] = [];
		setup.faux.setResponses([
			(_messages, options, _state, model) => {
				requests.push({ model: model.id, options });
				return fauxAssistantMessage([], { stopReason: "error", errorMessage: "usage_limit_terminal" });
			},
			(_messages, options, _state, model) => {
				requests.push({ model: model.id, options });
				return fauxAssistantMessage([fauxToolCall("work", {}, { id: "real-call" })], { stopReason: "toolUse" });
			},
			(_messages, options, _state, model) => {
				requests.push({ model: model.id, options });
				return fauxAssistantMessage("fallback answer");
			},
			(_messages, options, _state, model) => {
				requests.push({ model: model.id, options });
				return fauxAssistantMessage("fresh primary answer");
			},
		]);
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		sessions.push(harness);
		const root = await harness.root(context, {
			agent: { model: { provider: "faux", modelId: "primary" }, stream: { serviceTier: "priority" } },
		});
		const original = await root.submit({ type: "input", content: "unattended original" }, context);
		expect(await original.wait(context)).toMatchObject({ status: "done" });
		expect(calls).toBe(1);
		expect(await harness.snapshot(LiveDoc, root.id, context)).toEqual({});
		expect(await (await root.submit({ type: "input", content: "fresh input" }, context)).wait(context)).toMatchObject(
			{ status: "done" },
		);
		expect(requests).toMatchObject([
			{ model: "primary", options: { serviceTier: "priority" } },
			{ model: "fallback", options: { serviceTier: "default", reasoning: "low" } },
			{ model: "fallback", options: { serviceTier: "default", reasoning: "low" } },
			{ model: "primary", options: { serviceTier: "priority" } },
		]);
		expect(
			(await allEntries(root))
				.filter((entry) => entry.model?.[0]?.role === "assistant")
				.map((entry) => entry.model![0]),
		).toMatchObject([
			{ model: "primary", stopReason: "error", errorMessage: "usage_limit_terminal" },
			{ model: "fallback", stopReason: "toolUse" },
			{ model: "fallback", stopReason: "stop" },
			{ model: "primary", stopReason: "stop" },
		]);
		expect(setup.reports).toEqual([]);
	});

	it("retains fallback request preparation and real original failure through SQLite credential waiting and replacement", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-request-policy-"));
		directories.push(directory);
		const setup = setupPolicy();
		setup.faux.setResponses([
			fauxAssistantMessage([], { stopReason: "error", errorMessage: "usage_limit_terminal" }),
			fauxAssistantMessage("authorized fallback"),
		]);
		const requests: ModelRequestTarget[] = [];
		let ready = false;
		const options: HarnessOptions = {
			...setup,
			modelRequests: async (request, api, ctx) => {
				requests.push(request);
				if (request.model.id !== "fallback" || ready)
					return { status: "ready", options: {}, close: async () => {} } as const;
				await api.commit((tx) => bindReceipt(tx, "credential", "original"), ctx);
				return {
					status: "waiting",
					condition: { kind: "receipt", key: "credential", binding: "original" },
				} as const;
			},
		};
		let harness = await Harness.open(await openNodeSqliteStorage(join(directory, "state.sqlite")), options, context);
		sessions.push(harness);
		let root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "primary" } } });
		const submission = await root.submit({ type: "input", content: "original" }, context);
		harness.resume();
		await waitFor(() => requests.length === 2);
		expect(requests.map((request) => request.model.id)).toEqual(["primary", "fallback"]);
		expect(await submission.status(context)).toMatchObject({ status: "placed" });
		expect((await harness.snapshot(LiveDoc, root.id, context))?.run?.requestSelection).toMatchObject({
			model: { modelId: "fallback" },
		});
		await harness.close(context);
		ready = true;
		harness = await Harness.open(await openNodeSqliteStorage(join(directory, "state.sqlite")), options, context);
		sessions.push(harness);
		root = await harness.root(context);
		await harness.commit((tx) => acceptReceipt(tx, "credential", "original", { ready: true }), context);
		expect(await (await harness.submission(submission.id, context))!.wait(context)).toMatchObject({ status: "done" });
		expect(requests.map((request) => request.model.id)).toEqual(["primary", "fallback", "fallback"]);
		expect(requests[2]).toMatchObject({
			taskId: requests[1]!.taskId,
			attempt: 2,
			options: { serviceTier: "default" },
		});
		expect((await allEntries(root)).filter((entry) => entry.model?.[0]?.role === "assistant")).toHaveLength(2);
	});

	it("does not turn a fallback credential failure into another fallback attempt", async () => {
		const setup = setupPolicy();
		setup.faux.setResponses([
			fauxAssistantMessage([], { stopReason: "error", errorMessage: "usage_limit_terminal" }),
			fauxAssistantMessage([], { stopReason: "error", errorMessage: "invalid_credentials" }),
		]);
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "primary" } } });
		expect(await (await root.submit({ type: "input", content: "original" }, context)).wait(context)).toMatchObject({
			status: "unanswered",
			reason: "model_error",
			detail: "invalid_credentials",
		});
		expect(setup.faux.state.callCount).toBe(2);
		expect(await harness.snapshot(LiveDoc, root.id, context)).toEqual({});
	});

	it("aborts the actual fallback credential wait and leaves the next genuine input on primary", async () => {
		const setup = setupPolicy();
		setup.faux.setResponses([
			fauxAssistantMessage([], { stopReason: "error", errorMessage: "usage_limit_terminal" }),
			fauxAssistantMessage("fresh primary"),
		]);
		const requests: ModelRequestTarget[] = [];
		const harness = await Harness.open(
			new MemoryStorage(),
			{
				...setup,
				modelRequests: async (request, api, ctx) => {
					requests.push(request);
					if (request.model.id === "primary") return { status: "ready", options: {}, close: async () => {} };
					await api.commit((tx) => bindReceipt(tx, "credential", "bound"), ctx);
					return { status: "waiting", condition: { kind: "receipt", key: "credential", binding: "bound" } };
				},
			},
			context,
		);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "primary" } } });
		const original = await root.submit({ type: "input", content: "original" }, context);
		harness.resume();
		await waitFor(() => requests.length === 2);
		await root.abort(context);
		expect(await original.wait(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		expect(await (await root.submit({ type: "input", content: "fresh" }, context)).wait(context)).toMatchObject({
			status: "done",
		});
		expect(requests.map((request) => request.model.id)).toEqual(["primary", "fallback", "primary"]);
		expect(setup.faux.state.callCount).toBe(2);
		expect(await harness.snapshot(LiveDoc, root.id, context)).toEqual({});
	});
});
