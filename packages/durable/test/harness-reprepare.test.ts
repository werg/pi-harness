import { createModels, fauxAssistantMessage, fauxProvider, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	createRegistry,
	defineExtension,
	defineTool,
	Harness,
	LiveDoc,
	MemoryStorage,
	type ModelRequestTarget,
} from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { allEntries, chatSetup, openChat } from "./chat-support.ts";
import { context } from "./session-support.ts";
import { deferred, settled } from "./task-support.ts";

async function fixture(holdConnection = false) {
	const before = fauxProvider({ provider: "before" });
	const after = fauxProvider({ provider: "after" });
	after.setResponses([fauxAssistantMessage("New provider answer")]);
	const models = createModels();
	models.setProvider(before.provider);
	models.setProvider(after.provider);
	const requests: ModelRequestTarget[] = [];
	const waiting = deferred();
	const entered = deferred();
	const released = deferred();
	const cancelled = deferred();
	const harness = await Harness.open(
		new MemoryStorage(),
		{
			models,
			registry: createRegistry(),
			modelRequests: async (request) => {
				requests.push(request);
				if (request.model.provider === "before") {
					if (holdConnection) {
						return {
							status: "ready",
							options: {},
							close: async () => {
								await released.promise;
							},
						};
					}
					return {
						status: "waiting",
						condition: {
							kind: "input",
							conversationId: request.conversationId,
							after: request.cutoff,
							kinds: ["credential:before"],
						},
					};
				}
				return { status: "ready", options: {}, close: async () => {} };
			},
		},
		context,
	);
	const stop = harness.subscribeCommits((commit) => {
		if (
			commit.changes.some(
				(change) =>
					change.type === "task" &&
					change.value.kind === "pi.generation" &&
					change.value.state.status === "waiting",
			)
		)
			waiting.resolve();
	});
	before.setResponses([
		async (_messages, options) => {
			entered.resolve();
			await new Promise<void>((resolve) =>
				options!.signal!.addEventListener(
					"abort",
					() => {
						cancelled.resolve();
						resolve();
					},
					{ once: true },
				),
			);
			return fauxAssistantMessage("", { stopReason: "aborted" });
		},
	]);
	const root = await harness.root(context, {
		agent: { model: { provider: "before", modelId: before.getModel().id } },
	});
	const input = await root.submit(
		{ type: "input", requestId: "original", content: "Original human message" },
		context,
	);
	if (holdConnection) await entered.promise;
	else await waiting.promise;
	stop();
	const originalTask = requests[0]!.taskId;
	return { harness, root, input, originalTask, requests, before, after, released, cancelled };
}

describe("native generation reprepare", () => {
	it("refuses a generation executing tools without cancelling them or changing the run", async () => {
		const setup = chatSetup();
		const entered = deferred(),
			release = deferred();
		let aborted = false;
		const tool = defineTool({
			name: "held",
			description: "Owned tool work",
			parameters: Type.Object({}),
			execute: async (_args, _api, ctx) => {
				ctx.abortSignal!.addEventListener(
					"abort",
					() => {
						aborted = true;
					},
					{ once: true },
				);
				entered.resolve();
				await release.promise;
				return { output: [] };
			},
		});
		const extension = defineExtension({ name: "held", tools: [tool] });
		setup.registry.install(extension);
		setup.faux.setResponses([
			fauxAssistantMessage([fauxToolCall("held", {})], { stopReason: "toolUse" }),
			fauxAssistantMessage("Done"),
		]);
		const { harness, root } = await openChat(new MemoryStorage(), setup);
		try {
			await root.configure({ extensions: [extension], tools: [tool] }, context);
			const input = await root.submit({ type: "input", content: "Original" }, context);
			await entered.promise;
			const live = await harness.snapshot(LiveDoc, root.id, context);
			expect((await harness.getTask(live!.run!.taskId, context))!.state).toMatchObject({
				checkpoint: { phase: "tools" },
			});
			expect(await root.reprepare(live!.run!.taskId, context)).toBe(false);
			expect(await harness.snapshot(LiveDoc, root.id, context)).toEqual(live);
			expect(aborted).toBe(false);
			release.resolve();
			expect(await input.wait(context)).toMatchObject({ status: "done" });
		} finally {
			release.resolve();
			await harness.close(context);
		}
	});
	it("supersedes a parked request with current settings while preserving exactly the original input", async () => {
		const f = await fixture();
		try {
			await f.root.configure({ model: { provider: "after", modelId: f.after.getModel().id } }, context);
			expect(await f.root.reprepare(f.originalTask, context)).toBe(true);
			expect(await f.input.wait(context)).toMatchObject({ status: "done" });
			expect(f.requests.map((request) => request.model.provider)).toEqual(["before", "after"]);
			expect(f.requests[1]!.taskId).not.toBe(f.originalTask);
			expect((await f.harness.getTask(f.originalTask, context))?.state).toMatchObject({
				status: "terminal",
				outcome: { status: "aborted" },
			});
			const users = (await allEntries(f.root))
				.flatMap((entry) => entry.model ?? [])
				.filter((message) => message.role === "user");
			expect(users).toEqual([expect.objectContaining({ content: "Original human message" })]);
			expect(f.before.state.callCount).toBe(0);
		} finally {
			await f.harness.close(context);
		}
	});

	it("joins the superseded connection before dispatching a successor", async () => {
		const f = await fixture(true);
		let restarting: Promise<boolean> | undefined;
		try {
			await f.root.configure({ model: { provider: "after", modelId: f.after.getModel().id } }, context);
			restarting = f.root.reprepare(f.originalTask, context);
			await f.cancelled.promise;
			expect(await settled(restarting)).toBe(false);
			expect(f.requests).toHaveLength(1);
			f.released.resolve();
			expect(await restarting).toBe(true);
			await f.input.wait(context);
			expect(f.requests).toHaveLength(2);
		} finally {
			f.released.resolve();
			await restarting;
			await f.harness.close(context);
		}
	});

	it.each(["conversation", "task"] as const)(
		"lets explicit %s cancellation win during reprepare cleanup",
		async (scope) => {
			const f = await fixture(true);
			let restarting: Promise<boolean> | undefined;
			let stopping: Promise<unknown> | undefined;
			try {
				restarting = f.root.reprepare(f.originalTask, context);
				await f.cancelled.promise;
				stopping = scope === "conversation" ? f.root.abort(context) : f.harness.abortTask(f.originalTask, context);
				await f.root.commit(() => undefined, context);
				expect(await settled(stopping)).toBe(false);
			} finally {
				f.released.resolve();
				await Promise.all([restarting, stopping]);
				expect(await f.input.status(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
				expect(f.requests).toHaveLength(1);
				await f.harness.close(context);
			}
		},
	);

	it("guards the target generation and keeps later follow-ups queued until the original answer", async () => {
		const f = await fixture();
		try {
			const follow = await f.root.submit({ type: "input", content: "Later input", whenBusy: "followUp" }, context);
			expect(await f.root.reprepare((f.originalTask + 100) as typeof f.originalTask, context)).toBe(false);
			expect((await f.harness.snapshot(LiveDoc, f.root.id, context))?.run?.inputs).toEqual([f.input.id]);
			await f.root.configure({ model: { provider: "after", modelId: f.after.getModel().id } }, context);
			f.after.setResponses([fauxAssistantMessage("Original answer"), fauxAssistantMessage("Follow-up answer")]);
			expect(await f.root.reprepare(f.originalTask, context)).toBe(true);
			expect(await f.input.wait(context)).toMatchObject({ status: "done" });
			expect(await follow.wait(context)).toMatchObject({ status: "done" });
			expect(await f.root.reprepare(f.originalTask, context)).toBe(false);
		} finally {
			await f.harness.close(context);
		}
	});
});
