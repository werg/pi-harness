import { fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	defineExtension,
	defineTool,
	InboxDoc,
	LiveDoc,
	MemoryStorage,
	StorageRejected,
} from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { chatSetup, openChat, unanswered } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";
import { deferred, settled } from "./task-support.ts";

describe("native queue flush", () => {
	it("joins tool cancellation before placing a follow-up, and an explicit Stop wins during that join", async () => {
		const setup = chatSetup();
		const entered = deferred(),
			cancelled = deferred(),
			release = deferred();
		const tool = defineTool({
			name: "held",
			description: "Owned work",
			parameters: Type.Object({}),
			execute: async (_args, _api, ctx) => {
				ctx.abortSignal!.addEventListener("abort", () => cancelled.resolve(), { once: true });
				entered.resolve();
				await release.promise;
				return { content: [] };
			},
		});
		const extension = defineExtension({ name: "held", tools: [tool] });
		setup.registry.install(extension);
		setup.faux.setResponses([fauxAssistantMessage([fauxToolCall("held", {})], { stopReason: "toolUse" })]);
		const { harness, root } = await openChat(new MemoryStorage(), setup);
		let flushing: Promise<void> | undefined, stopping: Promise<void> | undefined;
		try {
			await root.configure({ extensions: [extension], tools: [tool] }, context);
			const original = await root.submit({ type: "input", content: "original" }, context);
			await entered.promise;
			const follow = await root.submit({ type: "input", content: "later", whenBusy: "followUp" }, context);
			flushing = root.flush(context);
			await cancelled.promise;
			expect(await settled(flushing)).toBe(false);
			expect(await follow.status(context)).toMatchObject({ status: "queued" });
			stopping = root.abort(context);
			expect(await settled(stopping)).toBe(false);
			release.resolve();
			await Promise.all([flushing, stopping]);
			expect(await follow.status(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
			expect(await original.status(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
			expect(setup.faux.state.callCount).toBe(1);
		} finally {
			release.resolve();
			await Promise.allSettled([flushing, stopping]);
			await harness.close(context);
		}
	});

	it("rolls back a rejected flush admission without withdrawing inputs or cancelling the request", async () => {
		const setup = chatSetup(),
			first = unanswered();
		setup.faux.setResponses([first.step]);
		const storage = new ControlledStorage();
		const { harness, root } = await openChat(storage, setup);
		try {
			const original = await root.submit({ type: "input", content: "original" }, context);
			await first.reached;
			const follow = await root.submit({ type: "input", content: "later", whenBusy: "followUp" }, context);
			const failure = new StorageRejected("Flush admission rejected");
			storage.failNextCommit(failure);
			await expect(root.flush(context)).rejects.toBe(failure);
			expect(await original.status(context)).toMatchObject({ status: "placed" });
			expect(await follow.status(context)).toMatchObject({ status: "queued" });
			expect((await harness.snapshot(LiveDoc, root.id, context))?.run?.interruption).toBeUndefined();
		} finally {
			await harness.close(context);
		}
	});
	it("joins the old request, advances exactly one follow-up, and retains later queued inputs", async () => {
		const setup = chatSetup();
		const first = unanswered(),
			next = unanswered();
		setup.faux.setResponses([first.step, next.step, fauxAssistantMessage("last")]);
		const { harness, root } = await openChat(new MemoryStorage(), setup);
		try {
			const original = await root.submit({ type: "input", content: "original" }, context);
			await first.reached;
			const one = await root.submit({ type: "input", content: "one", whenBusy: "followUp" }, context);
			const two = await root.submit({ type: "input", content: "two", whenBusy: "followUp" }, context);
			await root.flush(context);
			await next.reached;
			expect(await original.status(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
			expect(await one.status(context)).toMatchObject({ status: "placed" });
			expect(await two.status(context)).toMatchObject({ status: "queued" });
			expect((await harness.snapshot(InboxDoc, root.id, context))?.items.map((item) => item.id)).toEqual([two.id]);
			await root.flush(context);
			expect(await two.wait(context)).toMatchObject({ status: "done" });
		} finally {
			await harness.close(context);
		}
	});

	it("consumes steering before follow-ups and preserves the original input run", async () => {
		const setup = chatSetup();
		const first = unanswered(),
			next = unanswered();
		setup.faux.setResponses([first.step, next.step]);
		const { harness, root } = await openChat(new MemoryStorage(), setup);
		try {
			const original = await root.submit({ type: "input", content: "original" }, context);
			await first.reached;
			const follow = await root.submit({ type: "input", content: "later", whenBusy: "followUp" }, context);
			const steer = await root.submit({ type: "input", content: "now", whenBusy: "steer" }, context);
			await Promise.all([root.flush(context), root.flush(context)]);
			await next.reached;
			expect((await harness.snapshot(LiveDoc, root.id, context))?.run?.inputs).toEqual([original.id, steer.id]);
			expect(await original.status(context)).toMatchObject({ status: "placed" });
			expect(await steer.status(context)).toMatchObject({ status: "placed" });
			expect(await follow.status(context)).toMatchObject({ status: "queued" });
			await root.abort(context);
			expect(await follow.status(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		} finally {
			await harness.close(context);
		}
	});
});
