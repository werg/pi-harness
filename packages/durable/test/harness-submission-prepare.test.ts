import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	ConversationBusy,
	defineDoc,
	defineTask,
	InboxDoc,
	LiveDoc,
	ReadAfterWrite,
	ResetEntry,
	StorageRejected,
	type SubmissionId,
	type SubmissionPrepare,
	type TaskId,
	UserEntry,
	type UserInput,
} from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, chatSetup, openChat, unanswered } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";
import { deferred, settled } from "./task-support.ts";

type FeedbackState = {
	pending: { id: string; text: string }[];
	accepted: { deliveryId: string; submissionId: SubmissionId; feedbackIds: string[] }[];
};

const Feedback = defineDoc<FeedbackState>({
	kind: "test.feedback",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ pending: [], accepted: [] }),
});

const Background = defineTask<Record<string, never>, { phase: "parked" }, null>({
	name: "test.background",
	version: 1,
	initial: () => ({ phase: "parked" }),
	phases: { parked: async () => {} },
	abort: async () => {},
});

describe("transactional submission preparation", () => {
	it("prepares exact feedback and acceptance with the native input identity in one commit", async () => {
		const storage = new ControlledStorage();
		const setup = chatSetup();
		const busy = unanswered();
		setup.faux.setResponses([busy.step]);
		const { harness, root } = await openChat(storage, setup);
		try {
			await root.commit(async (tx) => {
				const feedback = await tx.doc(Feedback, root.id);
				feedback.pending.push({ id: "represented", text: "Fix card A" }, { id: "later", text: "Fix card B" });
			}, context);
			let taskId: TaskId | undefined;
			const prepare: SubmissionPrepare<UserInput> = async (tx, submissionId) => {
				// Document access works after the native identity is staged; consume only represented feedback.
				const feedback = await tx.doc(Feedback, root.id);
				const notes = feedback.pending.filter((note) => note.id === "represented");
				feedback.pending = feedback.pending.filter((note) => note.id !== "represented");
				feedback.accepted.push({
					deliveryId: "delivery-1",
					submissionId,
					feedbackIds: notes.map((note) => note.id),
				});
				taskId = await tx.createTask(Background, {}, { ownership: { kind: "conversation" }, background: true });
				return [...notes.map((note) => note.text), "hello"].join("\n\n");
			};
			const submission = await root.submit({ type: "input", requestId: "delivery-1", content: prepare }, context);
			await busy.reached;
			const record = await submission.status(context);
			if (record.status !== "placed") throw new Error(`Unexpected ${record.status}`);
			expect(await root.commit((tx) => tx.entry(UserEntry, record.entry), context)).toMatchObject({
				conversationId: root.id,
				model: [{ role: "user", content: "Fix card A\n\nhello", timestamp: expect.any(Number) }],
			});
			expect(await harness.snapshot(Feedback, root.id, context)).toEqual({
				pending: [{ id: "later", text: "Fix card B" }],
				accepted: [{ deliveryId: "delivery-1", submissionId: submission.id, feedbackIds: ["represented"] }],
			});
			expect(await harness.getTask(taskId!, context)).toMatchObject({ conversationId: root.id, background: true });
			expect((await harness.snapshot(LiveDoc, root.id, context))?.run?.inputs).toEqual([submission.id]);
			const batch = storage.commits.find((writes) =>
				writes.some((write) => write.type === "submission" && write.value.id === submission.id),
			)!;
			expect(batch).toEqual(
				expect.arrayContaining([
					expect.objectContaining({ type: "entry", value: expect.objectContaining({ id: record.entry }) }),
					expect.objectContaining({ type: "task", value: expect.objectContaining({ id: taskId }) }),
					expect.objectContaining({ type: "document.change" }),
					expect.objectContaining({ type: "submission", value: expect.objectContaining({ status: "placed" }) }),
				]),
			);
		} finally {
			await harness.close(context);
		}
	});

	it("rolls back product changes and reserved native identity with the original callback error", async () => {
		const storage = new ControlledStorage();
		const { harness, root } = await openChat(storage, chatSetup());
		try {
			await root.commit(
				async (tx) => void (await tx.doc(Feedback, root.id)).pending.push({ id: "f1", text: "note" }),
				context,
			);
			const before = storage.commits.length;
			const failure = new Error("exact delivery is invalid");
			let failedId: SubmissionId | undefined;
			await expect(
				root.submit(
					{
						type: "input",
						requestId: "failed",
						content: async (tx, id) => {
							failedId = id;
							const feedback = await tx.doc(Feedback, root.id);
							feedback.pending = [];
							feedback.accepted.push({ deliveryId: "failed", submissionId: id, feedbackIds: ["f1"] });
							await tx.appendEntry(root.id, { kind: "product.partial" });
							throw failure;
						},
					},
					context,
				),
			).rejects.toBe(failure);
			expect(storage.commits).toHaveLength(before);
			expect(await harness.submission(failedId!, context)).toBeUndefined();
			expect(await root.commit((tx) => tx.submissionByRequest(root.id, "failed"), context)).toBeUndefined();
			expect(await allEntries(root)).toEqual([]);
			expect(await harness.snapshot(LiveDoc, root.id, context)).toEqual({});
			expect(await harness.snapshot(Feedback, root.id, context)).toEqual({
				pending: [{ id: "f1", text: "note" }],
				accepted: [],
			});
			const retry = await root.submit({ type: "write", requestId: "failed", entry: { kind: "retry" } }, context);
			expect(retry.id).not.toBe(failedId);
			expect((await retry.status(context)).status).toBe("done");
		} finally {
			await harness.close(context);
		}
	});

	it("skips product preparation on request replay and type conflict, even while busy", async () => {
		const setup = chatSetup();
		const busy = unanswered();
		setup.faux.setResponses([busy.step]);
		const storage = new ControlledStorage();
		const { harness, root } = await openChat(storage, setup);
		try {
			let calls = 0;
			const first = await root.submit(
				{
					type: "input",
					requestId: "delivery",
					content: async (tx, id) => {
						calls++;
						(await tx.doc(Feedback, root.id)).accepted.push({
							deliveryId: "delivery",
							submissionId: id,
							feedbackIds: [],
						});
						return "hello";
					},
				},
				context,
			);
			await busy.reached;
			const before = storage.commits.length;
			const again = await root.submit(
				{
					type: "input",
					requestId: "delivery",
					whenBusy: "reject",
					content: () => {
						calls++;
						throw new Error("replay must not prepare");
					},
				},
				context,
			);
			expect(again.id).toBe(first.id);
			await expect(
				root.submit(
					{
						type: "write",
						requestId: "delivery",
						entry: () => {
							calls++;
							throw new Error("type conflict must not prepare");
						},
					},
					context,
				),
			).rejects.toThrow("already identifies a submission of type input");
			expect(calls).toBe(1);
			expect(storage.commits).toHaveLength(before);
			expect((await harness.snapshot(Feedback, root.id, context))?.accepted).toHaveLength(1);
		} finally {
			await harness.close(context);
		}
	});

	it("keeps busy rejection and native queued modes, committing product acceptance only for admitted inputs", async () => {
		const setup = chatSetup();
		const busy = unanswered();
		setup.faux.setResponses([busy.step]);
		const storage = new ControlledStorage();
		const { harness, root } = await openChat(storage, setup);
		try {
			await root.submit({ type: "input", content: "start" }, context);
			await busy.reached;
			const before = storage.commits.length;
			let rejectedCalls = 0;
			await expect(
				root.submit(
					{
						type: "input",
						whenBusy: "reject",
						content: () => {
							rejectedCalls++;
							return "not admitted";
						},
					},
					context,
				),
			).rejects.toBeInstanceOf(ConversationBusy);
			expect(rejectedCalls).toBe(0);
			expect(storage.commits).toHaveLength(before);
			const steer = await root.submit(
				{
					type: "input",
					whenBusy: "steer",
					content: async (tx, id) => {
						(await tx.doc(Feedback, root.id)).accepted.push({
							deliveryId: "steer",
							submissionId: id,
							feedbackIds: [],
						});
						return "steering";
					},
				},
				context,
			);
			const follow = await root.submit({ type: "input", whenBusy: "followUp", content: () => "follow-up" }, context);
			const write = await root.submit({ type: "write", entry: () => ({ kind: "note", data: "passive" }) }, context);
			expect(await harness.snapshot(InboxDoc, root.id, context)).toEqual({
				items: [
					{ id: steer.id, mode: "steer", content: "steering" },
					{ id: follow.id, mode: "followUp", content: "follow-up" },
					{ id: write.id, mode: "write", entry: { kind: "note", data: "passive" } },
				],
			});
			for (const submission of [steer, follow, write])
				expect((await submission.status(context)).status).toBe("queued");
			expect((await harness.snapshot(Feedback, root.id, context))?.accepted).toEqual([
				{ deliveryId: "steer", submissionId: steer.id, feedbackIds: [] },
			]);
		} finally {
			await harness.close(context);
		}
	});

	it("deduplicates after SQLite owner replacement without repeating product writes", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-submission-prepare-"));
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		let opened = await openChat(await openNodeSqliteStorage(path), setup);
		let calls = 0;
		try {
			const first = await opened.root.submit(
				{
					type: "write",
					requestId: "delivery",
					entry: async (tx, id) => {
						calls++;
						(await tx.doc(Feedback, opened.root.id)).accepted.push({
							deliveryId: "delivery",
							submissionId: id,
							feedbackIds: [],
						});
						return { kind: "accepted" };
					},
				},
				context,
			);
			const firstRecord = await first.status(context);
			await opened.harness.close(context);
			opened = await openChat(await openNodeSqliteStorage(path), setup);
			const replay = await opened.root.submit(
				{
					type: "write",
					requestId: "delivery",
					entry: () => {
						calls++;
						throw new Error("replacement must reuse original receipt");
					},
				},
				context,
			);
			expect(replay.id).toBe(first.id);
			expect(await replay.status(context)).toEqual(firstRecord);
			expect(calls).toBe(1);
			expect((await opened.harness.snapshot(Feedback, opened.root.id, context))?.accepted).toEqual([
				{ deliveryId: "delivery", submissionId: first.id, feedbackIds: [] },
			]);
			expect(await allEntries(opened.root)).toHaveLength(1);
		} finally {
			await opened.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});

	it("joins an admitted async preparation through caller cancellation and owner close", async () => {
		const storage = new ControlledStorage();
		const { harness, root } = await openChat(storage, chatSetup());
		const entered = deferred();
		const release = deferred();
		const controller = new AbortController();
		let acceptedId: SubmissionId | undefined;
		const admitting = root.submit(
			{
				type: "write",
				requestId: "joining",
				entry: async (tx, id) => {
					acceptedId = id;
					const feedback = await tx.doc(Feedback, root.id);
					feedback.accepted.push({ deliveryId: "joining", submissionId: id, feedbackIds: [] });
					entered.resolve();
					await release.promise;
					return { kind: "joined" };
				},
			},
			{ ...context, abortSignal: controller.signal },
		);
		try {
			await entered.promise;
			controller.abort(new Error("caller stopped waiting"));
			const closing = harness.close(context);
			expect(await settled(closing)).toBe(false);
			let lateCalls = 0;
			await expect(
				root.submit(
					{
						type: "write",
						entry: () => {
							lateCalls++;
							return { kind: "late" };
						},
					},
					context,
				),
			).rejects.toThrow("Session is closed");
			expect(lateCalls).toBe(0);
			release.resolve();
			expect((await admitting).id).toBe(acceptedId);
			await closing;
			const batch = storage.commits.find((writes) =>
				writes.some((write) => write.type === "submission" && write.value.id === acceptedId),
			)!;
			expect(batch).toEqual(
				expect.arrayContaining([
					expect.objectContaining({ type: "entry", value: expect.objectContaining({ kind: "joined" }) }),
					expect.objectContaining({
						type: "submission",
						value: expect.objectContaining({ id: acceptedId, status: "done" }),
					}),
					expect.objectContaining({
						type: "document.create",
						record: expect.objectContaining({ kind: Feedback.definition.kind }),
					}),
				]),
			);
		} finally {
			release.resolve();
			await admitting.catch(() => {});
			await harness.close(context);
		}
	});

	it("propagates deterministic Storage rejection and rolls back the prepared product document", async () => {
		const storage = new ControlledStorage();
		const { harness, root } = await openChat(storage, chatSetup());
		try {
			await root.commit(async (tx) => void (await tx.doc(Feedback, root.id)), context);
			const failure = new StorageRejected("no batch admitted");
			storage.failNextCommit(failure);
			let failedId: SubmissionId | undefined;
			await expect(
				root.submit(
					{
						type: "write",
						requestId: "rejected",
						entry: async (tx, id) => {
							failedId = id;
							(await tx.doc(Feedback, root.id)).accepted.push({
								deliveryId: "rejected",
								submissionId: id,
								feedbackIds: [],
							});
							return { kind: "rejected" };
						},
					},
					context,
				),
			).rejects.toBe(failure);
			expect(await harness.submission(failedId!, context)).toBeUndefined();
			expect(await allEntries(root)).toEqual([]);
			expect(await harness.snapshot(Feedback, root.id, context)).toEqual({ pending: [], accepted: [] });
			expect(
				(await (await root.submit({ type: "write", entry: { kind: "healthy" } }, context)).status(context)).status,
			).toBe("done");
		} finally {
			await harness.close(context);
		}
	});

	it("keeps kernel table-read restrictions and rolls back invalid preparation", async () => {
		const { harness, root } = await openChat(new ControlledStorage(), chatSetup());
		try {
			await expect(
				root.submit(
					{
						type: "write",
						entry: async (tx) => {
							await tx.conversation(root.id);
							return { kind: "invalid" };
						},
					},
					context,
				),
			).rejects.toBeInstanceOf(ReadAfterWrite);
			expect(await allEntries(root)).toEqual([]);
			expect((await harness.inspect(context)).submissions).toEqual([]);
		} finally {
			await harness.close(context);
		}
	});

	it("preserves passive placement, optional JSON fields, and stale-head settlement for prepared writes", async () => {
		const { harness, root } = await openChat(new ControlledStorage(), chatSetup());
		try {
			const old = await root.commit((tx) => tx.appendEntry(root.id, { kind: "old" }), context);
			const reset = await root.commit((tx) => tx.appendEntry(ResetEntry, root.id, { head: "self" }), context);
			const stale = await root.submit({ type: "write", entry: () => ({ kind: "summary", head: old.id }) }, context);
			expect(await stale.status(context)).toEqual({
				id: stale.id,
				conversationId: root.id,
				type: "write",
				status: "unanswered",
				reason: "stale",
			});
			const fresh = await root.submit(
				{ type: "write", entry: () => ({ kind: "summary", head: reset.id, model: undefined }) },
				context,
			);
			const placed = await fresh.status(context);
			if (placed.status !== "done" || placed.type !== "write") throw new Error(`Unexpected ${placed.status}`);
			expect(await root.commit((tx) => tx.entry(placed.entry), context)).toEqual({
				id: placed.entry,
				conversationId: root.id,
				kind: "summary",
				head: reset.id,
			});
			expect((await harness.inspect(context)).tasks).toEqual([]);
		} finally {
			await harness.close(context);
		}
	});
});
