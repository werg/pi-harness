import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import {
	defineDoc,
	defineExtension,
	defineTask,
	type EntryId,
	Harness,
	type HarnessOptions,
	InboxDoc,
	type Storage,
	StorageRejected,
	type StorageWrite,
	type SubmissionId,
	type TaskId,
	UserEntry,
	WakeDoc,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, chatSetup } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";

const sessions: Harness[] = [];
const directories: string[] = [];
afterEach(async () => {
	const results = await Promise.allSettled(sessions.splice(0).map((session) => session.close(context)));
	await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
	for (const result of results) if (result.status === "rejected") throw result.reason;
});
class RejectRevisionStorage extends ControlledStorage {
	failure: StorageRejected | undefined;
	override commit(writes: readonly StorageWrite[], context: Parameters<Storage["commit"]>[1]) {
		if (
			this.failure &&
			writes.some((write) => write.type === "entry" && write.value.kind === "test.authorized-edit")
		) {
			const error = this.failure;
			this.failure = undefined;
			return Promise.reject(error);
		}
		return super.commit(writes, context);
	}
}
const Original = defineDoc<{
	source: { id: SubmissionId; author: string; content: string }[];
	reads: { id: SubmissionId; entry: EntryId }[];
	acknowledged: EntryId[];
}>({
	kind: "test.source-intent",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ source: [], reads: [], acknowledged: [] }),
});
const ReadAck = defineTask<{ entry: EntryId }, { phase: "ack" }, null>({
	name: "test.native-read-ack",
	version: 1,
	initial: () => ({ phase: "ack" }),
	phases: {
		ack: async (task, rt, ctx) => {
			await rt.commit(async (tx) => {
				(await tx.doc(Original, rt.conversationId)).acknowledged.push(task.input.entry);
				return { status: "terminal", outcome: { status: "completed", result: null } };
			}, ctx);
		},
	},
	abort: async () => {},
});
async function fixture(storage = new ControlledStorage(), prepareCommit?: HarnessOptions["prepareCommit"]) {
	const setup = chatSetup();
	setup.faux.setResponses([fauxAssistantMessage("first answer"), fauxAssistantMessage("second answer")]);
	setup.registry.install(defineExtension({ name: "read-ack", tasks: [ReadAck] }));
	const options: HarnessOptions = {
		models: setup.models,
		registry: setup.registry,
		publishWake: async () => {},
		...(prepareCommit ? { prepareCommit } : {}),
	};
	const harness = await Harness.open(storage, options, context);
	sessions.push(harness);
	const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
	const submit = (requestId: string, content: string) =>
		root.submit(
			{
				type: "input",
				requestId,
				content: async (tx, id) => {
					(await tx.doc(Original, root.id)).source.push({ id, author: "user:original", content });
					return content;
				},
			},
			context,
		);
	return { harness, root, options, setup, submit };
}
const prepareAck: NonNullable<HarnessOptions["prepareCommit"]> = async (tx, staged) => {
	for (const submission of staged.submissions) {
		if (submission.type !== "input" || submission.status !== "placed") continue;
		expect(Object.isFrozen(submission)).toBe(true);
		const entry = staged.entries.find((entry) => entry.id === submission.entry);
		if (!entry) continue;
		expect(entry.kind).toBe(UserEntry.kind);
		const source = await tx.doc(Original, submission.conversationId);
		source.reads.push({ id: submission.id, entry: submission.entry });
		await tx.createTask(
			ReadAck,
			{ entry: submission.entry },
			{ conversationId: submission.conversationId, ownership: { kind: "conversation" }, background: true },
		);
	}
};

describe("native queued input correction", () => {
	it("preserves the original request and immutable intent while placing only the corrected native content", async () => {
		const f = await fixture();
		await f.submit("first", "first");
		const queued = await f.submit("second", "original");
		expect(await queued.status(context)).toMatchObject({ status: "queued" });
		await f.root.commit(async (tx) => {
			const intent = (await tx.doc(Original, f.root.id)).source.find((source) => source.id === queued.id);
			if (intent?.author !== "user:original") throw new Error("Foreign source author");
			expect(await tx.reviseQueuedInput(queued.id, { kind: "replace", content: "corrected" })).toBe("updated");
			await tx.appendEntry(f.root.id, {
				kind: "test.authorized-edit",
				data: { target: queued.id, author: intent.author, content: "corrected" },
			});
		}, context);
		const replay = await f.submit("second", "original");
		expect(replay.id).toBe(queued.id);
		expect((await f.harness.snapshot(Original, f.root.id, context))?.source).toEqual([
			expect.objectContaining({ content: "first" }),
			{ id: queued.id, author: "user:original", content: "original" },
		]);
		await f.harness.runPass(context);
		expect(
			(await allEntries(f.root))
				.filter((entry) => entry.kind === UserEntry.kind)
				.map((entry) => entry.model?.[0]?.content),
		).toEqual(["first", "corrected"]);
		expect((await queued.status(context)).status).toBe("done");
	});
	it("compares the source author in the same transaction and rejects without changing native queued content", async () => {
		const f = await fixture();
		await f.submit("first", "first");
		const queued = await f.submit("second", "original");
		const failure = new Error("Foreign source author");
		await expect(
			f.root.commit(async (tx) => {
				const source = (await tx.doc(Original, f.root.id)).source.find((source) => source.id === queued.id);
				if (source?.author !== "user:foreign") throw failure;
				await tx.reviseQueuedInput(queued.id, { kind: "replace", content: "foreign" });
			}, context),
		).rejects.toBe(failure);
		expect((await f.harness.snapshot(InboxDoc, f.root.id, context))?.items).toMatchObject([
			{ id: queued.id, content: "original" },
		]);
	});
	it("establishes native UserEntry placement as read-wins and never rewrites a placed or answered input", async () => {
		const f = await fixture();
		const placed = await f.submit("read", "original");
		expect(
			await f.root.commit(
				(tx) => tx.reviseQueuedInput(placed.id, { kind: "replace", content: "late edit" }),
				context,
			),
		).toBe("already_placed");
		expect(await f.root.commit((tx) => tx.reviseQueuedInput(placed.id, { kind: "withdraw" }), context)).toBe(
			"already_placed",
		);
		await f.harness.runPass(context);
		expect(
			await f.root.commit(
				(tx) => tx.reviseQueuedInput(placed.id, { kind: "replace", content: "later edit" }),
				context,
			),
		).toBe("already_placed");
		expect((await allEntries(f.root)).find((entry) => entry.kind === UserEntry.kind)?.model?.[0]?.content).toBe(
			"original",
		);
	});
	it("withdraws queued input once, sees the latest candidate in the same transaction and never resurrects on request replay", async () => {
		const f = await fixture();
		await f.submit("first", "first");
		const queued = await f.submit("second", "original");
		await f.root.commit(async (tx) => {
			expect(await tx.reviseQueuedInput(queued.id, { kind: "withdraw" })).toBe("withdrawn");
			expect(await tx.reviseQueuedInput(queued.id, { kind: "replace", content: "resurrection" })).toBe("settled");
		}, context);
		expect(await queued.status(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
		expect((await f.submit("second", "original")).id).toBe(queued.id);
		expect((await f.harness.snapshot(InboxDoc, f.root.id, context))?.items).toEqual([]);
		await f.harness.runPass(context);
		expect((await allEntries(f.root)).filter((entry) => entry.kind === UserEntry.kind)).toHaveLength(1);
	});
	it("retains original native abort behavior and refuses to correct a passive write", async () => {
		const f = await fixture();
		await f.submit("first", "first");
		const input = await f.submit("second", "queued");
		expect(await input.abort(context)).toBe("aborted");
		const write = await f.root.submit({ type: "write", requestId: "note", entry: { kind: "note" } }, context);
		await expect(
			f.root.commit((tx) => tx.reviseQueuedInput(write.id, { kind: "replace", content: "invalid" }), context),
		).rejects.toThrow("is not an input");
		expect(await write.abort(context)).toBe("aborted");
		expect(
			await f.root.commit((tx) => tx.reviseQueuedInput(999 as SubmissionId, { kind: "withdraw" }), context),
		).toBe("not_found");
	});
	it("rolls back queued correction, audit and derived work with the original storage rejection", async () => {
		const storage = new RejectRevisionStorage();
		const f = await fixture(storage);
		await f.submit("first", "first");
		const queued = await f.submit("second", "original");
		const failure = new StorageRejected("revision rejected");
		storage.failure = failure;
		let taskId: TaskId | undefined;
		await expect(
			f.root.commit(async (tx) => {
				await tx.reviseQueuedInput(queued.id, { kind: "replace", content: "rejected" });
				const audit = await tx.appendEntry(f.root.id, {
					kind: "test.authorized-edit",
					data: { target: queued.id },
				});
				taskId = await tx.createTask(
					ReadAck,
					{ entry: audit.id },
					{ ownership: { kind: "conversation" }, background: true },
				);
			}, context),
		).rejects.toBe(failure);
		expect((await f.harness.snapshot(InboxDoc, f.root.id, context))?.items).toMatchObject([
			{ id: queued.id, content: "original" },
		]);
		expect((await allEntries(f.root)).some((entry) => entry.kind === "test.authorized-edit")).toBe(false);
		expect(await f.harness.getTask(taskId!, context)).toBeUndefined();
		expect((await queued.status(context)).status).toBe("queued");
	});
	it("makes staged native submission placement, UserEntry and read-ack task one atomic batch with a frozen final record", async () => {
		const storage = new ControlledStorage();
		const f = await fixture(storage, prepareAck);
		const submission = await f.submit("input", "actual input");
		const record = await submission.status(context);
		if (record.status !== "placed") throw new Error("Input not placed");
		const batch = storage.commits.find((writes) =>
			writes.some((write) => write.type === "entry" && write.value.id === record.entry),
		)!;
		expect(batch).toEqual(
			expect.arrayContaining([
				expect.objectContaining({
					type: "entry",
					value: expect.objectContaining({ id: record.entry, kind: UserEntry.kind }),
				}),
				expect.objectContaining({
					type: "submission",
					value: expect.objectContaining({ id: submission.id, status: "placed", entry: record.entry }),
				}),
				expect.objectContaining({
					type: "task",
					value: expect.objectContaining({
						kind: ReadAck.definition.name,
						input: { entry: record.entry },
						background: true,
					}),
				}),
			]),
		);
		expect(await f.harness.snapshot(Original, f.root.id, context)).toMatchObject({
			reads: [{ id: submission.id, entry: record.entry }],
			acknowledged: [],
		});
		expect(await f.harness.snapshot(WakeDoc, context)).toBeDefined();
		await f.harness.runPass(context);
		expect(await f.harness.snapshot(Original, f.root.id, context)).toMatchObject({ acknowledged: [record.entry] });
	});
	it("rolls back original native placement and staged acknowledgement together with the original hook error", async () => {
		const original = new Error("read acknowledgement projection rejected");
		let reject = true;
		const f = await fixture(new ControlledStorage(), async (tx, staged, ctx) => {
			await prepareAck(tx, staged, ctx);
			if (reject && staged.submissions.some((record) => record.status === "placed")) throw original;
		});
		await expect(f.submit("input", "actual input")).rejects.toBe(original);
		expect(await allEntries(f.root)).toEqual([]);
		expect(await f.harness.snapshot(Original, f.root.id, context)).toBeUndefined();
		expect((await f.harness.inspect(context)).tasks).toEqual([]);
		reject = false;
		await f.submit("input", "actual input");
		expect((await allEntries(f.root)).filter((entry) => entry.kind === UserEntry.kind)).toHaveLength(1);
	});
	it("places the retained corrected content after SQLite process loss and recreates exact native read debt", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-revised-input-"));
		directories.push(directory);
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		setup.faux.setResponses([fauxAssistantMessage("first"), fauxAssistantMessage("second")]);
		setup.registry.install(defineExtension({ name: "read-ack", tasks: [ReadAck] }));
		const options: HarnessOptions = {
			models: setup.models,
			registry: setup.registry,
			publishWake: async () => {},
			prepareCommit: prepareAck,
		};
		const first = await Harness.open(await openNodeSqliteStorage(path), options, context);
		sessions.push(first);
		const root = await first.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		await root.submit({ type: "input", content: "first" }, context);
		const queued = await root.submit({ type: "input", requestId: "queued", content: "old" }, context);
		await root.commit((tx) => tx.reviseQueuedInput(queued.id, { kind: "replace", content: "latest" }), context);
		await first.close(context);
		const reopened = await Harness.open(await openNodeSqliteStorage(path), options, context);
		sessions.push(reopened);
		const conversation = await reopened.conversation(root.id, context);
		if (!conversation) throw new Error("Lost canonical conversation");
		expect((await conversation.submit({ type: "input", requestId: "queued", content: "old" }, context)).id).toBe(
			queued.id,
		);
		await reopened.runPass(context);
		expect(
			(await allEntries(conversation))
				.filter((entry) => entry.kind === UserEntry.kind)
				.map((entry) => entry.model?.[0]?.content),
		).toEqual(["first", "latest"]);
		const readState = await reopened.snapshot(Original, root.id, context);
		expect(readState?.reads).toHaveLength(2);
		expect(readState?.acknowledged).toEqual(readState?.reads.map((read) => read.entry));
	});
});
