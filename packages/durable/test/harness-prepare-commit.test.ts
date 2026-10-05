import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import {
	AssistantEntry,
	type CommitPublication,
	defineDoc,
	defineExtension,
	defineTask,
	type EntryId,
	Harness,
	type HarnessCommit,
	type HarnessOptions,
	ReadAfterWrite,
	StorageRejected,
	type TaskId,
	WakeDoc,
} from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, chatSetup } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";
import { completed, deferred, settled } from "./task-support.ts";

const Publication = defineDoc<{ entries: EntryId[]; delivered: EntryId[] }>({
	kind: "test.publication",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ entries: [], delivered: [] }),
});
const Deliver = defineTask<{ entryId: EntryId }, { phase: "deliver" }, null>({
	name: "test.deliver",
	version: 1,
	initial: () => ({ phase: "deliver" }),
	phases: {
		deliver: async (task, rt, ctx) => {
			await rt.commit(async (tx) => {
				(await tx.doc(Publication, rt.conversationId)).delivered.push(task.input.entryId);
				return completed(null);
			}, ctx);
		},
	},
	abort: async () => {},
});
const preparePublication: NonNullable<HarnessOptions["prepareCommit"]> = async (tx, staged) => {
	for (const entry of staged.entries) {
		if (entry.kind !== AssistantEntry.kind) continue;
		(await tx.doc(Publication, entry.conversationId)).entries.push(entry.id);
		await tx.createTask(
			Deliver,
			{ entryId: entry.id },
			{ conversationId: entry.conversationId, ownership: { kind: "conversation" }, background: true },
		);
	}
};
function options(prepareCommit: NonNullable<HarnessOptions["prepareCommit"]>) {
	const setup = chatSetup();
	setup.registry.install(defineExtension({ name: "publication", tasks: [Deliver] }));
	return {
		setup,
		options: {
			models: setup.models,
			registry: setup.registry,
			prepareCommit,
			publishWake: async () => {},
		} satisfies HarnessOptions,
	};
}

describe("transactional harness preparation", () => {
	it("retains actual generated assistant delivery debt and native wake atomically through SQLite reopen", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-prepare-commit-"));
		const path = join(directory, "session.sqlite");
		const prepared: HarnessCommit[] = [];
		const config = options(async (tx, staged, ctx) => {
			prepared.push(staged);
			await preparePublication(tx, staged, ctx);
		});
		const extension = config.setup.registry.snapshot().extension("publication")!;
		config.setup.registry.uninstall(extension);
		config.setup.faux.setResponses([fauxAssistantMessage("ship this answer")]);
		let harness = await Harness.open(await openNodeSqliteStorage(path), config.options, context);
		try {
			const publications: CommitPublication[] = [];
			harness.subscribeCommits((publication) => publications.push(publication));
			const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
			const submission = await root.submit({ type: "input", content: "hello" }, context);
			await harness.runPass(context);
			const receipt = await submission.status(context);
			if (receipt.status !== "done" || receipt.type !== "input") throw new Error("Generation did not finish");
			const answer = (await allEntries(root)).find((entry) => entry.id === receipt.answer)!;
			expect(answer.kind).toBe(AssistantEntry.kind);
			expect(answer.byTaskId).toBeDefined();
			const batch = publications.find((publication) =>
				publication.changes.some((change) => change.type === "entry" && change.value.id === answer.id),
			)!;
			expect(batch.changes).toEqual(
				expect.arrayContaining([
					expect.objectContaining({
						type: "task",
						value: expect.objectContaining({ kind: "test.deliver", input: { entryId: answer.id } }),
					}),
					expect.objectContaining({
						type: "document",
						record: expect.objectContaining({ kind: "test.publication" }),
						value: { entries: [answer.id], delivered: [] },
					}),
				]),
			);
			expect(prepared.filter((batch) => batch.entries.some((entry) => entry.id === answer.id))).toHaveLength(1);
			expect(await harness.snapshot(Publication, root.id, context)).toEqual({ entries: [answer.id], delivered: [] });
			const pending = (await harness.inspect(context)).tasks;
			expect(pending).toHaveLength(1);
			expect(pending[0]!.record).toMatchObject({
				kind: "test.deliver",
				background: true,
				state: { status: "pending" },
			});
			expect(pending[0]!.state).toMatchObject({ kind: "blocked", reason: "missing_task" });
			const taskId = pending[0]!.record.id;
			await harness.close(context);
			config.setup.registry.install(extension);
			harness = await Harness.open(await openNodeSqliteStorage(path), config.options, context);
			expect(await harness.snapshot(Publication, root.id, context)).toEqual({ entries: [answer.id], delivered: [] });
			expect((await harness.inspect(context)).tasks.map((task) => task.record.id)).toEqual([taskId]);
			expect(await harness.flushWake(context)).toMatchObject({ wakeAt: 0 });
			await harness.runPass(context);
			expect((await harness.getTask(taskId, context))?.state.status).toBe("terminal");
			expect(await harness.snapshot(Publication, root.id, context)).toEqual({
				entries: [answer.id],
				delivered: [answer.id],
			});
		} finally {
			await harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});
	it("rolls back original entries, derived tasks, documents and wake with the original hook failure", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-prepare-rollback-"));
		const failure = new Error("publication target is invalid");
		let reject = true;
		let derivedId: TaskId | undefined;
		const config = options(async (tx, staged, ctx) => {
			await preparePublication(tx, staged, ctx);
			if (staged.entries.length === 0) return;
			derivedId = await tx.createTask(
				Deliver,
				{ entryId: staged.entries[0]!.id },
				{
					conversationId: staged.entries[0]!.conversationId,
					ownership: { kind: "conversation" },
					background: true,
				},
			);
			if (reject) throw failure;
		});
		const harness = await Harness.open(
			await openNodeSqliteStorage(join(directory, "session.sqlite")),
			config.options,
			context,
		);
		try {
			const root = await harness.root(context);
			await expect(
				root.commit((tx) => tx.appendEntry(root.id, { kind: AssistantEntry.kind }), context),
			).rejects.toBe(failure);
			expect(await allEntries(root)).toEqual([]);
			expect(await harness.getTask(derivedId!, context)).toBeUndefined();
			expect(await harness.snapshot(Publication, root.id, context)).toBeUndefined();
			expect(await harness.snapshot(WakeDoc, context)).toBeUndefined();
			reject = false;
			await root.commit((tx) => tx.appendEntry(root.id, { kind: AssistantEntry.kind }), context);
			expect(await allEntries(root)).toHaveLength(1);
		} finally {
			await harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});
	it("provides a detached deeply frozen snapshot once and schedules hook-created tasks without reentering", async () => {
		const storage = new ControlledStorage();
		let observed: HarnessCommit | undefined;
		const callContext = { ...context, marker: "caller context" };
		const config = options(async (tx, staged, ctx) => {
			if (!staged.entries.some((entry) => entry.kind === "source")) return;
			expect(ctx).toBe(callContext);
			observed = staged;
			expect(Object.isFrozen(staged)).toBe(true);
			expect(Object.isFrozen(staged.entries)).toBe(true);
			expect(Object.isFrozen(staged.entries[0]!.data)).toBe(true);
			expect(Reflect.set(staged.entries[0]!.data as object, "text", "changed")).toBe(false);
			expect(Object.isFrozen(staged.tasks[0]!.input)).toBe(true);
			expect(Object.isFrozen(staged.tasks[0]!.state)).toBe(true);
			const candidate = await staged.task(staged.tasks[0]!.id);
			expect(candidate).toEqual(staged.tasks[0]);
			expect(Object.isFrozen(candidate?.state)).toBe(true);
			expect(await staged.task(-1 as TaskId)).toBeUndefined();
			await tx.appendEntry(staged.entries[0]!.conversationId, { kind: "derived" });
			await tx.createTask(
				Deliver,
				{ entryId: staged.entries[0]!.id },
				{
					conversationId: staged.entries[0]!.conversationId,
					ownership: { kind: "conversation" },
					background: true,
				},
			);
		});
		const harness = await Harness.open(storage, config.options, context);
		try {
			const root = await harness.root(context);
			await root.commit(async (tx) => {
				const entry = await tx.appendEntry(root.id, { kind: "source", data: { text: "original" } });
				await tx.createTask(
					Deliver,
					{ entryId: entry.id },
					{ ownership: { kind: "conversation" }, background: true },
				);
			}, callContext);
			expect(observed!.entries.map((entry) => entry.kind)).toEqual(["source"]);
			expect(observed!.tasks).toHaveLength(1);
			expect(await allEntries(root)).toMatchObject([
				{ kind: "source", data: { text: "original" } },
				{ kind: "derived" },
			]);
			const batch = storage.commits.at(-1)!;
			expect(batch.filter((write) => write.type === "entry")).toHaveLength(2);
			expect(batch.filter((write) => write.type === "task")).toHaveLength(2);
			expect(batch.some((write) => write.type === "document.create" && write.record.kind === "pi.wake")).toBe(true);
		} finally {
			await harness.close(context);
		}
	});
	it("propagates StorageRejected without retaining an assistant entry or publication debt", async () => {
		const storage = new ControlledStorage();
		const config = options(preparePublication);
		const harness = await Harness.open(storage, config.options, context);
		try {
			const root = await harness.root(context);
			const failure = new StorageRejected("atomic batch rejected");
			storage.failNextCommit(failure);
			await expect(
				root.commit((tx) => tx.appendEntry(root.id, { kind: AssistantEntry.kind }), context),
			).rejects.toBe(failure);
			expect(await allEntries(root)).toEqual([]);
			expect(await harness.snapshot(Publication, root.id, context)).toBeUndefined();
			expect(await harness.snapshot(WakeDoc, context)).toBeUndefined();
			expect((await harness.inspect(context)).tasks).toEqual([]);
		} finally {
			await harness.close(context);
		}
	});
	it("retains ordinary table-read restrictions in preparation", async () => {
		const config = options(async (tx, staged) => {
			if (staged.entries.length > 0) await tx.entry(staged.entries[0]!.id);
		});
		const harness = await Harness.open(new ControlledStorage(), config.options, context);
		try {
			const root = await harness.root(context);
			await expect(root.commit((tx) => tx.appendEntry(root.id, { kind: "source" }), context)).rejects.toBeInstanceOf(
				ReadAfterWrite,
			);
			expect(await allEntries(root)).toEqual([]);
		} finally {
			await harness.close(context);
		}
	});
	it("joins admitted asynchronous preparation before owner close", async () => {
		const entered = deferred();
		const release = deferred();
		const config = options(async (tx, staged, ctx) => {
			if (staged.entries.length === 0) return;
			entered.resolve();
			await release.promise;
			await preparePublication(tx, staged, ctx);
		});
		const storage = new ControlledStorage();
		const harness = await Harness.open(storage, config.options, context);
		const root = await harness.root(context);
		const writing = root.commit((tx) => tx.appendEntry(root.id, { kind: AssistantEntry.kind }), context);
		try {
			await entered.promise;
			const closing = harness.close(context);
			expect(await settled(closing)).toBe(false);
			release.resolve();
			await writing;
			await closing;
			expect(storage.commits.at(-1)!.some((write) => write.type === "task")).toBe(true);
		} finally {
			release.resolve();
			await writing.catch(() => {});
			await harness.close(context);
		}
	});
	it("rejects and joins unawaited hook operations before native scheduler preparation", async () => {
		const storage = new ControlledStorage();
		const config = options((tx, staged) => {
			if (staged.entries.length > 0) void tx.doc(Publication, staged.entries[0]!.conversationId);
		});
		const harness = await Harness.open(storage, config.options, context);
		const root = await harness.root(context);
		const gate = storage.holdFindDocument();
		const writing = root.commit((tx) => tx.appendEntry(root.id, { kind: "source" }), context);
		const failure = expect(writing).rejects.toThrow("pending Tx operations");
		try {
			await gate.entered;
			expect(await settled(writing)).toBe(false);
			gate.release();
			await failure;
			expect(await allEntries(root)).toEqual([]);
			expect(await harness.snapshot(Publication, root.id, context)).toBeUndefined();
		} finally {
			gate.release();
			await failure;
			await harness.close(context);
		}
	});
});
