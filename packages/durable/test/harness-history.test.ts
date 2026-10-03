import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, type Message } from "@earendil-works/pi-ai";
import {
	AgentDoc,
	bindReceipt,
	CompactionEntry,
	type ConversationHistory,
	type ConversationId,
	defineDoc,
	defineTask,
	type EntryId,
	InboxDoc,
	LiveDoc,
	ReceiptDoc,
	ResetEntry,
	StorageRejected,
	SystemEntry,
	UsageDoc,
} from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, chatSetup, openChat, unanswered } from "./chat-support.ts";
import { assistant, openHarness, system, toolResult, user } from "./harness-support.ts";
import { ControlledStorage, context } from "./session-support.ts";

const Work = defineTask<null, { phase: "parked" }, null>({
	name: "history.owner",
	version: 1,
	initial: () => ({ phase: "parked" }),
	phases: { parked: async () => {} },
	abort: async () => {},
});

describe("knowledge-only conversation transfer", () => {
	it("binds exact remapped knowledge anchors atomically without reading staged native tables", async () => {
		const Anchors = defineDoc<{ entries: Record<string, number> }>({
			kind: "product.history-anchors",
			version: 1,
			scope: "conversation",
			history: "latest",
			fork: "initial",
			initial: () => ({ entries: {} }),
		});
		const source = await openHarness(new ControlledStorage());
		const receiver = await openHarness(new ControlledStorage());
		try {
			const root = await source.harness.root(context);
			const first = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "knowledge", model: [user("source knowledge")] }),
				context,
			);
			const last = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "summary", head: first.id, model: [user("summary")] }),
				context,
			);
			const receivingRoot = await receiver.harness.root(context);
			await receivingRoot.commit(async (tx) => {
				for (let index = 0; index < 3; index++) await tx.appendEntry(receivingRoot.id, { kind: "unrelated" });
			}, context);
			const history = await root.exportHistory(last.id, context);
			const imported = await receiver.harness.importHistory(
				history,
				{
					ownership: { kind: "ownerless" },
					init: async (tx, id, entryIds) => {
						expect(Object.isFrozen(entryIds)).toBe(true);
						expect(Reflect.set(entryIds, first.id, 999)).toBe(false);
						(await tx.doc(Anchors, id)).entries = { ...entryIds };
					},
				},
				context,
			);
			const entries = await allEntries(imported);
			expect(await receiver.harness.snapshot(Anchors, imported.id, context)).toEqual({
				entries: { [first.id]: entries[0]!.id, [last.id]: entries[1]!.id },
			});
			expect(entries[0]!.id).not.toBe(first.id);
			expect(entries[1]!.head).toBe(entries[0]!.id);
			expect((await imported.context(context)).messages).toEqual((await root.context(context)).messages);
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("starts only fresh receiving work after the child seed, with inherited knowledge in its actual provider request", async () => {
		const source = await openHarness(new ControlledStorage());
		const setup = chatSetup();
		let requested: readonly Message[] | undefined;
		setup.faux.setResponses([
			(request) => {
				requested = request.messages;
				return fauxAssistantMessage("fresh child answer");
			},
		]);
		const receiver = await openChat(new ControlledStorage(), setup);
		try {
			const root = await source.harness.root(context);
			const frontier = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "knowledge", model: [user("inherited completed knowledge")] }),
				context,
			);
			const history = await root.exportHistory(frontier.id, context);
			const imported = await receiver.harness.importHistory(
				history,
				{
					ownership: { kind: "ownerless" },
					agent: { model: { provider: "faux", modelId: "faux-1" } },
				},
				context,
			);
			expect((await receiver.harness.inspect(context)).tasks).toEqual([]);
			expect(setup.faux.state.callCount).toBe(0);
			const seed = await imported.submit({ type: "input", content: "new child task" }, context);
			expect((await seed.wait(context)).status).toBe("done");
			expect(requested?.filter((message) => message.role === "user")).toEqual([
				user("inherited completed knowledge"),
				expect.objectContaining({ role: "user", content: "new child task" }),
			]);
			expect(setup.faux.state.callCount).toBe(1);
			expect((await allEntries(imported)).at(-1)).toMatchObject({
				kind: "pi.assistant",
				byTaskId: expect.any(Number),
				conversationId: imported.id,
			});
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("imports a frontier cut through an unfinished tool round as inert knowledge with native missing-result projection", async () => {
		const source = await openHarness(new ControlledStorage());
		const receiver = await openHarness(new ControlledStorage());
		try {
			const root = await source.harness.root(context);
			await root.commit((tx) => tx.appendEntry(root.id, { kind: "pi.user", model: [user("call tools")] }), context);
			const calls = await root.commit(
				(tx) =>
					tx.appendEntry(root.id, { kind: "pi.assistant", model: [assistant("calling", { calls: ["a", "b"] })] }),
				context,
			);
			await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "pi.tool-result", model: [toolResult("a")] }),
				context,
			);
			const selected = await root.fork(calls.id, { ownership: { kind: "ownerless" } }, context);
			const history = await root.exportHistory(calls.id, context);
			const imported = await receiver.harness.importHistory(history, { ownership: { kind: "ownerless" } }, context);
			const view = await imported.context(context);
			expect(view.messages).toEqual((await selected.context(context)).messages);
			expect(view.messages.slice(2)).toEqual([
				expect.objectContaining({
					role: "toolResult",
					toolCallId: "a",
					isError: true,
					details: { reason: "missing_result" },
				}),
				expect.objectContaining({
					role: "toolResult",
					toolCallId: "b",
					isError: true,
					details: { reason: "missing_result" },
				}),
			]);
			expect((await receiver.harness.inspect(context)).tasks).toEqual([]);
			expect((await receiver.harness.inspect(context)).submissions).toEqual([]);
			expect(await receiver.harness.snapshot(LiveDoc, imported.id, context)).toEqual({});
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("preserves exact reset, compaction, edit, positional-system and opaque model history through the selected frontier", async () => {
		const source = await openHarness(new ControlledStorage());
		const receiver = await openHarness(new ControlledStorage());
		const root = await source.harness.root(context);
		try {
			await root.configure({ instructions: "at frontier" }, context);
			await root.commit(async (tx) => {
				await tx.appendEntry(root.id, { kind: "old", model: [user("before reset")] });
				await tx.appendEntry(ResetEntry, root.id, { head: "self", model: [user("reset handoff")] });
				await tx.appendEntry(SystemEntry, root.id, { model: [system({ preamble: "original" })] });
			}, context);
			const first = await root.commit(
				(tx) =>
					tx.appendEntry(root.id, {
						kind: "product.delivery",
						model: [user("original")],
						data: { grant: "never transfer", invocation: "private" },
					}),
				context,
			);
			const call = assistant("tools", { calls: ["a", "b"] });
			call.content.unshift({
				type: "thinking",
				thinking: "opaque reasoning",
				thinkingSignature: "retain signature",
			});
			await root.commit(async (tx) => {
				await tx.appendEntry(root.id, { kind: "pi.assistant", model: [call] });
				await tx.appendEntry(root.id, { kind: "pi.tool-result", model: [toolResult("b")] });
				await tx.appendEntry(SystemEntry, root.id, { model: [system({ cwd: "/source" })] });
				await tx.appendEntry(root.id, { kind: "pi.tool-result", model: [toolResult("a")] });
				await tx.appendEntry(root.id, {
					kind: "edit",
					edits: [{ target: first.id, action: "replace", messages: [user("edited")] }],
					data: { approval: "never transfer" },
				});
				await tx.appendEntry(CompactionEntry, root.id, {
					head: first.id,
					model: [user("summary")],
					data: { reason: "manual" },
				});
				await tx.appendEntry(root.id, {
					kind: "pi.assistant",
					model: [assistant("aborted", { stopReason: "aborted" })],
				});
			}, context);
			const frontier = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "product.approval", data: { grant: "not knowledge" } }),
				context,
			);
			const expected = await root.fork(frontier.id, { ownership: { kind: "ownerless" } }, context);
			const messages = (await expected.context(context)).messages;
			const history = await root.exportHistory(frontier.id, context);
			await root.configure({ instructions: "too late" }, context);
			await root.commit((tx) => tx.appendEntry(root.id, { kind: "new", model: [user("not inherited")] }), context);
			expect(history.agent.instructions).toBe("at frontier");
			expect(history.source).toEqual({ conversationId: root.id, at: frontier.id });
			expect(history.entries.some((entry) => entry.id === frontier.id)).toBe(false);
			expect(Object.isFrozen(history)).toBe(true);
			expect(Object.isFrozen(history.entries)).toBe(true);
			expect(Object.isFrozen(history.entries[0]!.model)).toBe(true);
			expect(Reflect.set(history.agent, "instructions", "mutated")).toBe(false);
			const imported = await receiver.harness.importHistory(history, { ownership: { kind: "ownerless" } }, context);
			expect((await imported.context(context)).messages).toEqual(messages);
			const entries = await allEntries(imported);
			expect(entries).toHaveLength(history.entries.length);
			expect(entries.every((entry) => entry.kind === "pi.history" || entry.kind === SystemEntry.kind)).toBe(true);
			expect(
				entries.every(
					(entry) =>
						entry.data === undefined && entry.byTaskId === undefined && entry.conversationId === imported.id,
				),
			).toBe(true);
			const ids = new Map(history.entries.map((entry, index) => [entry.id, entries[index]!.id]));
			for (const [index, entry] of history.entries.entries()) {
				expect(entries[index]!.head).toBe(entry.head === undefined ? undefined : ids.get(entry.head));
				expect(entries[index]!.edits).toEqual(
					entry.edits?.map((edit) => ({ ...edit, target: ids.get(edit.target) })),
				);
			}
			expect(await receiver.harness.snapshot(AgentDoc, imported.id, context)).toMatchObject({
				instructions: "at frontier",
			});
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("exports nested inherited prefixes and pins configuration from the entry-owning ancestor", async () => {
		const source = await openHarness(new ControlledStorage());
		const receiver = await openHarness(new ControlledStorage());
		try {
			const root = await source.harness.root(context, { agent: { instructions: "ancestor settings" } });
			const inherited = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "knowledge", model: [user("ancestor")] }),
				context,
			);
			const branch = await root.fork(
				inherited.id,
				{ ownership: { kind: "ownerless" }, agent: { instructions: "branch settings" } },
				context,
			);
			const own = await branch.commit(
				(tx) => tx.appendEntry(branch.id, { kind: "knowledge", model: [user("branch")] }),
				context,
			);
			const nested = await branch.fork(own.id, { ownership: { kind: "ownerless" } }, context);
			const inheritedHistory = await nested.exportHistory(inherited.id, context);
			expect(inheritedHistory.agent.instructions).toBe("ancestor settings");
			expect(inheritedHistory.entries.map((entry) => entry.id)).toEqual([inherited.id]);
			const ownHistory = await nested.exportHistory(own.id, context);
			expect(ownHistory.agent.instructions).toBe("branch settings");
			const imported = await receiver.harness.importHistory(
				ownHistory,
				{ ownership: { kind: "ownerless" } },
				context,
			);
			expect((await imported.context(context)).messages).toEqual((await nested.context(context)).messages);
			await expect(branch.exportHistory(999_999 as EntryId, context)).rejects.toThrow("is not visible");
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("deliberately exports an empty prefix with pinned current settings and permits explicit receiving overrides", async () => {
		const source = await openHarness(new ControlledStorage());
		const receiver = await openHarness(new ControlledStorage());
		try {
			const root = await source.harness.root(context, {
				agent: { instructions: "pinned", cwd: "/source", thinkingLevel: "high" },
			});
			await root.commit((tx) => tx.appendEntry(root.id, { kind: "knowledge", model: [user("excluded")] }), context);
			const history = await root.exportHistory(null, context);
			await root.configure({ instructions: "changed" }, context);
			expect(history.entries).toEqual([]);
			expect(history.agent.instructions).toBe("pinned");
			const imported = await receiver.harness.importHistory(
				history,
				{ ownership: { kind: "ownerless" }, agent: { instructions: "receiver", cwd: null } },
				context,
			);
			expect(await allEntries(imported)).toEqual([]);
			expect(await receiver.harness.snapshot(AgentDoc, imported.id, context)).toEqual({
				instructions: "receiver",
				thinkingLevel: "high",
			});
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("creates fresh receiving ownership without source tasks, inbox, usage, receipts, grants or ExecutionOwner documents", async () => {
		const ExecutionOwner = defineDoc<{ incarnation: string }>({
			kind: "product.execution-owner",
			version: 1,
			scope: "session",
			initial: () => ({ incarnation: "source" }),
		});
		const Grants = defineDoc<{ token: string }>({
			kind: "product.grants",
			version: 1,
			scope: "conversation",
			history: "latest",
			fork: "current",
			initial: () => ({ token: "source grant" }),
		});
		const setup = chatSetup();
		const busy = unanswered();
		setup.faux.setResponses([busy.step]);
		const source = await openChat(new ControlledStorage(), setup);
		const receiver = await openHarness(new ControlledStorage());
		try {
			const knowledge = await source.root.commit(
				(tx) => tx.appendEntry(source.root.id, { kind: "knowledge", model: [user("completed knowledge")] }),
				context,
			);
			await source.root.commit(async (tx) => {
				await tx.doc(ExecutionOwner);
				await tx.doc(Grants, source.root.id);
				await bindReceipt(tx, "source-receipt", "source-binding");
				const taskId = await tx.createTask(Work, null, { ownership: { kind: "conversation" } });
				await tx.appendEntry(source.root.id, { kind: "product.eval", data: { taskId, grant: "source grant" } });
			}, context);
			await source.root.submit({ type: "input", content: "parent work" }, context);
			await busy.reached;
			await source.root.submit({ type: "input", content: "queued parent work" }, context);
			const sourceState = await source.harness.inspect(context);
			expect(sourceState.tasks.length).toBeGreaterThan(0);
			expect(sourceState.submissions.some((submission) => submission.status === "queued")).toBe(true);
			const history = await source.root.exportHistory(knowledge.id, context);
			const receivingRoot = await receiver.harness.root(context);
			const ownerId = await receivingRoot.commit(
				(tx) => tx.createTask(Work, null, { ownership: { kind: "conversation" } }),
				context,
			);
			const imported = await receiver.harness.importHistory(
				history,
				{ ownership: { kind: "task", taskId: ownerId } },
				context,
			);
			const record = await imported.commit((tx) => tx.conversation(imported.id), context);
			expect(record).toEqual({ id: imported.id, owner: { conversationId: receivingRoot.id, taskId: ownerId } });
			expect(await receiver.harness.snapshot(LiveDoc, imported.id, context)).toEqual({});
			expect(await receiver.harness.snapshot(InboxDoc, imported.id, context)).toEqual({ items: [] });
			expect(await receiver.harness.snapshot(UsageDoc, imported.id, context)).toEqual({ models: {}, tools: {} });
			expect(await receiver.harness.snapshot(ReceiptDoc, "source-receipt", context)).toBeUndefined();
			expect(await receiver.harness.snapshot(ExecutionOwner, context)).toBeUndefined();
			expect(await receiver.harness.snapshot(Grants, imported.id, context)).toBeUndefined();
			expect((await receiver.harness.inspect(context)).tasks.map((task) => task.record.id)).toEqual([ownerId]);
			expect((await receiver.harness.inspect(context)).submissions).toEqual([]);
			expect((await imported.context(context)).messages).toEqual([user("completed knowledge")]);
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("keeps model-less head anchors while dropping their arbitrary application payload", async () => {
		const source = await openHarness(new ControlledStorage());
		const receiver = await openHarness(new ControlledStorage());
		try {
			const root = await source.harness.root(context);
			const anchor = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "product.approval", data: { pending: "private" } }),
				context,
			);
			await root.commit((tx) => tx.appendEntry(root.id, { kind: "message", model: [user("kept")] }), context);
			const head = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "summary", head: anchor.id, model: [user("summary")] }),
				context,
			);
			const history = await root.exportHistory(head.id, context);
			expect(history.entries[0]).toEqual({ id: anchor.id });
			const imported = await receiver.harness.importHistory(history, { ownership: { kind: "ownerless" } }, context);
			expect((await imported.context(context)).messages).toEqual((await root.context(context)).messages);
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("rolls back the entire imported candidate on original init failure or deterministic storage rejection", async () => {
		const source = await openHarness(new ControlledStorage());
		const storage = new ControlledStorage();
		const receiver = await openHarness(storage);
		try {
			const root = await source.harness.root(context);
			const entry = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "knowledge", model: [user("knowledge")] }),
				context,
			);
			const history = await root.exportHistory(entry.id, context);
			let failedId: ConversationId | undefined;
			const failure = new Error("receiving initialization failed");
			const before = storage.commits.length;
			await expect(
				receiver.harness.importHistory(
					history,
					{
						ownership: { kind: "ownerless" },
						init: (_tx, id) => {
							failedId = id;
							throw failure;
						},
					},
					context,
				),
			).rejects.toBe(failure);
			expect(storage.commits).toHaveLength(before);
			expect(await receiver.harness.conversation(failedId!, context)).toBeUndefined();
			const rejected = new StorageRejected("import rejected");
			storage.failNextCommit(rejected);
			await expect(
				receiver.harness.importHistory(
					history,
					{
						ownership: { kind: "ownerless" },
						init: (_tx, id) => {
							failedId = id;
						},
					},
					context,
				),
			).rejects.toBe(rejected);
			expect(await receiver.harness.conversation(failedId!, context)).toBeUndefined();
			const imported = await receiver.harness.importHistory(history, { ownership: { kind: "ownerless" } }, context);
			expect((await imported.context(context)).messages).toEqual([user("knowledge")]);
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("rejects dangling heads/edits, duplicate IDs, future entries and a nonempty null frontier before native admission", async () => {
		const source = await openHarness(new ControlledStorage());
		const storage = new ControlledStorage();
		const receiver = await openHarness(storage);
		try {
			const root = await source.harness.root(context);
			const entry = await root.commit(
				(tx) => tx.appendEntry(root.id, { kind: "knowledge", model: [user("knowledge")] }),
				context,
			);
			const history = await root.exportHistory(entry.id, context);
			const malformed: ConversationHistory[] = [
				{ ...history, entries: [{ ...history.entries[0]!, head: 999_999 as EntryId }] },
				{
					...history,
					entries: [{ ...history.entries[0]!, edits: [{ target: 999_999 as EntryId, action: "omit" }] }],
				},
				{ ...history, entries: [history.entries[0]!, history.entries[0]!] },
				{ ...history, entries: [{ ...history.entries[0]!, id: (entry.id + 1) as EntryId }] },
				{ ...history, source: { ...history.source, at: null } },
			];
			const before = storage.commits.length;
			for (const value of malformed)
				await expect(
					receiver.harness.importHistory(value, { ownership: { kind: "ownerless" } }, context),
				).rejects.toThrow();
			expect(storage.commits).toHaveLength(before);
			expect((await receiver.harness.inspect(context)).tasks).toEqual([]);
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
		}
	});

	it("transfers across paged scans and source destruction, retaining standalone receiving SQLite history after reopen", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-history-transfer-"));
		const source = await openHarness(new ControlledStorage());
		const path = join(directory, "receiving.sqlite");
		let receiver = await openHarness(await openNodeSqliteStorage(path));
		try {
			const root = await source.harness.root(context);
			let at: EntryId | undefined;
			await root.commit(async (tx) => {
				for (let index = 0; index < 270; index++)
					at = (await tx.appendEntry(root.id, { kind: "knowledge", model: [user(`message ${index}`)] })).id;
			}, context);
			const history = await root.exportHistory(at!, context);
			expect(history.entries).toHaveLength(270);
			await source.harness.close(context);
			const imported = await receiver.harness.importHistory(
				JSON.parse(JSON.stringify(history)) as ConversationHistory,
				{ ownership: { kind: "ownerless" } },
				context,
			);
			const id = imported.id;
			const messages = (await imported.context(context)).messages;
			await receiver.harness.close(context);
			receiver = await openHarness(await openNodeSqliteStorage(path));
			const reopened = (await receiver.harness.conversation(id, context))!;
			expect((await reopened.context(context)).messages).toEqual(messages);
			expect((await receiver.harness.inspect(context)).tasks).toEqual([]);
			expect((await receiver.harness.inspect(context)).submissions).toEqual([]);
			await reopened.commit(
				(tx) => tx.appendEntry(reopened.id, { kind: "new knowledge", model: [user("fresh receiving work")] }),
				context,
			);
			expect((await reopened.context(context)).messages).toHaveLength(271);
		} finally {
			await source.harness.close(context);
			await receiver.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});
});
