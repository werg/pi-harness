import { defineDoc, defineTask } from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { SessionImpl } from "../src/session/session.ts";
import { ControlledStorage, context, createConversation } from "./session-support.ts";

const Task = defineTask<null, { phase: "run" }, null>({
	name: "test.deferred.atomic.task",
	version: 1,
	initial: () => ({ phase: "run" }),
	phases: { run: async () => {} },
	abort: async () => {},
});

const TaskProgress = defineDoc<{ text: string }>({
	kind: "test.deferred.atomic.task-progress",
	version: 1,
	scope: "task",
	initial: () => ({ text: "" }),
});

const ConversationProgress = defineDoc<{ text: string }>({
	kind: "test.deferred.atomic.conversation-progress",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ text: "" }),
});

const OtherProgress = defineDoc<{ text: string }>({
	kind: "test.deferred.atomic.other-progress",
	version: 1,
	scope: "conversation",
	history: "latest",
	fork: "initial",
	initial: () => ({ text: "" }),
});

describe("atomic deferred progress", () => {
	it("stores task and conversation progress together when a later transaction reads one", async () => {
		const storage = new ControlledStorage();
		const session = new SessionImpl(storage);
		const conversationId = await createConversation(session);
		const taskId = await session.commit(async (tx) => {
			const id = await tx.createTask(Task, null, { ownership: { kind: "conversation" }, conversationId });
			await tx.doc(TaskProgress, id);
			await tx.doc(ConversationProgress, conversationId);
			await tx.doc(OtherProgress, conversationId);
			return id;
		}, context);
		const taskRecord = await storage.findDocument(
			{ kind: TaskProgress.definition.kind, scope: { kind: "task", taskId } },
			"current",
			context,
		);
		const conversationRecord = await storage.findDocument(
			{ kind: ConversationProgress.definition.kind, scope: { kind: "conversation", conversationId } },
			"current",
			context,
		);
		const otherRecord = await storage.findDocument(
			{ kind: OtherProgress.definition.kind, scope: { kind: "conversation", conversationId } },
			"current",
			context,
		);
		expect(taskRecord).toBeDefined();
		expect(conversationRecord).toBeDefined();
		expect(otherRecord).toBeDefined();
		const admitted = storage.commits.length;
		await session.commit(
			async (tx) => {
				(await tx.doc(TaskProgress, taskId)).text = "partial";
				(await tx.doc(ConversationProgress, conversationId)).text = "partial";
			},
			context,
			{ persistence: { kind: "deferred", withinMs: Infinity } },
		);
		await session.commit(
			async (tx) => {
				(await tx.doc(OtherProgress, conversationId)).text = "unrelated";
			},
			context,
			{ persistence: { kind: "deferred", withinMs: Infinity } },
		);
		expect(storage.commits.length).toBe(admitted);
		expect((await session.snapshot(TaskProgress, taskId, context))?.text).toBe("partial");
		expect((await storage.document(taskRecord!.id, "current", context))?.value).toEqual({ text: "" });
		await session.commit(async (tx) => {
			await tx.doc(ConversationProgress, conversationId);
			await tx.appendEntry(conversationId, { kind: "test.atomic.read" });
		}, context);
		const writes = storage.commits.at(-1)!;
		const changed = writes.filter((write) => write.type === "document.change").map((write) => write.id);
		expect(changed).toContain(taskRecord!.id);
		expect(changed).toContain(conversationRecord!.id);
		expect(changed).not.toContain(otherRecord!.id);
		expect((await storage.document(taskRecord!.id, "current", context))?.value).toEqual({ text: "partial" });
		expect((await storage.document(conversationRecord!.id, "current", context))?.value).toEqual({ text: "partial" });
		expect((await storage.document(otherRecord!.id, "current", context))?.value).toEqual({ text: "" });
		const beforeOverlap = storage.commits.length;
		await session.commit(
			async (tx) => {
				(await tx.doc(TaskProgress, taskId)).text = "second";
				(await tx.doc(OtherProgress, conversationId)).text = "second";
			},
			context,
			{ persistence: { kind: "deferred", withinMs: Infinity } },
		);
		await session.commit(
			async (tx) => {
				(await tx.doc(OtherProgress, conversationId)).text = "third";
				(await tx.doc(ConversationProgress, conversationId)).text = "third";
			},
			context,
			{ persistence: { kind: "deferred", withinMs: Infinity } },
		);
		expect(storage.commits.length).toBe(beforeOverlap);
		await session.commit(async (tx) => {
			await tx.doc(TaskProgress, taskId);
			await tx.appendEntry(conversationId, { kind: "test.atomic.overlap" });
		}, context);
		const overlapping = storage.commits
			.at(-1)!
			.filter((write) => write.type === "document.change")
			.map((write) => write.id);
		expect(new Set(overlapping)).toEqual(new Set([taskRecord!.id, conversationRecord!.id, otherRecord!.id]));
		await session.commit(
			async (tx) => {
				(await tx.doc(TaskProgress, taskId)).text = "last";
				(await tx.doc(ConversationProgress, conversationId)).text = "last";
			},
			context,
			{ persistence: { kind: "deferred", withinMs: Infinity } },
		);
		await session.commitWith(async (tx) => {
			const task = await tx.task(taskId);
			if (!task) throw new Error("Missing task");
			const { memos: _memos, ...terminalTask } = task;
			tx.setTask({ ...terminalTask, state: { status: "terminal", outcome: { status: "completed", result: null } } });
		}, context);
		const terminal = storage.commits.at(-1)!;
		const changes = terminal.filter((write) => write.type === "document.change").map((write) => write.id);
		expect(changes).toContain(taskRecord!.id);
		expect(changes).toContain(conversationRecord!.id);
		expect(terminal.some((write) => write.type === "document.retire" && write.id === taskRecord!.id)).toBe(true);
		expect((await storage.document(conversationRecord!.id, "current", context))?.value).toEqual({ text: "last" });
		await session.close(context);
	});
});
