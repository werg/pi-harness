import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { JsonValue } from "@earendil-works/chord";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { describe, expect, it } from "vitest";
import { idFromNumber } from "../src/ids.ts";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import type { ConversationId, DocumentCreate, DocumentId, EntryId, Storage, TaskId, TaskRecord } from "../src/types.ts";
import { ROOT_CONVERSATION_ID } from "../src/types.ts";

const context = BACKGROUND_CONTEXT;
// More than Workerd's 2 MB row limit; UTF-8 code points also cross chunk boundaries.
const large = "🙂漢\ud800".repeat(350_000);
const entryId = idFromNumber<EntryId>(2);
const taskId = idFromNumber<TaskId<JsonValue>>(3);
const documentId = idFromNumber<DocumentId>(4);
const copiedId = idFromNumber<DocumentId>(5);
const latestId = idFromNumber<DocumentId>(6);

async function withStorage(use: (storage: Storage, path: string) => Promise<void>) {
	const directory = await mkdtemp(join(tmpdir(), "pi-large-payload-"));
	const path = join(directory, "storage.sqlite");
	const storage = await openNodeSqliteStorage(path);
	try {
		await use(storage, path);
	} finally {
		await storage.close(context);
		await rm(directory, { recursive: true, force: true });
	}
}

function task(input: string): TaskRecord<JsonValue, JsonValue, JsonValue> {
	return {
		id: taskId,
		conversationId: ROOT_CONVERSATION_ID,
		kind: "large",
		version: 1,
		input,
		state: { status: "pending", checkpoint: null },
		abortRequested: false,
		background: false,
	};
}

describe("SQLite large payload ownership", () => {
	it("retains large records and historical revisions exactly across reopen, paging and copy", async () => {
		await withStorage(async (storage, path) => {
			const record: DocumentCreate = {
				id: documentId,
				kind: "history",
				scope: { kind: "conversation", conversationId: ROOT_CONVERSATION_ID },
				history: "rewindable",
				fork: "asOf",
			};
			const first = await storage.commit(
				[
					{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
					{
						type: "entry",
						value: { id: entryId, conversationId: ROOT_CONVERSATION_ID, kind: "image", data: { large } },
					},
					{ type: "task", value: task(large) },
					{ type: "document.create", record, content: { kind: "base", version: 1, value: { large } } },
				],
				context,
			);
			const second = await storage.commit(
				[
					{
						type: "document.change",
						id: documentId,
						content: { kind: "delta", version: 1, ops: [["s", ["large"], `${large}new`]] },
					},
				],
				context,
			);
			await storage.commit(
				[
					{
						type: "document.copy",
						record: {
							...record,
							id: copiedId,
							key: undefined,
							scope: { kind: "conversation", conversationId: idFromNumber<ConversationId>(99) },
						},
						source: { id: documentId, at: first },
					},
				],
				context,
			);
			await storage.close(context);
			const reopened = await openNodeSqliteStorage(path);
			try {
				expect((await reopened.entry(entryId, context))?.entry.data).toEqual({ large });
				expect(
					(await reopened.scanEntries({ conversationId: ROOT_CONVERSATION_ID }, 1, undefined, context)).items.map(
						(row) => row.data,
					),
				).toEqual([{ large }]);
				expect((await reopened.scanTasks({}, 1, undefined, context)).items[0]?.input).toBe(large);
				expect((await reopened.document(documentId, first, context))?.value).toEqual({ large });
				expect((await reopened.document(documentId, second, context))?.value).toEqual({ large: `${large}new` });
				expect((await reopened.document(copiedId, "current", context))?.value).toEqual({ large });
			} finally {
				await reopened.close(context);
			}
			const db = new DatabaseSync(path);
			try {
				expect(db.prepare("SELECT max(length(data)) AS bytes FROM payload_chunks").get()).toEqual({
					bytes: 512 * 1024,
				});
			} finally {
				db.close();
			}
		});
	});

	it("rolls a partial replacement back and reclaims superseded current-only chunks", async () => {
		await withStorage(async (storage, path) => {
			await storage.commit(
				[
					{ type: "task", value: task(large) },
					{
						type: "document.create",
						record: { id: latestId, kind: "latest", scope: { kind: "session" } },
						content: { kind: "base", version: 1, value: { large } },
					},
				],
				context,
			);
			const db = new DatabaseSync(path);
			try {
				// Authoritative storage fault after the delete and first replacement chunk.
				db.exec(
					`CREATE TRIGGER reject_chunk BEFORE INSERT ON payload_chunks WHEN NEW.id = 3 AND NEW.ordinal = 1 BEGIN SELECT RAISE(ABORT, 'chunk admission rejected'); END`,
				);
				await expect(
					storage.commit([{ type: "task", value: task(`${large}replacement`) }], context),
				).rejects.toThrow("chunk admission rejected");
				expect((await storage.task(taskId, context))?.input).toBe(large);
				db.exec("DROP TRIGGER reject_chunk");
				await storage.commit(
					[
						{ type: "task", value: task("small") },
						{
							type: "document.change",
							id: latestId,
							content: { kind: "base", version: 1, value: { small: true } },
						},
					],
					context,
				);
				expect(db.prepare("SELECT count(*) AS chunks FROM payload_chunks WHERE id = 3").get()).toEqual({
					chunks: 1,
				});
				expect(
					db.prepare("SELECT count(*) AS chunks FROM payload_chunks WHERE id = 6 AND revision > 0").get(),
				).toEqual({ chunks: 1 });
				await storage.commit([{ type: "document.retire", id: latestId }], context);
				expect(
					db.prepare("SELECT count(*) AS chunks FROM payload_chunks WHERE id = 6 AND revision > 0").get(),
				).toEqual({ chunks: 0 });
				expect((await storage.task(taskId, context))?.input).toBe("small");
			} finally {
				db.close();
			}
		});
	});
});
