import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { registerStorageConformance } from "@earendil-works/pi-durable/testing";
import { afterEach, describe, expect, it } from "vitest";
import { idFromNumber } from "../src/ids.ts";
import type { SqliteExecutor } from "../src/storage/sqlite/database.ts";
import {
	applyCompositeSqliteMigrationsInTransaction,
	type CommittingSqliteDatabase,
	CompositeSqliteStorage,
	type CompositeSqliteStorageOptions,
	defaultDocumentRoute,
} from "../src/storage/sqlite/index.ts";
import { openNodeCompositeSqliteStorage, openNodeSqliteDatabase } from "../src/storage/sqlite/node.ts";
import {
	type DocumentId,
	type EntryId,
	ROOT_CONVERSATION_ID,
	type StorageWrite,
	type SubmissionId,
	type TaskId,
	type TaskRecord,
} from "../src/types.ts";

const adopt = (): void => {};
const context = BACKGROUND_CONTEXT;
const directories = new Set<string>();
const openStorages = new Set<CompositeSqliteStorage>();

afterEach(async () => {
	for (const storage of openStorages) await storage.close(context).catch(() => {});
	openStorages.clear();
	for (const directory of directories) await rm(directory, { recursive: true, force: true });
	directories.clear();
});

async function temporaryPath(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-durable-composite-"));
	directories.add(directory);
	return join(directory, "storage.sqlite");
}

async function openComposite(path: string, options: Omit<CompositeSqliteStorageOptions, "commit"> = {}) {
	const storage = await openNodeCompositeSqliteStorage(path, options);
	openStorages.add(storage);
	return storage;
}

registerStorageConformance({ describe, expect, it }, "CompositeSqliteStorage on Node SQLite", async (use) => {
	const path = await temporaryPath();
	const storage = await openNodeCompositeSqliteStorage(path);
	try {
		await use(storage);
	} finally {
		await storage.close(context).catch(() => {});
	}
});

const rows = (path: string, table: string): number => {
	const db = new DatabaseSync(path, { readOnly: true });
	try {
		return Number((db.prepare(`SELECT count(*) AS n FROM ${table}`).get() as { n: number }).n);
	} finally {
		db.close();
	}
};

const task = (id: number): TaskRecord<never, never, never> =>
	({
		id: idFromNumber<TaskId>(id),
		conversationId: ROOT_CONVERSATION_ID,
		kind: "test.task",
		version: 1,
		state: { status: "pending" },
		abortRequested: false,
		background: false,
		input: null,
	}) as unknown as TaskRecord<never, never, never>;

const entry = (id: number, kind = "entry"): StorageWrite => ({
	type: "entry",
	value: { id: idFromNumber<EntryId>(id), conversationId: ROOT_CONVERSATION_ID, kind },
});

const document = (
	id: number,
	kind: string,
	scope: "session" | "conversation",
	fork?: "current" | "initial",
): StorageWrite => ({
	type: "document.create",
	record:
		scope === "session"
			? { id: idFromNumber<DocumentId>(id), kind, scope: { kind: "session" } }
			: {
					id: idFromNumber<DocumentId>(id),
					kind,
					scope: { kind: "conversation", conversationId: ROOT_CONVERSATION_ID },
					history: "latest",
					fork: fork ?? "current",
				},
	content: { kind: "base", version: 1, value: { kind } },
});

describe("CompositeSqliteStorage", () => {
	it("keeps conversations, entries and inherited conversation documents in the versioned set, and tasks, submissions and the rest in the local set", async () => {
		const path = await temporaryPath();
		const storage = await openComposite(path);
		await storage.commit(
			[
				{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
				entry(2),
				{ type: "task", value: task(3) },
				{
					type: "submission",
					value: {
						id: idFromNumber<SubmissionId>(4),
						conversationId: ROOT_CONVERSATION_ID,
						type: "input",
						status: "queued",
					} as never,
				},
				document(5, "pi.knowledge", "conversation", "current"),
				document(6, "pi.initial", "conversation", "initial"),
				document(7, "pi.session", "session"),
			],
			adopt,
			context,
		);
		expect(rows(path, "conversations")).toBe(1);
		expect(rows(path, "entries")).toBe(1);
		expect(rows(path, "tasks")).toBe(0);
		expect(rows(path, "vsl_tasks")).toBe(1);
		expect(rows(path, "vsl_submissions")).toBe(1);
		expect(rows(path, "documents")).toBe(1);
		expect(rows(path, "vsl_documents")).toBe(2);
		expect((await storage.task(idFromNumber<TaskId>(3), context))?.kind).toBe("test.task");
		expect((await storage.document(idFromNumber<DocumentId>(5), "current", context))?.value).toEqual({
			kind: "pi.knowledge",
		});
		expect((await storage.document(idFromNumber<DocumentId>(7), "current", context))?.value).toEqual({
			kind: "pi.session",
		});
		expect(
			(await storage.findDocument({ kind: "pi.session", scope: { kind: "session" } }, "current", context))?.id,
		).toBe(7);
	});

	it("changes and retires a document in the set that holds it, also after a reopen", async () => {
		const path = await temporaryPath();
		const first = await openComposite(path);
		await first.commit(
			[
				{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
				document(2, "pi.knowledge", "conversation", "current"),
				document(3, "pi.session", "session"),
			],
			adopt,
			context,
		);
		await first.close(context);
		const second = await openComposite(path);
		await second.commit(
			[
				{
					type: "document.change",
					id: idFromNumber<DocumentId>(2),
					content: { kind: "base", version: 1, value: { changed: 1 } },
				},
				{
					type: "document.change",
					id: idFromNumber<DocumentId>(3),
					content: { kind: "base", version: 1, value: { changed: 2 } },
				},
			],
			adopt,
			context,
		);
		expect((await second.document(idFromNumber<DocumentId>(2), "current", context))?.value).toEqual({ changed: 1 });
		expect((await second.document(idFromNumber<DocumentId>(3), "current", context))?.value).toEqual({ changed: 2 });
		await second.commit([{ type: "document.retire", id: idFromNumber<DocumentId>(3) }], adopt, context);
		expect(await second.document(idFromNumber<DocumentId>(3), "current", context)).toBeUndefined();
	});

	it("scans documents across both sets in ID order and pages through them", async () => {
		const path = await temporaryPath();
		const storage = await openComposite(path, {
			documentRoute: (create) => (create.kind.startsWith("v.") ? "versioned" : "local"),
		});
		await storage.commit(
			[
				{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
				document(2, "v.a", "conversation", "current"),
				document(3, "l.b", "conversation", "current"),
				document(4, "v.c", "conversation", "current"),
				document(5, "l.d", "conversation", "current"),
				document(6, "v.e", "conversation", "current"),
			],
			adopt,
			context,
		);
		const scope = { kind: "conversation", conversationId: ROOT_CONVERSATION_ID } as const;
		const ids: number[] = [];
		let cursor: Awaited<ReturnType<typeof storage.scanDocuments>>["next"];
		do {
			const page = await storage.scanDocuments({ scope, at: "current" }, 2, cursor, context);
			expect(page.items.length).toBeLessThanOrEqual(2);
			ids.push(...page.items.map((record) => record.id));
			cursor = page.next;
		} while (cursor !== undefined);
		expect(ids).toEqual([2, 3, 4, 5, 6]);
	});

	it("mints IDs and sequences from the local counters, which survive a rewind of the versioned set", async () => {
		const path = await temporaryPath();
		const first = await openComposite(path);
		await first.commit([{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }], adopt, context);
		const ids: number[] = [];
		for (let index = 0; index < 3; index++) {
			const id = await first.mintId<EntryId>();
			ids.push(id);
			await first.commit([entry(id)], adopt, context);
		}
		expect(ids).toEqual([2, 3, 4]);
		await first.close(context);
		// A rewind of the versioned set (a reset to an earlier commit): its rows go, the local counters stay.
		const db = new DatabaseSync(path);
		db.exec("DELETE FROM entries WHERE id > 2; DELETE FROM record_ids WHERE id > 2");
		db.close();
		const second = await openComposite(path);
		const id = await second.mintId<EntryId>();
		expect(id).toBe(5);
		const seq = await second.commit([entry(id)], adopt, context);
		expect(seq).toBeGreaterThan(3);
		expect((await second.entry(idFromNumber<EntryId>(5), context))?.commitSeq).toBe(seq);
	});

	it("starts a fork's counters above what the versioned set stores, with no local set to read", async () => {
		const parentPath = await temporaryPath();
		const parent = await openComposite(parentPath);
		await parent.commit(
			[
				{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
				document(2, "pi.knowledge", "conversation", "current"),
			],
			adopt,
			context,
		);
		for (let index = 0; index < 4; index++) {
			const id = await parent.mintId<EntryId>();
			await parent.commit([entry(id)], adopt, context);
		}
		const highest = await parent.mintId<EntryId>();
		await parent.close(context);
		// A fork copies the file and empties the local set (a Dolt fork starts local tables empty).
		const forkPath = await temporaryPath();
		const source = new DatabaseSync(parentPath, { readOnly: true });
		source.exec(`VACUUM INTO '${forkPath.replaceAll("'", "''")}'`);
		source.close();
		const copy = new DatabaseSync(forkPath);
		copy.exec(
			"DELETE FROM vsl_tasks; DELETE FROM vsl_submissions; DELETE FROM vsl_documents; DELETE FROM vsl_document_revisions; DELETE FROM vsl_payload_chunks; DELETE FROM vsl_record_ids; DELETE FROM vsl_entries; DELETE FROM vsl_conversations; UPDATE vsl_durable_metadata SET next_id = '2', next_seq = 1",
		);
		copy.close();
		const fork = await openComposite(forkPath);
		const next = await fork.mintId<EntryId>();
		// Above every ID the versioned set stores: none is reused, whatever the parent minted after the fork point.
		expect(next).toBeGreaterThanOrEqual(7);
		expect(highest).toBeGreaterThanOrEqual(7);
		const seq = await fork.commit([entry(next)], adopt, context);
		const stored = await fork.entry(next, context);
		expect(stored?.commitSeq).toBe(seq);
		expect(await fork.scanTasks({}, 10, undefined, context).then((page) => page.items)).toEqual([]);
	});

	it("commits both sets in one transaction: a failure leaves nothing in either", async () => {
		const path = await temporaryPath();
		const storage = await openComposite(path);
		await storage.commit([{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }], adopt, context);
		await expect(
			storage.commit(
				[
					entry(2),
					{ type: "task", value: task(3) },
					{
						type: "document.change",
						id: idFromNumber<DocumentId>(99),
						content: { kind: "base", version: 1, value: {} },
					},
				],
				adopt,
				context,
			),
		).rejects.toThrow();
		expect(rows(path, "entries")).toBe(0);
		expect(rows(path, "vsl_tasks")).toBe(0);
	});
});

class RecordingDatabase implements CommittingSqliteDatabase {
	readonly requests: Array<{ readonly commit: string | null | undefined }> = [];
	readonly inner: CommittingSqliteDatabase;
	constructor(inner: CommittingSqliteDatabase) {
		this.inner = inner;
	}
	exec = (sql: string) => this.inner.exec(sql);
	run = (sql: string, ...params: never[]) => this.inner.run(sql, ...params);
	runCounted = (sql: string, ...params: never[]) => this.inner.runCounted(sql, ...params);
	get = <T extends object>(sql: string, ...params: never[]) => this.inner.get<T>(sql, ...params);
	all = <T extends object>(sql: string, ...params: never[]) => this.inner.all<T>(sql, ...params);
	close = () => this.inner.close();
	async transaction<T>(
		callback: (transaction: SqliteExecutor) => Promise<T>,
		afterCommit?: (value: T) => void,
		options?: { readonly commit?: (value: T) => string | null },
	): Promise<T> {
		let result!: T;
		const value = await this.inner.transaction(async (transaction) => {
			result = await callback(transaction);
			return result;
		}, afterCommit);
		this.requests.push({ commit: options?.commit === undefined ? undefined : options.commit(value) });
		return value;
	}
}

describe("CompositeSqliteStorage commit policy", () => {
	it("asks the database to end a transaction with the commit the policy names for its writes and sequence, and asks for none otherwise", async () => {
		const path = await temporaryPath();
		const database = new RecordingDatabase(
			(await openNodeSqliteDatabase(path)) as unknown as CommittingSqliteDatabase,
		);
		await database.transaction((transaction) => applyCompositeSqliteMigrationsInTransaction(transaction));
		const storage = await CompositeSqliteStorage.open(database, {
			commit: (writes, seq, state) =>
				writes.some((write) => write.type === "entry" && write.value.kind === "turn.closed")
					? `agent: turn closed at ${seq} entries<=${state.entryHigh}`
					: null,
		});
		openStorages.add(storage);
		await storage.commit([{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }, entry(2)], adopt, context);
		const seq = await storage.commit([entry(3, "turn.closed")], adopt, context);
		await storage.commit([{ type: "task", value: task(4) }], adopt, context);
		expect(database.requests.at(-3)).toEqual({ commit: null });
		expect(database.requests.at(-2)).toEqual({ commit: `agent: turn closed at ${seq} entries<=3` });
		expect(database.requests.at(-1)).toEqual({ commit: null });
	});
});

describe("defaultDocumentRoute", () => {
	it("versions what a fork inherits and keeps the rest local", () => {
		const conversation = { kind: "conversation", conversationId: ROOT_CONVERSATION_ID } as const;
		const create = (extra: object) => ({ id: idFromNumber<DocumentId>(2), kind: "k", ...extra }) as never;
		expect(defaultDocumentRoute(create({ scope: conversation, history: "latest", fork: "current" }))).toBe(
			"versioned",
		);
		expect(defaultDocumentRoute(create({ scope: conversation, history: "rewindable", fork: "asOf" }))).toBe(
			"versioned",
		);
		expect(defaultDocumentRoute(create({ scope: conversation, history: "latest", fork: "initial" }))).toBe("local");
		expect(defaultDocumentRoute(create({ scope: { kind: "session" } }))).toBe("local");
		expect(defaultDocumentRoute(create({ scope: { kind: "task", taskId: idFromNumber<TaskId>(3) } }))).toBe("local");
	});
});

describe("a Dolt cell's dialect", () => {
	it("keeps `--` comments out of every migration statement (a Dolt store drops them and its schema shape cannot represent them)", async () => {
		const { createSqliteMigrations } = await import("../src/storage/sqlite/index.ts");
		const { createSqliteSchema } = await import("../src/storage/sqlite/index.ts");
		for (const prefix of [undefined, "vsl_"]) {
			for (const migration of createSqliteMigrations(createSqliteSchema(prefix))) {
				for (const statement of migration.statements) expect(statement).not.toContain("--");
			}
		}
	});
});

describe("NativeDatabase commit option", () => {
	it("hands a transaction's commit to the native storage only when asked, and syncs after it", async () => {
		const { NativeDatabase } = await import("../src/storage/sqlite/index.ts");
		const calls: Array<{ options: unknown }> = [];
		let synced = 0;
		const native = {
			sql: { exec: () => ({ toArray: () => [], one: () => ({}) }) },
			transaction: async <T>(callback: () => Promise<T>, options?: { commit?: (value: T) => string | null }) => {
				calls.push({ options });
				const value = await callback();
				return value;
			},
			sync: async () => {
				synced++;
			},
		};
		const database = new NativeDatabase(native as never);
		await database.transaction(async () => 1);
		await database.transaction(async () => 7, undefined, { commit: (value) => `commit ${value}` });
		expect(calls[0]!.options).toBeUndefined();
		expect((calls[1]!.options as { commit: (value: number) => string }).commit(7)).toBe("commit 7");
		expect(synced).toBe(2);
	});
});
