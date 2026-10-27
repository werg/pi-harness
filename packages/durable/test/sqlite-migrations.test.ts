import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { afterEach, describe, expect, it } from "vitest";
import { idFromNumber } from "../src/ids.ts";
import {
	applySqliteMigrations,
	applySqliteMigrationsInTransaction,
	CURRENT_SQLITE_SCHEMA_VERSION,
	SQLITE_MIGRATIONS,
	type SqliteMigration,
	SqliteStorage,
} from "../src/storage/sqlite/index.ts";
import { openNodeSqliteDatabase, openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { readJsonRow, writeJsonPayload } from "../src/storage/sqlite/payload.ts";
import { type DocumentId, type EntryId, ROOT_CONVERSATION_ID } from "../src/types.ts";

const directories = new Set<string>();

async function databasePath(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-durable-migrations-"));
	directories.add(directory);
	return join(directory, "storage.sqlite");
}

afterEach(async () => {
	for (const directory of directories) await rm(directory, { recursive: true, force: true });
	directories.clear();
});

describe("durable SQLite migrations", () => {
	it("reopens the persisted wake schedule and its publication acknowledgement", async () => {
		const path = await databasePath();
		const id = idFromNumber<DocumentId>(2);
		const initial = await openNodeSqliteStorage(path);
		await initial.commit(
			[
				{
					type: "document.create",
					record: { id, kind: "pi.wake", scope: { kind: "session" } },
					content: { kind: "base", version: 1, value: { revision: 7, publishedRevision: 6, wakeAt: 12345 } },
				},
			],
			() => {},
			BACKGROUND_CONTEXT,
		);
		await initial.close(BACKGROUND_CONTEXT);
		const reopened = await openNodeSqliteStorage(path);
		try {
			expect((await reopened.document(id, "current", BACKGROUND_CONTEXT))?.value).toEqual({
				revision: 7,
				publishedRevision: 6,
				wakeAt: 12345,
			});
		} finally {
			await reopened.close(BACKGROUND_CONTEXT);
		}
	});

	it("keeps prefixed chunked schemas independent in one database", async () => {
		const path = await databasePath();
		const first = await openNodeSqliteStorage(path, { prefix: "one_" });
		const second = await openNodeSqliteStorage(path, { prefix: "two_" });
		try {
			await first.commit(
				[{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }],
				() => {},
				BACKGROUND_CONTEXT,
			);
			expect(await first.conversation(ROOT_CONVERSATION_ID, BACKGROUND_CONTEXT)).toEqual({
				id: ROOT_CONVERSATION_ID,
			});
			expect(await second.conversation(ROOT_CONVERSATION_ID, BACKGROUND_CONTEXT)).toBeUndefined();
			await second.commit(
				[{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }],
				() => {},
				BACKGROUND_CONTEXT,
			);
			expect(await second.conversation(ROOT_CONVERSATION_ID, BACKGROUND_CONTEXT)).toEqual({
				id: ROOT_CONVERSATION_ID,
			});
		} finally {
			await first.close(BACKGROUND_CONTEXT);
			await second.close(BACKGROUND_CONTEXT);
		}
	});

	it("preserves chunked version 2 payloads across a failed schema transaction", async () => {
		const database = await openNodeSqliteDatabase(await databasePath());
		try {
			await applySqliteMigrations(database);
			await database.run(
				"INSERT INTO tasks (id, conversation_id, kind, status, abort_requested, background) VALUES (3, 1, ?, 'waiting', 0, 0)",
				JSON.stringify("old"),
			);
			const record = { id: 3, input: "🙂".repeat(600_000) };
			await database.transaction((transaction) => writeJsonPayload("payload_chunks", transaction, 3, 0, record));
			const before = await database.get("SELECT count(*) AS count FROM payload_chunks WHERE id = 3");
			const failed: readonly SqliteMigration[] = [
				...SQLITE_MIGRATIONS,
				{ version: 3, statements: ["CREATE TABLE migration_probe (value TEXT)", "THIS IS NOT SQL"] },
			];
			await expect(applySqliteMigrations(database, failed)).rejects.toThrow();
			expect(await database.get("SELECT version FROM durable_schema")).toEqual({ version: 2 });
			expect(await database.get("SELECT status FROM tasks WHERE id = 3")).toEqual({ status: "waiting" });
			expect(await database.get("SELECT count(*) AS count FROM payload_chunks WHERE id = 3")).toEqual(before);
			expect(
				JSON.parse(
					(await readJsonRow<{ record: string }>(
						"payload_chunks",
						database,
						"SELECT id FROM tasks WHERE id = ?",
						[3],
					))!.record,
				),
			).toEqual(record);
		} finally {
			await database.close();
		}
	});

	it("rejects the unrelated inline version 2 schema without discarding its records", async () => {
		const path = await databasePath();
		const database = await openNodeSqliteDatabase(path);
		await database.exec("CREATE TABLE durable_schema (singleton INTEGER PRIMARY KEY, version INTEGER NOT NULL)");
		await database.run("INSERT INTO durable_schema VALUES (1, 2)");
		await database.exec("CREATE TABLE tasks (id INTEGER PRIMARY KEY, record TEXT NOT NULL)");
		await database.run("INSERT INTO tasks VALUES (3, ?)", JSON.stringify({ retained: true }));
		await expect(SqliteStorage.open(database)).rejects.toThrow();
		const reopened = await openNodeSqliteDatabase(path);
		try {
			expect(await reopened.get("SELECT version FROM durable_schema")).toEqual({ version: 2 });
			expect(await reopened.get("SELECT record FROM tasks WHERE id = 3")).toEqual({
				record: JSON.stringify({ retained: true }),
			});
		} finally {
			await reopened.close();
		}
	});

	it("rejects the pre-release inline-payload baseline without modifying its state", async () => {
		const path = await databasePath();
		const database = await openNodeSqliteDatabase(path);
		await database.exec("CREATE TABLE durable_schema (singleton INTEGER PRIMARY KEY, version INTEGER NOT NULL)");
		await database.run("INSERT INTO durable_schema VALUES (1, 1)");
		await database.exec("CREATE TABLE pre_release_data (value TEXT)");
		await database.run("INSERT INTO pre_release_data VALUES ('untouched')");
		await expect(applySqliteMigrations(database)).rejects.toThrow("predates the supported fresh-state baseline");
		expect(await database.get("SELECT version FROM durable_schema")).toEqual({ version: 1 });
		expect(await database.get("SELECT value FROM pre_release_data")).toEqual({ value: "untouched" });
		await database.close();
	});

	it("composes platform, Pi and domain initialization in one rollback boundary", async () => {
		const database = await openNodeSqliteDatabase(await databasePath());
		try {
			const failure = new Error("domain initialization failed");
			await expect(
				database.transaction(async (transaction) => {
					await transaction.exec("CREATE TABLE platform_state (key TEXT PRIMARY KEY)");
					await applySqliteMigrationsInTransaction(transaction);
					await transaction.exec("CREATE TABLE domain_cards (id TEXT PRIMARY KEY)");
					throw failure;
				}),
			).rejects.toBe(failure);
			expect(await database.all("SELECT name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'")).toEqual([]);
			await database.transaction(async (transaction) => {
				await transaction.exec("CREATE TABLE platform_state (key TEXT PRIMARY KEY)");
				await applySqliteMigrationsInTransaction(transaction);
				await transaction.exec("CREATE TABLE domain_cards (id TEXT PRIMARY KEY)");
			});
			expect(await database.get("SELECT version FROM durable_schema")).toEqual({
				version: CURRENT_SQLITE_SCHEMA_VERSION,
			});
		} finally {
			await database.close();
		}
	});

	it("creates the current schema and can be applied repeatedly", async () => {
		const database = await openNodeSqliteDatabase(await databasePath());
		try {
			await applySqliteMigrations(database);
			await applySqliteMigrations(database);
			expect(await database.get("SELECT version FROM durable_schema WHERE singleton = 1")).toEqual({
				version: CURRENT_SQLITE_SCHEMA_VERSION,
			});
			expect(await database.get("SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1")).toEqual({
				next_id: "2",
				next_seq: 1,
			});
		} finally {
			await database.close();
		}
	});

	it("rejects a database newer than the portable core", async () => {
		const path = await databasePath();
		const database = await openNodeSqliteDatabase(path);
		await applySqliteMigrations(database);
		await database.run(
			"UPDATE durable_schema SET version = ? WHERE singleton = 1",
			CURRENT_SQLITE_SCHEMA_VERSION + 1,
		);
		await database.close();

		await expect(openNodeSqliteStorage(path)).rejects.toThrow("is newer than supported version");
	});

	it("rolls initial bootstrap and every pending migration back together", async () => {
		const database = await openNodeSqliteDatabase(await databasePath());
		try {
			const failed: readonly SqliteMigration[] = [
				{
					version: 1,
					statements: [
						"CREATE TABLE migration_first (value TEXT) STRICT",
						"INSERT INTO migration_first (value) VALUES ('retained')",
					],
				},
				{
					version: 2,
					statements: ["CREATE TABLE migration_second (value TEXT) STRICT", "THIS IS NOT SQL"],
				},
			];
			await expect(applySqliteMigrations(database, failed)).rejects.toThrow();
			expect(
				await database.get(
					"SELECT count(*) AS count FROM sqlite_schema WHERE name IN ('durable_schema', 'migration_first', 'migration_second')",
				),
			).toEqual({ count: 0 });

			await applySqliteMigrations(database, [
				failed[0],
				{ version: 2, statements: ["CREATE TABLE migration_second (value TEXT) STRICT"] },
			]);
			expect(await database.get("SELECT version FROM durable_schema WHERE singleton = 1")).toEqual({
				version: 2,
			});
			expect(await database.get("SELECT value FROM migration_first")).toEqual({
				value: "retained",
			});
		} finally {
			await database.close();
		}
	});

	it("rolls a failed migration back and preserves stored data for a successful retry", async () => {
		const path = await databasePath();
		const storage = await openNodeSqliteStorage(path);
		await storage.commit(
			[
				{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
				{
					type: "entry",
					value: {
						id: idFromNumber<EntryId>(2),
						conversationId: ROOT_CONVERSATION_ID,
						kind: "retained",
						data: { retained: true },
					},
				},
			],
			() => {},
			BACKGROUND_CONTEXT,
		);
		await storage.close(BACKGROUND_CONTEXT);

		const database = await openNodeSqliteDatabase(path);
		const nextVersion = CURRENT_SQLITE_SCHEMA_VERSION + 1;
		const failedMigrations: readonly SqliteMigration[] = [
			...SQLITE_MIGRATIONS,
			{
				version: nextVersion,
				statements: ["CREATE TABLE migration_probe (value TEXT) STRICT", "THIS IS NOT SQL"],
			},
		];
		await expect(applySqliteMigrations(database, failedMigrations)).rejects.toThrow();
		expect(await database.get("SELECT version FROM durable_schema WHERE singleton = 1")).toEqual({
			version: CURRENT_SQLITE_SCHEMA_VERSION,
		});
		expect(
			await database.get(
				"SELECT count(*) AS count FROM sqlite_schema WHERE type = 'table' AND name = 'migration_probe'",
			),
		).toEqual({ count: 0 });

		const successfulMigrations: readonly SqliteMigration[] = [
			...SQLITE_MIGRATIONS,
			{ version: nextVersion, statements: ["CREATE TABLE migration_probe (value TEXT) STRICT"] },
		];
		await applySqliteMigrations(database, successfulMigrations);
		expect(await database.get("SELECT version FROM durable_schema WHERE singleton = 1")).toEqual({
			version: nextVersion,
		});
		expect(
			await readJsonRow("payload_chunks", database, "SELECT id, commit_seq FROM entries WHERE id = 2", []),
		).toEqual({
			id: 2,
			record: JSON.stringify({
				id: 2,
				conversationId: ROOT_CONVERSATION_ID,
				kind: "retained",
				data: { retained: true },
			}),
			commit_seq: 1,
		});
		expect(await database.get("SELECT next_id, next_seq FROM durable_metadata WHERE singleton = 1")).toEqual({
			next_id: "3",
			next_seq: 2,
		});
		await database.close();
	});
});
