import { DatabaseSync, type StatementSync } from "node:sqlite";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { DurableObjectSqliteDatabase as ChordDurableObjectSqliteDatabase } from "@earendil-works/chord/sqlite/cloudflare";
import {
	NodeSqliteDatabase as ChordNodeSqliteDatabase,
	openNodeSqliteDatabase as chordOpenNodeSqliteDatabase,
} from "@earendil-works/chord/sqlite/node";
import { DurableObjectSqliteDatabase } from "@earendil-works/pi-durable/storage/sqlite/cloudflare";
import { describe, expect, it } from "vitest";
import { durableObjectStorage } from "../../chord/test/sqlite/helpers.ts";
import { idFromNumber } from "../src/ids.ts";
import type { SqliteDatabase, SqliteExecutor, SqliteValue } from "../src/storage/sqlite/index.ts";
import { applySqliteMigrations, NativeDatabase, SqliteStorage } from "../src/storage/sqlite/index.ts";
import { NodeSqliteDatabase, openNodeSqliteDatabase, openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { type DocumentId, type EntryId, ROOT_CONVERSATION_ID } from "../src/types.ts";

/** Storage-level commits here have no in-memory owner to adopt them. */
const adopt = (): void => {};

type SettlementMode = "immediate" | "delay" | "reject";

class ControlledSettlementDatabase implements SqliteDatabase {
	private readonly delegate: NodeSqliteDatabase;
	private mode: SettlementMode = "immediate";
	private pendingSettlement: (() => void) | undefined;

	constructor(delegate: NodeSqliteDatabase) {
		this.delegate = delegate;
	}

	exec(sql: string): Promise<void> {
		return this.delegate.exec(sql);
	}

	run(sql: string, ...params: SqliteValue[]): Promise<void> {
		return this.delegate.run(sql, ...params);
	}

	runCounted(sql: string, ...params: SqliteValue[]): Promise<number> {
		return this.delegate.runCounted(sql, ...params);
	}

	get<T extends object>(sql: string, ...params: SqliteValue[]): Promise<T | undefined> {
		return this.delegate.get<T>(sql, ...params);
	}

	all<T extends object>(sql: string, ...params: SqliteValue[]): Promise<T[]> {
		return this.delegate.all<T>(sql, ...params);
	}

	transaction<T>(callback: (transaction: SqliteExecutor) => Promise<T>, afterCommit?: (value: T) => void): Promise<T> {
		const mode = this.mode;
		this.mode = "immediate";
		if (mode === "immediate") return this.delegate.transaction(callback, afterCommit);
		const gate = new Promise<void>((resolve) => {
			this.pendingSettlement = resolve;
		});
		return this.delegate.transaction(async (transaction) => {
			const value = await callback(transaction);
			await gate;
			if (mode === "reject") throw new Error("controlled settlement rejection");
			return value;
		}, afterCommit);
	}

	close(): Promise<void> {
		return this.delegate.close();
	}

	controlNextSettlement(mode: Exclude<SettlementMode, "immediate">): void {
		if (this.pendingSettlement !== undefined) throw new Error("A settlement is already pending");
		this.mode = mode;
	}

	settle(): void {
		const settle = this.pendingSettlement;
		if (settle === undefined) throw new Error("No settlement is pending");
		this.pendingSettlement = undefined;
		settle();
	}
}

class PrepareCountingDatabaseSync extends DatabaseSync {
	private readonly prepareCounts = new Map<string, number>();

	override prepare(sql: string): StatementSync {
		this.prepareCounts.set(sql, (this.prepareCounts.get(sql) ?? 0) + 1);
		return super.prepare(sql);
	}

	repeatedPrepares(): string[] {
		return [...this.prepareCounts].filter(([, count]) => count > 1).map(([sql]) => sql);
	}

	prepareCount(sql: string): number {
		return this.prepareCounts.get(sql) ?? 0;
	}
}

describe("portable SQLite facade settlement", () => {
	it("adopts a native transaction only after sync and before the next transaction", async () => {
		const underlying = durableObjectStorage();
		const events: string[] = [];
		const database = new NativeDatabase({
			sql: {
				exec(sql, ...values) {
					const cursor = underlying.sql.exec(sql, ...values);
					return {
						toArray: () => cursor.toArray(),
						one: () => {
							const row = cursor.next();
							if (row.done) throw new Error("No row");
							return row.value;
						},
					};
				},
			},
			transaction: (callback) => underlying.transaction(callback),
			sync: async () => {
				events.push("sync");
			},
		});
		try {
			const first = database.transaction(
				async (tx) => {
					await tx.exec("CREATE TABLE native_adoption (value INTEGER)");
					return 1;
				},
				(value) => events.push(`adopt ${value}`),
			);
			const second = database.transaction(async () => {
				events.push("second starts");
			});
			await Promise.all([first, second]);
			expect(events.slice(0, 3)).toEqual(["sync", "adopt 1", "second starts"]);
		} finally {
			await database.close();
		}
	});

	it("evicts old prepared statements while retaining recently reused statements", async () => {
		const connection = new PrepareCountingDatabaseSync(":memory:");
		const database = new NodeSqliteDatabase(connection);
		try {
			for (let i = 0; i < 256; i++) await database.get(`SELECT ${i} AS n`);
			await database.transaction(async (transaction) => {
				expect(await transaction.get("SELECT 0 AS n")).toEqual({ n: 0 });
			});
			for (let i = 256; i < 300; i++) await database.get(`SELECT ${i} AS n`);
			expect(await database.get("SELECT 0 AS n")).toEqual({ n: 0 });
			expect(connection.prepareCount("SELECT 0 AS n")).toBe(1);
			expect(await database.get("SELECT 1 AS n")).toEqual({ n: 1 });
			expect(connection.prepareCount("SELECT 1 AS n")).toBe(2);
		} finally {
			await database.close();
		}
	});

	it("re-exports the Chord driver classes and Node opener without wrapping them", () => {
		expect(NodeSqliteDatabase).toBe(ChordNodeSqliteDatabase);
		expect(openNodeSqliteDatabase).toBe(chordOpenNodeSqliteDatabase);
		expect(DurableObjectSqliteDatabase).toBe(ChordDurableObjectSqliteDatabase);
	});

	it("prepares each storage statement once per connection and reuses it across transactions", async () => {
		const connection = new PrepareCountingDatabaseSync(":memory:");
		const database = new NodeSqliteDatabase(connection);
		await applySqliteMigrations(database);
		const storage = await SqliteStorage.open(database);
		await storage.commit([{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }], adopt, BACKGROUND_CONTEXT);
		await storage.commit(
			Array.from({ length: 100 }, (_, index) => ({
				type: "entry" as const,
				value: { id: idFromNumber<EntryId>(index + 2), conversationId: ROOT_CONVERSATION_ID, kind: "cached" },
			})),
			adopt,
			BACKGROUND_CONTEXT,
		);
		await storage.commit(
			[
				{
					type: "entry",
					value: { id: idFromNumber<EntryId>(102), conversationId: ROOT_CONVERSATION_ID, kind: "cached-again" },
				},
			],
			adopt,
			BACKGROUND_CONTEXT,
		);
		expect((await storage.entry(idFromNumber<EntryId>(2), BACKGROUND_CONTEXT))?.entry.kind).toBe("cached");
		expect((await storage.entry(idFromNumber<EntryId>(102), BACKGROUND_CONTEXT))?.entry.kind).toBe("cached-again");
		await expect(
			storage.commit([{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }], adopt, BACKGROUND_CONTEXT),
		).rejects.toThrow("ID 1 already belongs to conversation");
		expect((await storage.entry(idFromNumber<EntryId>(2), BACKGROUND_CONTEXT))?.entry.kind).toBe("cached");
		expect(connection.repeatedPrepares()).toEqual([]);
		await storage.close(BACKGROUND_CONTEXT);
	});

	it("reads storage commits in call order", async () => {
		const database = await openNodeSqliteDatabase(":memory:");
		await applySqliteMigrations(database);
		const storage = await SqliteStorage.open(database);
		const commit = storage.commit(
			[{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } }],
			adopt,
			BACKGROUND_CONTEXT,
		);
		const read = storage.conversation(ROOT_CONVERSATION_ID, BACKGROUND_CONTEXT);
		await commit;
		expect(await read).toEqual({ id: ROOT_CONVERSATION_ID });
		await storage.close(BACKGROUND_CONTEXT);
	});

	it("lets admitted multi-query reads finish before storage closes", async () => {
		const storage = await openNodeSqliteStorage(":memory:");
		const entryId = idFromNumber<EntryId>(2);
		await storage.commit(
			[
				{ type: "conversation", value: { id: ROOT_CONVERSATION_ID } },
				{ type: "entry", value: { id: entryId, conversationId: ROOT_CONVERSATION_ID, kind: "probe" } },
			],
			adopt,
			BACKGROUND_CONTEXT,
		);
		const scan = storage.scanEntries({ conversationId: ROOT_CONVERSATION_ID }, 10, undefined, BACKGROUND_CONTEXT);
		const entry = storage.entry(ROOT_CONVERSATION_ID, entryId, BACKGROUND_CONTEXT);
		const head = storage.findLatestHeadMarker(ROOT_CONVERSATION_ID, undefined, BACKGROUND_CONTEXT);
		const closed = storage.close(BACKGROUND_CONTEXT);
		// A repeated close settles only when the database is closed.
		expect(storage.close(BACKGROUND_CONTEXT)).toBe(closed);
		expect((await scan).items.map((item) => item.id)).toEqual([entryId]);
		expect((await entry)?.entry.kind).toBe("probe");
		expect(await head).toBeUndefined();
		await closed;
		await expect(
			storage.scanEntries({ conversationId: ROOT_CONVERSATION_ID }, 10, undefined, BACKGROUND_CONTEXT),
		).rejects.toThrow("SqliteStorage is closed");
	});

	it("awaits async transaction settlement and adopts IDs only after success", async () => {
		const database = new ControlledSettlementDatabase(await openNodeSqliteDatabase(":memory:"));
		database.controlNextSettlement("delay");
		const opening = applySqliteMigrations(database);
		let opened = false;
		void opening.then(() => {
			opened = true;
		});
		await Promise.resolve();
		expect(opened).toBe(false);
		database.settle();
		await opening;
		const storage = await SqliteStorage.open(database);

		database.controlNextSettlement("delay");
		const committing = storage.commit(
			[
				{
					type: "entry",
					value: { id: idFromNumber<EntryId>(100), conversationId: ROOT_CONVERSATION_ID, kind: "settled" },
				},
			],
			adopt,
			BACKGROUND_CONTEXT,
		);
		let committed = false;
		void committing.then(() => {
			committed = true;
		});
		await Promise.resolve();
		expect(committed).toBe(false);
		expect(await storage.mintId<EntryId>()).toBe(2);
		database.settle();
		await expect(committing).resolves.toBe(1);
		expect(await storage.mintId<EntryId>()).toBe(101);

		database.controlNextSettlement("reject");
		const rejected = storage.commit(
			[
				{
					type: "entry",
					value: { id: idFromNumber<EntryId>(200), conversationId: ROOT_CONVERSATION_ID, kind: "rejected" },
				},
			],
			adopt,
			BACKGROUND_CONTEXT,
		);
		expect(await storage.mintId<EntryId>()).toBe(102);
		database.settle();
		await expect(rejected).rejects.toThrow("controlled settlement rejection");
		expect(await storage.mintId<EntryId>()).toBe(103);
		expect(await storage.entry(idFromNumber<EntryId>(200), BACKGROUND_CONTEXT)).toBeUndefined();
		await storage.close(BACKGROUND_CONTEXT);
	});

	it("reads a document from one committed state while a commit replaces its base", async () => {
		const id = idFromNumber<DocumentId>(5);
		// Each yield count starts the commit at a different point of the read's record and revision queries.
		for (let yields = 0; yields < 16; yields++) {
			const storage = await openNodeSqliteStorage(":memory:");
			await storage.commit(
				[
					{
						type: "document.create",
						record: { id, kind: "replaced", scope: { kind: "session" } },
						content: { kind: "base", version: 1, value: { value: 1 } },
					},
				],
				adopt,
				BACKGROUND_CONTEXT,
			);
			const read = storage.document(id, "current", BACKGROUND_CONTEXT);
			for (let index = 0; index < yields; index++) await Promise.resolve();
			const replace = storage.commit(
				[{ type: "document.change", id, content: { kind: "base", version: 1, value: { value: 2 } } }],
				adopt,
				BACKGROUND_CONTEXT,
			);
			const [stored] = await Promise.all([read, replace]);
			expect([{ value: 1 }, { value: 2 }]).toContainEqual(stored?.value);
			await storage.close(BACKGROUND_CONTEXT);
		}
	});
});
