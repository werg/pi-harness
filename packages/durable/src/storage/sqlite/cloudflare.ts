import { DurableObjectSqliteDatabase, type DurableObjectSqliteStorage } from "@earendil-works/chord/sqlite/cloudflare";
import { applySqliteMigrationsForSchema } from "./migrations.ts";
import { createSqliteSchema } from "./schema.ts";
import { SqliteStorage, type SqliteStorageOptions } from "./storage.ts";

export { DurableObjectSqliteDatabase, type DurableObjectSqliteStorage } from "@earendil-works/chord/sqlite/cloudflare";

/**
 * Open durable storage on a SQLite-backed Durable Object, optionally under a table prefix. Calls given the same
 * storage object share an operation queue.
 */
export async function openDurableObjectSqliteStorage(
	storage: DurableObjectSqliteStorage,
	options: SqliteStorageOptions = {},
): Promise<SqliteStorage> {
	const database = new DurableObjectSqliteDatabase(storage);
	try {
		await applySqliteMigrationsForSchema(database, createSqliteSchema(options.prefix));
	} catch (error) {
		try {
			await database.close();
		} catch (closeError) {
			throw new AggregateError([error, closeError], "SQLite initialization and close failed");
		}
		throw error;
	}
	return SqliteStorage.open(database, options);
}
