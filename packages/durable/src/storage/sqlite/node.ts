import { validateSqliteSchemaPrefix } from "@earendil-works/chord/sqlite";
import { type NodeSqliteDatabaseOptions, openNodeSqliteDatabase } from "@earendil-works/chord/sqlite/node";
import { applySqliteMigrationsForSchema } from "./migrations.ts";
import { createSqliteSchema } from "./schema.ts";
import { SqliteStorage, type SqliteStorageOptions } from "./storage.ts";

export {
	NodeSqliteDatabase,
	type NodeSqliteDatabaseOptions,
	openNodeSqliteDatabase,
} from "@earendil-works/chord/sqlite/node";

/** Node SQLite connection settings plus durable storage options. */
export type NodeSqliteStorageOptions = NodeSqliteDatabaseOptions & SqliteStorageOptions;

/** Open or create file-backed durable storage using Node's built-in SQLite, optionally under a table prefix. */
export async function openNodeSqliteStorage(
	path: string,
	options: NodeSqliteStorageOptions = {},
): Promise<SqliteStorage> {
	const { prefix, ...databaseOptions } = options;
	if (prefix !== undefined) validateSqliteSchemaPrefix(prefix);
	const database = await openNodeSqliteDatabase(path, databaseOptions);
	try {
		await applySqliteMigrationsForSchema(database, createSqliteSchema(prefix));
	} catch (error) {
		try {
			await database.close();
		} catch (closeError) {
			throw new AggregateError([error, closeError], "SQLite initialization and close failed");
		}
		throw error;
	}
	return SqliteStorage.open(database, { prefix });
}
