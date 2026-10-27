import type { SqliteDatabase, SqliteExecutor } from "./database.ts";
import { SQLITE_PAYLOAD_CHUNK_BYTES } from "./payload.ts";
import { DEFAULT_SQLITE_SCHEMA, type SqliteSchema } from "./schema.ts";

export type SqliteMigration = {
	readonly version: number;
	readonly statements: readonly string[];
};

/** Immutable migration history for the validated schema identifiers. */
export function createSqliteMigrations(schema: SqliteSchema): readonly SqliteMigration[] {
	// next_id is TEXT because node:sqlite rejects INTEGER results outside JavaScript's safe integer range.
	const initialSchema: readonly string[] = [
		`CREATE TABLE ${schema.tables.durableMetadata} (
		singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
		next_id TEXT NOT NULL,
		next_seq INTEGER NOT NULL
	) STRICT`,
		`INSERT INTO ${schema.tables.durableMetadata} (singleton, next_id, next_seq) VALUES (1, '2', 1)`,
		`CREATE TABLE ${schema.tables.recordIds} (
		id INTEGER PRIMARY KEY,
		record_type TEXT NOT NULL CHECK (record_type IN ('conversation', 'entry', 'task', 'submission', 'document'))
	) STRICT`,
		`CREATE TABLE ${schema.tables.conversations} (
		id INTEGER PRIMARY KEY,
		owner_conversation_id INTEGER,
		owner_task_id INTEGER
	) STRICT`,
		`CREATE INDEX ${schema.indexes.conversationsByOwnerConversation} ON ${schema.tables.conversations} (owner_conversation_id, id)`,
		`CREATE INDEX ${schema.indexes.conversationsByOwnerTask} ON ${schema.tables.conversations} (owner_task_id, id)`,
		`CREATE TABLE ${schema.tables.entries} (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		head INTEGER,
		commit_seq INTEGER NOT NULL
	) STRICT`,
		`CREATE INDEX ${schema.indexes.entriesByConversation} ON ${schema.tables.entries} (conversation_id, id DESC)`,
		`CREATE INDEX ${schema.indexes.entryHeadsByConversation} ON ${schema.tables.entries} (conversation_id, id DESC) WHERE head IS NOT NULL`,
		`CREATE TABLE ${schema.tables.tasks} (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		kind TEXT NOT NULL,
		status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'waiting', 'completing', 'terminal')),
		abort_requested INTEGER NOT NULL CHECK (abort_requested IN (0, 1)),
		background INTEGER NOT NULL CHECK (background IN (0, 1))
	) STRICT`,
		`CREATE INDEX ${schema.indexes.tasksByStatus} ON ${schema.tables.tasks} (status, id)`,
		`CREATE INDEX ${schema.indexes.tasksByConversation} ON ${schema.tables.tasks} (conversation_id, id)`,
		`CREATE INDEX ${schema.indexes.tasksByKind} ON ${schema.tables.tasks} (kind, id)`,
		`CREATE INDEX ${schema.indexes.tasksByAbortRequested} ON ${schema.tables.tasks} (abort_requested, id)`,
		`CREATE INDEX ${schema.indexes.tasksByBackground} ON ${schema.tables.tasks} (background, id)`,
		`CREATE TABLE ${schema.tables.submissions} (
		id INTEGER PRIMARY KEY,
		conversation_id INTEGER NOT NULL,
		request_id TEXT,
		status TEXT NOT NULL CHECK (status IN ('queued', 'placed', 'done', 'unanswered'))
	) STRICT`,
		`CREATE INDEX ${schema.indexes.submissionsByRequest} ON ${schema.tables.submissions} (conversation_id, request_id)`,
		`CREATE INDEX ${schema.indexes.submissionsByConversation} ON ${schema.tables.submissions} (conversation_id, id)`,
		`CREATE INDEX ${schema.indexes.submissionsByStatus} ON ${schema.tables.submissions} (status, id)`,
		`CREATE TABLE ${schema.tables.documents} (
		id INTEGER PRIMARY KEY,
		kind TEXT NOT NULL,
		family INTEGER NOT NULL CHECK (family IN (0, 1)),
		key_value TEXT NOT NULL,
		scope_kind TEXT NOT NULL CHECK (scope_kind IN ('session', 'conversation', 'task')),
		owner_id INTEGER NOT NULL,
		created_at INTEGER NOT NULL,
		retired_at INTEGER
	) STRICT`,
		`CREATE INDEX ${schema.indexes.documentsByAddress}
		ON ${schema.tables.documents} (kind, scope_kind, owner_id, family, key_value, created_at DESC, retired_at)`,
		`CREATE INDEX ${schema.indexes.documentsByScope} ON ${schema.tables.documents} (scope_kind, owner_id, id)`,
		`CREATE INDEX ${schema.indexes.documentsByScopeKind} ON ${schema.tables.documents} (scope_kind, owner_id, kind, id)`,
		`CREATE TABLE ${schema.tables.documentRevisions} (
		document_id INTEGER NOT NULL,
		seq INTEGER NOT NULL,
		kind TEXT NOT NULL CHECK (kind IN ('base', 'delta')),
		version INTEGER NOT NULL,
		PRIMARY KEY (document_id, seq)
	) STRICT`,
		`CREATE INDEX ${schema.indexes.documentRevisionsByKind} ON ${schema.tables.documentRevisions} (document_id, kind, seq DESC)`,
		`CREATE TABLE ${schema.tables.payloadChunks} (
		id INTEGER NOT NULL,
		revision INTEGER NOT NULL CHECK (revision >= 0),
		ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
		byte_length INTEGER NOT NULL CHECK (byte_length > 0),
		data BLOB NOT NULL CHECK (length(data) > 0 AND length(data) <= ${SQLITE_PAYLOAD_CHUNK_BYTES}),
		PRIMARY KEY (id, revision, ordinal)
	) STRICT`,
	];

	return [{ version: 2, statements: initialSchema }];
}

export const SQLITE_MIGRATIONS = createSqliteMigrations(DEFAULT_SQLITE_SCHEMA);

export const CURRENT_SQLITE_SCHEMA_VERSION = SQLITE_MIGRATIONS.at(-1)?.version ?? 0;

type SchemaRow = { readonly version: number };

/** Apply all pending schema migrations atomically. */
export async function applySqliteMigrations(
	database: SqliteDatabase,
	migrations: readonly SqliteMigration[] = SQLITE_MIGRATIONS,
): Promise<void> {
	await applySqliteMigrationsForSchema(database, DEFAULT_SQLITE_SCHEMA, migrations);
}

/** Apply the selected schema history atomically. */
export async function applySqliteMigrationsForSchema(
	database: SqliteDatabase,
	schema: SqliteSchema,
	migrations: readonly SqliteMigration[] = createSqliteMigrations(schema),
): Promise<void> {
	await database.transaction((transaction) => applySqliteMigrationsInTransaction(transaction, migrations, schema));
}

/** Compose Pi migrations with the caller's schema lifecycle in its existing transaction. */
export async function applySqliteMigrationsInTransaction(
	transaction: SqliteExecutor,
	migrations: readonly SqliteMigration[] = SQLITE_MIGRATIONS,
	schema: SqliteSchema = DEFAULT_SQLITE_SCHEMA,
): Promise<void> {
	if ((migrations[0]?.version ?? 1) < 1) throw new Error("Durable SQLite migrations require positive versions");
	for (let index = 0; index < migrations.length; index++) {
		if (migrations[index]?.version !== (migrations[0]?.version ?? 1) + index) {
			throw new Error("Durable SQLite migrations must have contiguous positive versions");
		}
	}

	await transaction.exec(`CREATE TABLE IF NOT EXISTS ${schema.tables.durableSchema} (
			singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
			version INTEGER NOT NULL CHECK (version >= 0)
		) STRICT`);
	await transaction.run(`INSERT OR IGNORE INTO ${schema.tables.durableSchema} (singleton, version) VALUES (1, 0)`);
	const row = await transaction.get<SchemaRow>(
		`SELECT version FROM ${schema.tables.durableSchema} WHERE singleton = 1`,
	);
	if (row === undefined) throw new Error("Durable SQLite schema metadata is missing");
	const currentVersion = migrations.at(-1)?.version ?? 0;
	if (row.version > currentVersion) {
		throw new Error(`Durable SQLite schema version ${row.version} is newer than supported version ${currentVersion}`);
	}
	if (row.version !== 0 && row.version < (migrations[0]?.version ?? 1)) {
		throw new Error(`Durable SQLite schema version ${row.version} predates the supported fresh-state baseline`);
	}
	for (const migration of migrations) {
		if (migration.version <= row.version) continue;
		for (const statement of migration.statements) await transaction.exec(statement);
		await transaction.run(
			`UPDATE ${schema.tables.durableSchema} SET version = ? WHERE singleton = 1`,
			migration.version,
		);
	}
}
