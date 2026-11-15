import type { Context, JsonValue } from "@earendil-works/chord";
import { idFromNumber, seqFromNumber } from "../../ids.ts";
import type {
	ConversationId,
	ConversationQuery,
	ConversationRecord,
	Cursor,
	DocumentAddress,
	DocumentCreate,
	DocumentId,
	DocumentPoint,
	DocumentQuery,
	DocumentRecord,
	EntryId,
	EntryQuery,
	EntryRecord,
	Id,
	Page,
	Seq,
	Storage,
	StorageWrite,
	StoredDocument,
	SubmissionId,
	SubmissionQuery,
	SubmissionRecord,
	TaskId,
	TaskQuery,
	TaskRecord,
} from "../../types.ts";
import type { SqliteDatabase, SqliteExecutor } from "./database.ts";
import { applySqliteMigrationsInTransaction, createSqliteMigrations } from "./migrations.ts";
import { createSqliteSchema } from "./schema.ts";
import { SqliteStorage } from "./storage.ts";

/** Which table set holds a record. */
export type CompositeRoute = "versioned" | "local";

/** The table prefix of the local set. The `vsl_` prefix is the one a Dolt cell's framework declares local. */
export const DEFAULT_LOCAL_PREFIX = "vsl_";

type StoredTask = TaskRecord<JsonValue, JsonValue, JsonValue>;

/**
 * A database whose transaction can end in a version-control commit (a Dolt cell's `transaction(fn, { commit })`):
 * the writes and the commit are one durable step. The commit is a function of the transaction's result, so its message
 * can name the sequence the transaction stored; `null` leaves the writes uncommitted.
 */
export interface CommittingSqliteDatabase extends SqliteDatabase {
	transaction<T>(
		callback: (transaction: SqliteExecutor) => Promise<T>,
		afterCommit?: (value: T) => void,
		options?: { readonly commit?: (value: T) => string | null },
	): Promise<T>;
}

export type CompositeSqliteStorageOptions = {
	/** Prefix of the local table set. Default `vsl_`. */
	readonly localPrefix?: string;
	/**
	 * The table set a new document is created in. Default {@link defaultDocumentRoute}: a conversation document that a
	 * fork inherits is versioned, everything else is local.
	 */
	readonly documentRoute?: (create: DocumentCreate) => CompositeRoute;
	/**
	 * The commit a storage transaction ends with: the version-control commit message for the writes it stored at `seq`,
	 * or null for none. Absent, no transaction commits. The database must be a {@link CommittingSqliteDatabase}.
	 */
	readonly commit?: (writes: readonly StorageWrite[], seq: Seq) => string | null;
};

/**
 * Conversation knowledge is what a fork of the store inherits: a conversation document that is initialized from the
 * source's current or as-of state. A conversation document that starts from its initial value in a fork, a session
 * document and a task document are operational: they are local.
 */
export function defaultDocumentRoute(create: DocumentCreate): CompositeRoute {
	return create.scope.kind === "conversation" && create.fork !== "initial" ? "versioned" : "local";
}

/**
 * The schema statements of both table sets, in the caller's transaction: the versioned set under the default names, the
 * local set under `localPrefix`.
 */
export async function applyCompositeSqliteMigrationsInTransaction(
	transaction: SqliteExecutor,
	localPrefix: string = DEFAULT_LOCAL_PREFIX,
): Promise<void> {
	const versioned = createSqliteSchema();
	const local = createSqliteSchema(localPrefix);
	await applySqliteMigrationsInTransaction(transaction, createSqliteMigrations(versioned), versioned);
	await applySqliteMigrationsInTransaction(transaction, createSqliteMigrations(local), local);
}

/**
 * One durable store over two table sets in one database.
 *
 * - **Versioned**: conversations, entries and the documents that are conversation knowledge. A version-control engine
 *   (a Dolt cell) can commit, fork and rewind exactly these rows.
 * - **Local**: tasks, submissions and every other document, in tables the engine leaves out of commits, resets and
 *   forks. A fork starts with none of them.
 *
 * The ID and sequence counters live in the local set and are the composite's alone, so an ID or a sequence is never
 * reused after a rewind of the versioned set, and two stores forked from one commit never share a counter. A fork's
 * local set is empty, so the counters restart above the highest ID and sequence the versioned set stores.
 *
 * One commit writes both sets in one database transaction, so the Session's atomic commit holds across them.
 */
export class CompositeSqliteStorage implements Storage {
	readonly #db: CommittingSqliteDatabase;
	readonly #versioned: SqliteStorage;
	readonly #local: SqliteStorage;
	readonly #options: CompositeSqliteStorageOptions;
	#nextId: number;
	#closed = false;
	#closing: Promise<void> | undefined;
	readonly #documentRoutes = new Map<DocumentId, CompositeRoute>();

	private constructor(
		db: CommittingSqliteDatabase,
		versioned: SqliteStorage,
		local: SqliteStorage,
		nextId: number,
		options: CompositeSqliteStorageOptions,
	) {
		this.#db = db;
		this.#versioned = versioned;
		this.#local = local;
		this.#nextId = nextId;
		this.#options = options;
	}

	/** Open the store after the caller's schema gate created both table sets ({@link applyCompositeSqliteMigrationsInTransaction}). */
	static async open(
		db: CommittingSqliteDatabase,
		options: CompositeSqliteStorageOptions = {},
	): Promise<CompositeSqliteStorage> {
		try {
			const versioned = await SqliteStorage.attach(db);
			const local = await SqliteStorage.attach(db, { prefix: options.localPrefix ?? DEFAULT_LOCAL_PREFIX });
			let counters = await local.counters();
			// A local set that never counted (a fork, or a store made before the split) starts above what the versioned
			// set stores. A set that has counted is the authority: a rewind of the versioned set must not lower it.
			if (counters.nextId <= 2 && counters.nextSeq <= 1) {
				const stored = await versioned.highWater();
				counters = {
					nextId: Math.max(counters.nextId, stored.id + 1),
					nextSeq: Math.max(counters.nextSeq, stored.seq + 1),
				};
				if (counters.nextId > 2 || counters.nextSeq > 1) {
					await db.transaction(async (transaction) => {
						await local.setCounters(transaction, counters.nextId, counters.nextSeq);
					});
				}
			}
			return new CompositeSqliteStorage(db, versioned, local, counters.nextId, options);
		} catch (error) {
			try {
				await db.close();
			} catch (closeError) {
				throw new AggregateError([error, closeError], "SQLite open and close failed");
			}
			throw error;
		}
	}

	async commit(writes: readonly StorageWrite[], adopt: (seq: Seq) => void, _context: Context): Promise<Seq> {
		this.#assertOpen();
		const routed = await this.#route(writes);
		let candidateNextId = this.#nextId;
		for (const write of writes) {
			const id =
				write.type === "document.create" || write.type === "document.copy"
					? write.record.id
					: write.type === "document.change" || write.type === "document.retire"
						? undefined
						: write.value.id;
			if (id !== undefined) candidateNextId = Math.max(candidateNextId, id + 1);
		}
		const commit = this.#options.commit;
		return this.#db.transaction(
			async (transaction) => {
				const counters = await this.#local.counters(transaction);
				const seq = seqFromNumber(counters.nextSeq);
				await this.#checkGlobalIds(transaction, routed);
				await this.#versioned.applyWrites(transaction, routed.versioned, seq);
				await this.#local.applyWrites(transaction, routed.local, seq);
				await this.#local.setCounters(
					transaction,
					Math.max(counters.nextId, candidateNextId),
					counters.nextSeq + 1,
				);
				return seq;
			},
			(seq) => {
				this.#nextId = Math.max(this.#nextId, candidateNextId);
				for (const [id, route] of routed.created) this.#documentRoutes.set(id, route);
				adopt(seq);
			},
			commit === undefined ? undefined : { commit: (seq) => commit(writes, seq) },
		);
	}

	async mintId<I extends Id<string>>(): Promise<I> {
		this.#assertOpen();
		if (!Number.isSafeInteger(this.#nextId)) throw new Error("ID space is exhausted");
		return idFromNumber<I>(this.#nextId++);
	}

	async conversation(id: ConversationId, context: Context): Promise<ConversationRecord | undefined> {
		this.#assertOpen();
		return this.#versioned.conversation(id, context);
	}

	async scanConversations(
		query: ConversationQuery,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<ConversationRecord, Cursor>> {
		this.#assertOpen();
		return this.#versioned.scanConversations(query, limit, cursor, context);
	}

	entry(id: EntryId, context: Context): Promise<{ readonly entry: EntryRecord; readonly commitSeq: Seq } | undefined>;
	entry(
		conversationId: ConversationId,
		id: EntryId,
		context: Context,
	): Promise<{ readonly entry: EntryRecord; readonly commitSeq: Seq } | undefined>;
	async entry(
		first: EntryId | ConversationId,
		second: EntryId | Context,
		third?: Context,
	): Promise<{ readonly entry: EntryRecord; readonly commitSeq: Seq } | undefined> {
		this.#assertOpen();
		return third === undefined
			? this.#versioned.entry(first as EntryId, second as Context)
			: this.#versioned.entry(first as ConversationId, second as EntryId, third);
	}

	async findLatestHeadMarker(
		conversationId: ConversationId,
		atOrBeforeEntryId: EntryId | undefined,
		context: Context,
	): Promise<(EntryRecord & { readonly head: EntryId }) | undefined> {
		this.#assertOpen();
		return this.#versioned.findLatestHeadMarker(conversationId, atOrBeforeEntryId, context);
	}

	async scanEntries(
		query: EntryQuery,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<EntryRecord, Cursor>> {
		this.#assertOpen();
		return this.#versioned.scanEntries(query, limit, cursor, context);
	}

	async task(id: TaskId, context: Context): Promise<StoredTask | undefined> {
		this.#assertOpen();
		return this.#local.task(id, context);
	}

	async scanTasks(
		query: TaskQuery,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<StoredTask, Cursor>> {
		this.#assertOpen();
		return this.#local.scanTasks(query, limit, cursor, context);
	}

	async submission(id: SubmissionId, context: Context): Promise<SubmissionRecord | undefined> {
		this.#assertOpen();
		return this.#local.submission(id, context);
	}

	async scanSubmissions(
		query: SubmissionQuery,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<SubmissionRecord, Cursor>> {
		this.#assertOpen();
		return this.#local.scanSubmissions(query, limit, cursor, context);
	}

	async submissionByRequest(
		conversationId: ConversationId,
		requestId: string,
		context: Context,
	): Promise<SubmissionRecord | undefined> {
		this.#assertOpen();
		return this.#local.submissionByRequest(conversationId, requestId, context);
	}

	async findDocument(
		address: DocumentAddress,
		at: DocumentPoint,
		context: Context,
	): Promise<DocumentRecord | undefined> {
		this.#assertOpen();
		const [versioned, local] = await Promise.all([
			this.#versioned.findDocument(address, at, context),
			this.#local.findDocument(address, at, context),
		]);
		if (versioned === undefined) return local;
		if (local === undefined) return versioned;
		// One address can have an incarnation in each set only across a change of route; the newer one is the document.
		return local.createdAt > versioned.createdAt ? local : versioned;
	}

	async document(id: DocumentId, at: DocumentPoint, context: Context): Promise<StoredDocument | undefined> {
		this.#assertOpen();
		const route = this.#documentRoutes.get(id);
		if (route !== undefined) return this.#part(route).document(id, at, context);
		const versioned = await this.#versioned.document(id, at, context);
		if (versioned !== undefined) return versioned;
		return this.#local.document(id, at, context);
	}

	async scanDocuments(
		query: DocumentQuery,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<DocumentRecord, Cursor>> {
		this.#assertOpen();
		const [versioned, local] = await Promise.all([
			this.#versioned.scanDocuments(query, limit, cursor, context),
			this.#local.scanDocuments(query, limit, cursor, context),
		]);
		const merged = [...versioned.items, ...local.items].sort((a, b) => a.id - b.id);
		const items = merged.slice(0, limit);
		if (merged.length <= limit && versioned.next === undefined && local.next === undefined) return { items };
		return { items, next: { after: items.at(-1)!.id } };
	}

	close(context: Context): Promise<void> {
		if (this.#closing === undefined) {
			this.#closed = true;
			// The parts wait for their admitted reads and never close the connection: the composite closes it once.
			this.#closing = Promise.all([this.#versioned.close(context), this.#local.close(context)]).then(() =>
				this.#db.close(),
			);
		}
		return this.#closing;
	}

	/** A record ID names one record, whichever set holds it: a write may not take an ID the other set already owns. */
	async #checkGlobalIds(
		transaction: SqliteExecutor,
		routed: { readonly versioned: readonly StorageWrite[]; readonly local: readonly StorageWrite[] },
	): Promise<void> {
		for (const [writes, other] of [
			[routed.versioned, this.#local],
			[routed.local, this.#versioned],
		] as const) {
			for (const write of writes) {
				if (write.type === "document.change" || write.type === "document.retire") continue;
				const id =
					write.type === "document.create" || write.type === "document.copy" ? write.record.id : write.value.id;
				const owner = await other.recordType(transaction, id);
				if (owner !== undefined) throw new Error(`ID ${id} already belongs to ${owner}`);
			}
		}
	}

	#part(route: CompositeRoute): SqliteStorage {
		return route === "versioned" ? this.#versioned : this.#local;
	}

	#assertOpen(): void {
		if (this.#closed) throw new Error("CompositeSqliteStorage is closed");
	}

	/** Split one commit's writes between the table sets. A change or retirement follows the set that holds its document. */
	async #route(writes: readonly StorageWrite[]): Promise<{
		readonly versioned: StorageWrite[];
		readonly local: StorageWrite[];
		readonly created: Map<DocumentId, CompositeRoute>;
	}> {
		const versioned: StorageWrite[] = [];
		const local: StorageWrite[] = [];
		const created = new Map<DocumentId, CompositeRoute>();
		const documentRoute = this.#options.documentRoute ?? defaultDocumentRoute;
		for (const write of writes) {
			let route: CompositeRoute;
			switch (write.type) {
				case "conversation":
				case "entry":
					route = "versioned";
					break;
				case "task":
				case "submission":
					route = "local";
					break;
				case "document.create":
				case "document.copy":
					route = documentRoute(write.record);
					created.set(write.record.id, route);
					break;
				case "document.change":
				case "document.retire":
					route = created.get(write.id) ?? this.#documentRoutes.get(write.id) ?? (await this.#locate(write.id));
					break;
			}
			(route === "versioned" ? versioned : local).push(write);
		}
		return { versioned, local, created };
	}

	async #locate(id: DocumentId): Promise<CompositeRoute> {
		const route: CompositeRoute = (await this.#versioned.hasDocument(id)) ? "versioned" : "local";
		this.#documentRoutes.set(id, route);
		return route;
	}
}
