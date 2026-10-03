import type { SqliteExecutor, SqliteValue } from "./database.ts";

// Workerd limits an entire SQLite row to 2 MB, including query result rows. Never concatenate in SQL.
export const SQLITE_PAYLOAD_CHUNK_BYTES = 512 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

type ChunkRow = {
	readonly id: number;
	readonly payload_revision?: number;
	readonly chunk_ordinal: number | null;
	readonly chunk_length: number | null;
	readonly chunk_data: Uint8Array | ArrayBuffer | null;
};

/** Hydrate a bounded, indexed selection in one statement, preserving its snapshot and avoiding per-record queries. */
export async function readJsonRows<T extends object>(
	executor: SqliteExecutor,
	selection: string,
	params: readonly SqliteValue[],
	options: { readonly revision?: boolean; readonly descending?: boolean } = {},
): Promise<T[]> {
	const revision = options.revision === true;
	const rows = await executor.all<ChunkRow>(
		`WITH selected AS (${selection})
		SELECT selected.*, chunks.ordinal AS chunk_ordinal, chunks.byte_length AS chunk_length,
			chunks.data AS chunk_data
		FROM selected LEFT JOIN payload_chunks AS chunks
			ON chunks.id = selected.id AND chunks.revision = ${revision ? "selected.payload_revision" : "0"}
		ORDER BY ${revision ? "selected.payload_revision" : "selected.id"} ${options.descending ? "DESC" : "ASC"}, chunks.ordinal`,
		...params,
	);
	const values: T[] = [];
	let position = 0;
	while (position < rows.length) {
		const first = rows[position]!;
		if (first.chunk_length === null) throw new Error(`SQLite payload ${first.id} is missing`);
		const bytes = new Uint8Array(first.chunk_length);
		let offset = 0;
		let ordinal = 0;
		while (position < rows.length) {
			const row = rows[position]!;
			if (row.id !== first.id || row.payload_revision !== first.payload_revision) break;
			if (row.chunk_ordinal !== ordinal++ || row.chunk_length !== bytes.length || row.chunk_data === null) {
				throw new Error(`SQLite payload ${first.id} has inconsistent chunks`);
			}
			const chunk = row.chunk_data instanceof Uint8Array ? row.chunk_data : new Uint8Array(row.chunk_data);
			if (offset + chunk.byteLength > bytes.length) throw new Error(`SQLite payload ${first.id} exceeds its length`);
			bytes.set(chunk, offset);
			offset += chunk.byteLength;
			position++;
		}
		if (offset !== bytes.length) throw new Error(`SQLite payload ${first.id} is incomplete`);
		const { chunk_ordinal: _ordinal, chunk_length: _length, chunk_data: _data, ...metadata } = first;
		values.push({ ...metadata, [revision ? "content" : "record"]: decoder.decode(bytes) } as T);
	}
	return values;
}

/** Replace a complete JSON value in the caller's transaction, using the same representation at every size. */
export async function writeJsonPayload(
	executor: SqliteExecutor,
	id: number,
	revision: number,
	value: unknown,
): Promise<void> {
	const bytes = encoder.encode(JSON.stringify(value));
	await executor.run("DELETE FROM payload_chunks WHERE id = ? AND revision = ?", id, revision);
	for (let offset = 0, ordinal = 0; offset < bytes.length; offset += SQLITE_PAYLOAD_CHUNK_BYTES, ordinal++) {
		await executor.run(
			"INSERT INTO payload_chunks (id, revision, ordinal, byte_length, data) VALUES (?, ?, ?, ?, ?)",
			id,
			revision,
			ordinal,
			bytes.length,
			bytes.subarray(offset, offset + SQLITE_PAYLOAD_CHUNK_BYTES),
		);
	}
}

export async function readJsonRow<T extends object>(
	executor: SqliteExecutor,
	selection: string,
	params: readonly SqliteValue[],
	options: { readonly revision?: boolean; readonly descending?: boolean } = {},
): Promise<T | undefined> {
	return (await readJsonRows<T>(executor, selection, params, options))[0];
}
