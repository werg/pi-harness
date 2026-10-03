import type { SqliteDatabase, SqliteExecutor, SqliteValue } from "./database.ts";

/** Native SQLite storage without a dependency on Node or provider SDKs. */
export interface NativeStorage {
	sql: {
		exec(
			sql: string,
			...values: (null | string | number | ArrayBuffer)[]
		): { toArray(): Record<string, unknown>[]; one(): Record<string, unknown> };
	};
	transaction<T>(callback: () => Promise<T>): Promise<T>;
	sync(): Promise<void>;
}

function binding(value: SqliteValue): null | string | number | ArrayBuffer {
	if (typeof value === "bigint") {
		const number = Number(value);
		if (!Number.isSafeInteger(number)) throw new RangeError("Unsafe SQLite integer binding");
		return number;
	}
	if (value instanceof Uint8Array) return value.slice().buffer;
	return value;
}

/** Use only within the caller's native transaction; do not retain the executor beyond it. */
export function nativeSqliteExecutor(sql: NativeStorage["sql"], check: () => void): SqliteExecutor {
	const all = async <T extends object>(query: string, ...params: SqliteValue[]): Promise<T[]> => {
		check();
		return sql.exec(query, ...params.map(binding)).toArray() as T[];
	};
	return {
		all,
		get: async <T extends object>(query: string, ...params: SqliteValue[]) => (await all<T>(query, ...params))[0],
		exec: async (query) => {
			await all(query);
		},
		run: async (query, ...params) => {
			await all(query, ...params);
		},
	};
}

/** One serialized connection. Transaction handles execute directly; root calls queue. */
export class NativeDatabase implements SqliteDatabase {
	private tail: Promise<unknown> = Promise.resolve();
	private closed = false;
	readonly native: NativeStorage;

	constructor(native: NativeStorage) {
		this.native = native;
	}

	private enqueue<T>(operation: () => Promise<T>): Promise<T> {
		const result = this.tail.then(() => {
			if (this.closed) throw new Error("Native SQLite connection is closed");
			return operation();
		});
		this.tail = result.catch(() => {});
		return result;
	}

	private executor(check: () => void): SqliteExecutor {
		return nativeSqliteExecutor(this.native.sql, check);
	}

	exec(sql: string): Promise<void> {
		return this.enqueue(() => this.executor(() => {}).exec(sql));
	}
	run(sql: string, ...params: SqliteValue[]): Promise<void> {
		return this.enqueue(() => this.executor(() => {}).run(sql, ...params));
	}
	get<T extends object>(sql: string, ...params: SqliteValue[]): Promise<T | undefined> {
		return this.enqueue(() => this.executor(() => {}).get<T>(sql, ...params));
	}
	all<T extends object>(sql: string, ...params: SqliteValue[]): Promise<T[]> {
		return this.enqueue(() => this.executor(() => {}).all<T>(sql, ...params));
	}
	transaction<T>(callback: (transaction: SqliteExecutor) => Promise<T>): Promise<T> {
		return this.enqueue(async () => {
			let active = true;
			const handle = this.executor(() => {
				if (!active) throw new Error("SQLite transaction handle is no longer active");
			});
			const result = await this.native.transaction(async () => {
				try {
					return await callback(handle);
				} finally {
					active = false;
				}
			});
			// Explicit confirmation precedes portable Storage's adoption of IDs/publication.
			// Failure here is an ordinary uncertain error, never a guaranteed StorageRejected.
			await this.native.sync();
			return result;
		});
	}
	close(): Promise<void> {
		const result = this.tail.then(async () => {
			if (this.closed) return;
			this.closed = true;
			await this.native.sync();
			// The DO owns the database lifetime; a portable connection cannot close it.
		});
		this.tail = result.catch(() => {});
		return result;
	}
}
