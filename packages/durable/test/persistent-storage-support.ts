import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { Id, Storage, StorageWrite } from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { context } from "./session-support.ts";

export type PersistentControlledStorage = Storage & {
	failNextCommit(error: unknown): void;
	rejectCommit(predicate: (writes: readonly StorageWrite[]) => unknown | undefined): void;
	rejectMint(predicate: () => unknown | undefined): void;
	readonly commits: readonly (readonly StorageWrite[])[];
};

/** Fault injection over an actual file-backed backend: close and reads always reach SQLite. */
export async function createPersistentStorage(): Promise<{
	readonly storage: PersistentControlledStorage;
	reopen(): Promise<PersistentControlledStorage>;
	close(): Promise<void>;
}> {
	const root = join(
		process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"),
		"vibestudio-performance",
		"pi-storage-tests",
	);
	await mkdir(root, { recursive: true, mode: 0o700 });
	const directory = await mkdtemp(join(root, "session-"));
	const path = join(directory, "session.sqlite");
	const opened = new Set<Storage>();
	const reopen = async (): Promise<PersistentControlledStorage> => {
		const backend = await openNodeSqliteStorage(path);
		opened.add(backend);
		const commit = backend.commit.bind(backend);
		const mint = backend.mintId.bind(backend);
		const commits: (readonly StorageWrite[])[] = [];
		let next: { error: unknown } | undefined;
		let rejectCommit: ((writes: readonly StorageWrite[]) => unknown | undefined) | undefined;
		let rejectMint: (() => unknown | undefined) | undefined;
		backend.commit = async (writes, ctx) => {
			commits.push(structuredClone(writes));
			const pending = next;
			next = undefined;
			if (pending !== undefined) throw pending.error;
			const error = rejectCommit?.(writes);
			if (error !== undefined) throw error;
			return commit(writes, ctx);
		};
		backend.mintId = async <I extends Id<string>>() => {
			const error = rejectMint?.();
			if (error !== undefined) throw error;
			return mint<I>();
		};
		return Object.assign(backend, {
			commits,
			failNextCommit: (error: unknown) => {
				next = { error };
			},
			rejectCommit: (predicate: (writes: readonly StorageWrite[]) => unknown | undefined) => {
				rejectCommit = predicate;
			},
			rejectMint: (predicate: () => unknown | undefined) => {
				rejectMint = predicate;
			},
		});
	};
	try {
		const storage = await reopen();
		return {
			storage,
			reopen,
			close: async () => {
				const results = await Promise.allSettled([...opened].map((storage) => storage.close(context)));
				const failures = results.flatMap((result) => (result.status === "rejected" ? [result.reason] : []));
				if (failures.length > 0) throw new AggregateError(failures, "Persistent test storage cleanup failed");
				await rm(directory, { recursive: true, force: true });
				opened.clear();
			},
		};
	} catch (error) {
		await rm(directory, { recursive: true, force: true });
		throw error;
	}
}
