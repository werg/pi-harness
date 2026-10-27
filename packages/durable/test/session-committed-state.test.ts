import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { describe, expect, it } from "vitest";
import { defineDoc } from "../src/documents.ts";
import { SessionImpl } from "../src/session/session.ts";
import { MemoryStorage } from "../src/storage/memory.ts";
import type { Seq, StorageWrite } from "../src/types.ts";

const context = BACKGROUND_CONTEXT;
const doc = defineDoc<{ count: number }>({
	kind: "test.committed-state",
	version: 1,
	scope: "session",
	initial: () => ({ count: 0 }),
	checkpointWhen: () => true,
});

class GatedStorage extends MemoryStorage {
	gate: { entered: () => void; wait: Promise<void> } | undefined;
	failure: Error | undefined;
	override async commit(
		writes: readonly StorageWrite[],
		adopt: (seq: Seq) => void,
		ctx: typeof context,
	): Promise<Seq> {
		if (this.gate) {
			this.gate.entered();
			await this.gate.wait;
		}
		if (this.failure) throw this.failure;
		return super.commit(writes, adopt, ctx);
	}
}

describe("committed state publication", () => {
	it("does not expose a staged document while Storage is held", async () => {
		const storage = new GatedStorage();
		const session = new SessionImpl(storage);
		await session.commit(async (tx) => {
			await tx.doc(doc);
		}, context);
		const state = (await session.documentState(doc, context))!;
		const watch = (await session.watchDoc(doc, context))!;
		const delivered = Promise.withResolvers<number>();
		watch.start(async (change) => delivered.resolve(change.value.count));
		const entered = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		storage.gate = { entered: entered.resolve, wait: release.promise };
		const change = session.commit(async (tx) => {
			(await tx.doc(doc)).count = 1;
		}, context);
		try {
			await entered.promise;
			expect(state.value).toEqual({ count: 0 });
			expect(watch.value).toEqual({ count: 0 });
			release.resolve();
			await change;
			expect(state.value).toEqual({ count: 1 });
			expect(await delivered.promise).toBe(1);
		} finally {
			release.resolve();
			await change;
			state.dispose();
			await watch.stop();
			await session.close(context);
		}
	});

	it("propagates a Storage failure without publishing its staged value", async () => {
		const storage = new GatedStorage();
		const session = new SessionImpl(storage);
		await session.commit(async (tx) => {
			await tx.doc(doc);
		}, context);
		const state = (await session.documentState(doc, context))!;
		const failure = new Error("storage failed");
		storage.failure = failure;
		await expect(
			session.commit(async (tx) => {
				(await tx.doc(doc)).count = 1;
			}, context),
		).rejects.toBe(failure);
		expect(state.value).toEqual({ count: 0 });
		expect(await session.closed).toEqual({ reason: "failed", error: failure });
		state.dispose();
	});
});
