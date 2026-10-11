import { describe, expect, it } from "vitest";
import { WakeIndex } from "../src/harness/wake-index.ts";
import type { TaskId } from "../src/types.ts";

describe("bounded task wake projection", () => {
	it("preserves the earliest deadline through replacements, withdrawals and temporary candidate overlays", () => {
		const index = new WakeIndex();
		const values = new Map<TaskId, number>();
		let random = 17;
		const next = () => {
			random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
			return random;
		};
		const minimum = (excluded: ReadonlySet<TaskId>) => {
			let result: number | null = null;
			for (const [id, at] of values) if (!excluded.has(id) && (result === null || at < result)) result = at;
			return result;
		};
		for (let operation = 0; operation < 2_000; operation++) {
			const id = (next() % 97) as TaskId;
			const at = next() % 5 === 0 ? null : next() % 10_000;
			index.set(id, at);
			if (at === null) values.delete(id);
			else values.set(id, at);
			const excluded = new Set<TaskId>([(next() % 97) as TaskId, id]);
			expect(index.minimumWithout(excluded)).toBe(minimum(excluded));
			// A discarded transaction overlay must leave the committed projection intact.
			expect(index.minimumWithout(new Set())).toBe(minimum(new Set()));
		}
		for (const id of values.keys()) index.delete(id);
		expect(index.minimumWithout(new Set())).toBeNull();
	});
});
