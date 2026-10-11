import type { TaskId } from "../types.ts";

/** One bounded, replaceable wake candidate per task. Mutated only on the Session line. */
export class WakeIndex {
	readonly #heap: { id: TaskId; at: number }[] = [];
	readonly #positions = new Map<TaskId, number>();

	set(id: TaskId, at: number | null): void {
		this.delete(id);
		if (at === null) return;
		this.#positions.set(id, this.#heap.length);
		this.#heap.push({ id, at });
		this.#up(this.#heap.length - 1);
	}

	delete(id: TaskId): void {
		const position = this.#positions.get(id);
		if (position === undefined) return;
		this.#positions.delete(id);
		const last = this.#heap.pop()!;
		if (position === this.#heap.length) return;
		this.#heap[position] = last;
		this.#positions.set(last.id, position);
		this.#down(this.#up(position));
	}

	/** Due nodes only: a future node excludes its entire heap subtree. */
	ready(now: number): TaskId[] {
		const ready: TaskId[] = [];
		const pending = [0];
		while (pending.length > 0) {
			const position = pending.pop()!;
			const value = this.#heap[position];
			if (value === undefined || value.at > now) continue;
			ready.push(value.id);
			pending.push(position * 2 + 1, position * 2 + 2);
		}
		return ready;
	}

	/** The candidate overlay replaces these tasks; temporarily remove them and restore before returning. */
	minimumWithout(ids: ReadonlySet<TaskId>): number | null {
		const removed: { id: TaskId; at: number }[] = [];
		try {
			for (const id of ids) {
				const position = this.#positions.get(id);
				if (position === undefined) continue;
				removed.push(this.#heap[position]!);
				this.delete(id);
			}
			return this.#heap[0]?.at ?? null;
		} finally {
			for (const { id, at } of removed) this.set(id, at);
		}
	}

	#swap(a: number, b: number): void {
		const value = this.#heap[a]!;
		this.#heap[a] = this.#heap[b]!;
		this.#heap[b] = value;
		this.#positions.set(this.#heap[a]!.id, a);
		this.#positions.set(value.id, b);
	}

	#up(start: number): number {
		let position = start;
		while (position > 0) {
			const parent = Math.floor((position - 1) / 2);
			if (this.#heap[parent]!.at <= this.#heap[position]!.at) break;
			this.#swap(parent, position);
			position = parent;
		}
		return position;
	}

	#down(start: number): void {
		let position = start;
		for (;;) {
			const left = position * 2 + 1;
			if (left >= this.#heap.length) return;
			const right = left + 1;
			const child = right < this.#heap.length && this.#heap[right]!.at < this.#heap[left]!.at ? right : left;
			if (this.#heap[position]!.at <= this.#heap[child]!.at) return;
			this.#swap(position, child);
			position = child;
		}
	}
}
