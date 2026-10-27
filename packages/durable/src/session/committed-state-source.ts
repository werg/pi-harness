import type {
	Context,
	ReplicatedStateSource,
	ReplicatedStateSourceAttachment,
	ReplicatedStateSourceFrame,
} from "@earendil-works/chord";
import type { Op } from "@earendil-works/chord/delta";
import { replicatedState } from "@earendil-works/chord/state";
import type { JsonObject } from "../types.ts";

export type ObservedDocumentValue = Readonly<JsonObject> | null;

/** Bind a view to the same committed observer stream as its watches. */
export function committedStateObserver<T>(value: T, release: () => void, report: (error: Error) => void) {
	const source = new CommittedStateSource(value, release);
	return {
		state: replicatedState(source, { onError: report }),
		advance: (change: { readonly value: T; readonly ops: readonly Op[] }, context: Context) =>
			source.advance(change.value, change.ops, context),
		end: () => source.closeSession(),
	};
}

/** Canonical terminal update for a retired document incarnation. */
export const RETIREMENT_OPERATIONS: readonly Op[] = [["r", null]];

/**
 * Session-to-Chord bridge owned one-to-one by one attached state: a document, or a conversation view. A `null` value
 * retires it. @internal
 */
export class CommittedStateSource<T = ObservedDocumentValue> implements ReplicatedStateSource<T> {
	readonly #attachments = new Set<SessionSourceAttachment<T>>();
	#release: (() => void) | undefined;
	/** Released on disposal. */
	#value: T | undefined;
	#cursor = 0;
	#retired = false;
	#closed = false;

	constructor(value: T, release: () => void) {
		this.#value = value;
		this.#release = release;
	}

	attach(): ReplicatedStateSourceAttachment<T> {
		if (this.#closed) throw new Error("State source is closed");
		const attachment = new SessionSourceAttachment({ value: this.#value as T, cursor: this.#cursor }, () => {
			this.#attachments.delete(attachment);
			if (this.#attachments.size === 0) this.#finishDisposal();
		});
		this.#attachments.add(attachment);
		return attachment;
	}

	advance(value: T, ops: readonly Op[], context: Context): void {
		if (this.#closed || this.#retired) return;
		this.#value = value;
		this.#cursor += 1;
		if (value === null) this.#retired = true;
		const frame: ReplicatedStateSourceFrame<T> = {
			cursor: this.#cursor,
			value,
			ops,
			context,
		};
		for (const attachment of [...this.#attachments]) attachment.publish(frame);
	}

	/** Stop publishing; the attached states keep their last value. A failure ends them the same way. */
	closeSession(): void {
		if (this.#closed) return;
		for (const attachment of [...this.#attachments]) attachment.dispose();
		this.#finishDisposal();
	}

	#finishDisposal(): void {
		if (this.#closed) return;
		this.#closed = true;
		this.#value = undefined;
		const release = this.#release;
		this.#release = undefined;
		release?.();
	}
}

class SessionSourceAttachment<T> implements ReplicatedStateSourceAttachment<T> {
	readonly snapshot: { readonly value: T; readonly cursor: number };
	#release: (() => void) | undefined;
	readonly #frames: ReplicatedStateSourceFrame<T>[] = [];
	#listener: ((frame: ReplicatedStateSourceFrame<T>) => void) | undefined;
	#activated = false;
	#disposed = false;
	#scheduled = false;
	#delivering = false;

	constructor(snapshot: { readonly value: T; readonly cursor: number }, release: () => void) {
		this.snapshot = snapshot;
		this.#release = release;
	}

	activate(listener: (frame: ReplicatedStateSourceFrame<T>) => void): void {
		if (this.#activated) throw new Error("State attachment is already active");
		if (this.#disposed) throw new Error("State attachment is disposed");
		this.#activated = true;
		this.#listener = listener;
		this.#drain();
	}

	publish(frame: ReplicatedStateSourceFrame<T>): void {
		if (this.#disposed) return;
		this.#frames.push(frame);
		if (!this.#activated || this.#delivering || this.#scheduled) return;
		this.#scheduled = true;
		queueMicrotask(() => {
			this.#scheduled = false;
			if (this.#disposed) return;
			try {
				this.#drain();
			} catch {
				// Chord's frame listener contains source-contract failures. Isolate an
				// unexpected direct listener failure to this attachment as well.
				this.dispose();
			}
		});
	}

	dispose(): void {
		if (this.#disposed) return;
		this.#disposed = true;
		this.#frames.length = 0;
		this.#listener = undefined;
		const release = this.#release;
		this.#release = undefined;
		release?.();
	}

	#drain(): void {
		const listener = this.#listener;
		if (listener === undefined || this.#delivering || this.#disposed) return;
		this.#delivering = true;
		try {
			while (!this.#disposed) {
				const frame = this.#frames.shift();
				if (frame === undefined) break;
				listener(frame);
			}
		} finally {
			this.#delivering = false;
		}
	}
}
