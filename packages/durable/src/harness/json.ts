import { copyJson, isJsonValue, type JsonRepresentation, type JsonValue } from "@earendil-works/chord";
import type { Api, Message, Model } from "@earendil-works/pi-ai";
import type { TaskOutcomeError } from "../types.ts";

/** Detach original error diagnostics for the canonical failure record and terminal outcome. */
export function snapshotError(error: unknown, ancestors = new Set<Error>()): TaskOutcomeError {
	const message = error instanceof Error ? error.message : String(error);
	if (!(error instanceof Error)) return { message };
	if (ancestors.has(error)) return { message, detail: { circularCause: true } };
	ancestors.add(error);
	try {
		const detail: Record<string, JsonValue> = {};
		if (error instanceof AggregateError)
			detail.errors = error.errors.map((value: unknown) => (value instanceof Error ? value.message : String(value)));
		if (error.cause !== undefined)
			detail.cause =
				error.cause instanceof Error
					? snapshotError(error.cause, ancestors)
					: isJsonValue(error.cause)
						? copyJson(error.cause)
						: String(error.cause);
		return { message, ...(Object.keys(detail).length === 0 ? {} : { detail }) };
	} finally {
		ancestors.delete(error);
	}
}

/** Complete catalog descriptor selected before transport or credential acquisition. */
export type PinnedModel = JsonRepresentation<Model<Api>>;

/** Actual provider input, including request-hook replacements and opaque replay signatures. */
export type PinnedMessages = JsonRepresentation<Message[]>;

/** Validate and detach checkpoint/port data without losing its declared JSON shape. */
export function pinJson<T>(value: T): JsonRepresentation<T> {
	return copyJson(value, { omitUndefinedProperties: true }) as JsonRepresentation<T>;
}

export function pinModel(model: Model<Api>): PinnedModel {
	return pinJson(model);
}

export function pinMessages(messages: readonly Message[]): PinnedMessages {
	return pinJson(messages);
}

type JsonContainer = Record<string, JsonValue> | JsonValue[];

/**
 * Assign `value` at `target[key]` leaf by leaf. Chord records a container assignment as one full set and only emits an
 * append when a string leaf is reassigned with a longer string, so writing the partial whole would store and publish the
 * complete message on every flush.
 */
export function assignJson(target: JsonContainer, key: string | number, value: JsonValue): void {
	const slots = target as Record<string | number, JsonValue>;
	const current = slots[key];
	if (isRecord(current) && isRecord(value)) {
		for (const name of Object.keys(current)) if (!Object.hasOwn(value, name)) delete current[name];
		for (const [name, child] of Object.entries(value)) assignJson(current, name, child);
		return;
	}
	if (Array.isArray(current) && Array.isArray(value) && current.length <= value.length) {
		const items = current as JsonValue[];
		for (let index = 0; index < value.length; index++) {
			if (index < items.length) assignJson(items, index, value[index]!);
			else items.push(value[index]!);
		}
		return;
	}
	if (current !== value) slots[key] = value;
}

function isRecord(value: JsonValue | undefined): value is Record<string, JsonValue> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Structural equality of two JSON values; object key order is ignored. */
export function jsonEqual(left: JsonValue | undefined, right: JsonValue | undefined): boolean {
	if (left === right) return true;
	if (typeof left !== "object" || typeof right !== "object" || left === null || right === null) return false;
	if (Array.isArray(left) || Array.isArray(right)) {
		if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
		return left.every((value, index) => jsonEqual(value, right[index]));
	}
	const keys = Object.keys(left);
	if (keys.length !== Object.keys(right).length) return false;
	return keys.every((key) => Object.hasOwn(right, key) && jsonEqual(left[key], right[key]));
}
