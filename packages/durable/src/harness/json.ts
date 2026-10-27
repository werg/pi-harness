import type { JsonRepresentation, JsonValue } from "@earendil-works/chord";
import { appendText, type Draft } from "@earendil-works/chord/delta";
import { copyJson, isJsonValue } from "@earendil-works/chord/json";
import type { Api, Message, Model } from "@earendil-works/pi-ai";
import { errorMessage } from "../errors.ts";
import type { TaskOutcomeError } from "../types.ts";

/** Detach original error diagnostics for the canonical failure record and terminal outcome. */
export function snapshotError(error: unknown, ancestors = new Set<Error>()): TaskOutcomeError {
	const message = errorMessage(error);
	if (!(error instanceof Error)) return { message };
	if (ancestors.has(error)) return { message, detail: { circularCause: true } };
	ancestors.add(error);
	try {
		const detail: Record<string, JsonValue> = {};
		if (error instanceof AggregateError) detail.errors = error.errors.map((value: unknown) => errorMessage(value));
		if (error.cause !== undefined)
			detail.cause =
				error.cause instanceof Error
					? snapshotError(error.cause, ancestors)
					: isJsonValue(error.cause)
						? copyJson(error.cause)
						: errorMessage(error.cause);
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

/** How `assignJson()` writes one value instead of assigning it leaf by leaf. */
export type AssignPlan =
	/** The string grew by `text` since it was last assigned: append it, when the lengths agree. */
	| { readonly kind: "append"; readonly text: string }
	/** Leave the value as it is; the caller writes it. */
	| { readonly kind: "keep" }
	/** Plans for some members or items of a container. */
	| { readonly kind: "fields"; readonly fields: ReadonlyMap<string | number, AssignPlan> };

/**
 * Assign `value` at `target[key]` leaf by leaf. Chord records a container assignment as one full set and only emits an
 * append when a string leaf is reassigned with a longer string, so writing the partial whole would store and publish the
 * complete message on every flush. That append costs a compare of the whole string, though; where `plan` says what a
 * string member grew by, and the committed string plus that text has the new string's length, the text is appended
 * with `appendText()` instead.
 */
export function assignJson<T extends object, K extends keyof T & (string | number)>(
	target: T,
	key: K,
	value: T[K] & JsonValue,
	plan?: AssignPlan,
): void {
	if (plan?.kind === "keep") return;
	// Every target is a document container: JSON members or items, whatever its declared type names them.
	const slots = target as Record<string | number, JsonValue>;
	const current = Object.hasOwn(slots, key) ? slots[key] : undefined;
	const fields = plan?.kind === "fields" ? plan.fields : undefined;
	if (isRecord(current) && isRecord(value)) {
		for (const name of Object.keys(current)) if (!Object.hasOwn(value, name)) delete current[name];
		for (const [name, child] of Object.entries(value))
			assignJson(current as Record<string, JsonValue>, name, child, fields?.get(name));
		return;
	}
	if (Array.isArray(current) && Array.isArray(value) && current.length <= value.length) {
		const items = current as JsonValue[];
		for (let index = 0; index < value.length; index++) {
			if (index < items.length) assignJson(items, index, value[index]!, fields?.get(index));
			else items.push(value[index]!);
		}
		return;
	}
	if (
		plan?.kind === "append" &&
		typeof current === "string" &&
		typeof value === "string" &&
		!Array.isArray(target) &&
		current.length + plan.text.length === value.length
	) {
		// The cast names what the checks above established: `target[key]` is a string.
		appendText(target as Draft<Record<string, string>>, String(key), plan.text);
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
