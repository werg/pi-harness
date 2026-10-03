import { copyJson, type JsonValue } from "@earendil-works/chord";
import { defineDocFamily } from "../documents.ts";
import type { JsonObject, Tx } from "../types.ts";
import { jsonEqual, type PinnedModel, pinJson } from "./json.ts";
import type { ModelRequestTarget } from "./types.ts";

/** One task request round; deferred observations share its prepared endpoint. */
const ModelRequestDoc = defineDocFamily<{ intent: JsonValue; operations: JsonObject; model: PinnedModel | null }, null>(
	{
		kind: "pi.model-request",
		version: 1,
		scope: "task",
		family: true,
		initial: () => ({ intent: null, operations: {}, model: null }),
		checkpointWhen: () => true,
	},
);

/** Persist the original scheduler intent before any preparation or authority work. */
export async function admitModelRequest(tx: Tx, target: ModelRequestTarget): Promise<PinnedModel | undefined> {
	const key = JSON.stringify([target.purpose, target.attempt, target.cutoff]);
	const retained = await tx.doc(ModelRequestDoc, target.taskId, key, null);
	const { operation, options: _options, handle: _handle, ...intent } = target;
	const operationInput = copyJson(
		{ options: target.options, handle: target.handle },
		{ omitUndefinedProperties: true },
	);
	if (retained.intent !== null && !jsonEqual(retained.intent, intent))
		throw new Error("Model request conflicts with its original intent");
	if (Object.hasOwn(retained.operations, operation) && !jsonEqual(retained.operations[operation], operationInput))
		throw new Error("Model operation conflicts with its original request");
	retained.intent = intent;
	retained.operations[operation] = operationInput;
	return retained.model === null ? undefined : pinJson(retained.model);
}

/** Endpoint resolution may change baseUrl only; every semantic model field remains pinned. */
export async function prepareModelRequest(
	tx: Tx,
	target: ModelRequestTarget,
	model: PinnedModel,
): Promise<PinnedModel> {
	const { baseUrl: _originalUrl, ...original } = target.model;
	const { baseUrl: _preparedUrl, ...prepared } = model;
	if (!jsonEqual(original, prepared)) throw new Error("Model preparation cannot change the selected model");
	await admitModelRequest(tx, target);
	const key = JSON.stringify([target.purpose, target.attempt, target.cutoff]);
	const retained = await tx.doc(ModelRequestDoc, target.taskId, key, null);
	if (retained.model !== null && !jsonEqual(retained.model, model))
		throw new Error("Model preparation conflicts with its committed endpoint");
	retained.model = model;
	return pinJson(retained.model);
}
