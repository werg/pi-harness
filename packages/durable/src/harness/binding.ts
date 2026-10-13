import type { JsonValue } from "@earendil-works/chord";
import { copyJson } from "@earendil-works/chord/json";
import type { ToolBinding, ToolExecutionMode, ToolRegistration } from "./types.ts";

/** A version promises unchanged execution, cancellation, hooks and external-operation semantics. */
export function bindTool(tool: ToolRegistration, mode: ToolExecutionMode): ToolBinding {
	const version = tool.version ?? 1;
	if (!Number.isSafeInteger(version) || version < 1) throw new Error(`Invalid version for tool ${tool.name}`);
	const contract = copyJson(
		{
			name: tool.name,
			version,
			// Bind the JSON schema sent to the provider, excluding TypeBox symbol metadata.
			parameters: JSON.parse(JSON.stringify(tool.parameters)) as JsonValue,
			structuredOutputSchema:
				tool.structuredOutputSchema === undefined
					? null
					: (JSON.parse(JSON.stringify(tool.structuredOutputSchema)) as JsonValue),
			callers: tool.callers ?? null,
			replay: tool.replay ?? "unsafe",
			cancel: tool.cancel !== undefined,
			executionMode: tool.executionMode ?? null,
			outputLimits: tool.outputLimits ?? null,
		},
		{ omitUndefinedProperties: true },
	);
	return {
		name: tool.name,
		version,
		signature: canonical(contract),
		executionMode: tool.executionMode ?? mode,
		...(tool.executionData === undefined ? {} : { data: copyJson(tool.executionData) }),
	};
}

function canonical(value: JsonValue): string {
	if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
	if (value !== null && typeof value === "object")
		return `{${Object.keys(value)
			.sort()
			.map((key) => `${JSON.stringify(key)}:${canonical(value[key]!)}`)
			.join(",")}}`;
	return JSON.stringify(value);
}

export function toolMatches(tool: ToolRegistration | undefined, binding: ToolBinding): tool is ToolRegistration {
	return tool !== undefined && bindTool(tool, binding.executionMode).signature === binding.signature;
}
