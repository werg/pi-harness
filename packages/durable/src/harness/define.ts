import { copyJson, type JsonValue } from "@earendil-works/chord";
import type { TSchema } from "@earendil-works/pi-ai";
import type {
	AnyTask,
	Extension,
	HookRegistration,
	HooksOf,
	PromptSection,
	ToolBinding,
	ToolExecutionMode,
	ToolRegistration,
	Wrap,
} from "./types.ts";

/** Identity function that types an extension. */
export function defineExtension<Tool extends ToolRegistration = ToolRegistration>(
	extension: Extension<Tool>,
): Extension<Tool> {
	return extension;
}

/** Identity function that types a tool: `args` from `parameters`, details from what it reports. */
export function defineTool<TParameters extends TSchema, TDetails extends JsonValue = JsonValue>(
	tool: ToolRegistration<TParameters, TDetails>,
): ToolRegistration<TParameters, TDetails> {
	return tool;
}

/** A prompt section; tagged unless `tag` is false. */
export function section<Tool extends ToolRegistration = ToolRegistration>(
	key: string,
	render: PromptSection<Tool>["render"],
	options?: { readonly tag?: boolean },
): PromptSection<Tool> {
	return options?.tag === undefined ? { key, render } : { key, render, tag: options.tag };
}

/** Hook handlers for tasks with `task`'s name. */
export function hook<K extends AnyTask>(task: K, handlers: Partial<HooksOf<K>>): HookRegistration {
	return { task: task.definition.name, handlers };
}

/** Wrap the tool named like `tool` wherever the wrapping extension is selected. */
export function wrapTool<Tool extends ToolRegistration>(tool: Tool, wrapper: (tool: Tool) => Tool): Wrap<Tool> {
	return { tool: tool.name, wrap: wrapper };
}

/** Wrap the section `key` wherever the wrapping extension is selected. */
export function wrapSection<Tool extends ToolRegistration = ToolRegistration>(
	key: string,
	wrapper: (section: PromptSection<Tool>) => PromptSection<Tool>,
): Wrap<Tool> {
	return { section: key, wrap: wrapper };
}

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
