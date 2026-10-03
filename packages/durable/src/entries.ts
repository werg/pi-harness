import type { JsonValue } from "@earendil-works/chord";
import type { CompactionReason, ToolDiagnostic } from "./harness/types.ts";
import type { Entry, EntryRecord, JsonObject, TypedEntry } from "./types.ts";

/** Define a typed entry kind whose `is()` guard narrows by `EntryRecord.kind`. */
export function defineEntry<D extends JsonValue = never>(kind: string): Entry<D> {
	if (typeof kind !== "string" || kind.length === 0) throw new TypeError("Entry kind must be a non-empty string");
	return {
		kind,
		is: (entry: EntryRecord | undefined): entry is TypedEntry<D> => entry !== undefined && entry.kind === kind,
	};
}

/** User input: `model` is `[UserMessage]`. Written by submissions. */
export const UserEntry = defineEntry("pi.user");
/** Provider result with any stop reason: `model` is `[AssistantMessage]`. Written by generation. */
export const AssistantEntry = defineEntry("pi.assistant");
/** A real caller-selected tool invocation, with no provider/model message. */
export const DirectToolCallEntry = defineEntry<{ call: { id: string; name: string; arguments: JsonObject } }>(
	"pi.direct-tool-call",
);
/** Original direct source and complete tool result; never an orphan provider tool-result message. */
export const DirectToolResultEntry = defineEntry<{
	sourceEntryId: number;
	callId: string;
	name: string;
	result: JsonValue;
}>("pi.direct-tool-result");
/** Positional prompt and tool change: `model` is `[SystemMessage]` with empty `content`. */
export const SystemEntry = defineEntry("pi.system");
/**
 * Tool result: `model` is `[ToolResultMessage]`, whose content ends with the rendered diagnostics block; `data` holds
 * the structured diagnostics, possibly none. Written by tool tasks, and by generation for calls it did not offer.
 */
export const ToolResultEntry = defineEntry<{ diagnostics: ToolDiagnostic[] }>("pi.tool-result");
/**
 * Start of a new context: always `head: "self"`, with `model` absent for a plain reset or `[UserMessage]` carrying the
 * handoff text. Written by `Conversation.reset()` and the `handoff` tool control.
 */
export const ResetEntry = defineEntry("pi.reset");
/**
 * Compaction summary: `model` is `[UserMessage]` with the wrapped summary, `head` the first kept entry. Written by
 * compaction tasks, directly or through a write submission.
 */
export const CompactionEntry = defineEntry<{ reason: CompactionReason }>("pi.compaction");
