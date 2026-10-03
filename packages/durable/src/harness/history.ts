import { type Context, copyJson, type JsonValue } from "@earendil-works/chord";
import { materializeDocumentValue } from "../documents.ts";
import { SystemEntry } from "../entries.ts";
import type { SessionImpl } from "../session/session.ts";
import type { ContextEdit, ConversationId, Cursor, EntryId, EntryRecord, Storage, Tx } from "../types.ts";
import { AgentDoc } from "./agent.ts";
import type {
	AgentState,
	ConversationHistory,
	ConversationHistoryEntry,
	ConversationHistoryEntryMap,
} from "./types.ts";

const SCAN_PAGE_SIZE = 256;
const HISTORY_KIND = "pi.history";

/** Capture only transcript knowledge and its agent settings, never source executable state or arbitrary documents. */
export function exportConversationHistory(
	session: SessionImpl,
	storage: Storage,
	conversationId: ConversationId,
	at: EntryId | null,
	context: Context,
): Promise<ConversationHistory> {
	return session.readOnLine(async () => {
		if ((await storage.conversation(conversationId, context)) === undefined) {
			throw new Error(`Conversation ${conversationId} does not exist`);
		}
		let agent: Readonly<AgentState> = {};
		const entries: EntryRecord[] = [];
		if (at === null) {
			agent = (await session.conversationDocumentOnLine(AgentDoc, conversationId, context))?.value ?? {};
		} else {
			const frontier = await storage.entry(conversationId, at, context);
			if (frontier === undefined) throw new Error(`Entry ${at} is not visible from conversation ${conversationId}`);
			// Inherited entries belong to their ancestor; match ordinary native fork as-of configuration semantics.
			const record = await storage.findDocument(
				{
					kind: AgentDoc.definition.kind,
					scope: { kind: "conversation", conversationId: frontier.entry.conversationId },
				},
				frontier.commitSeq,
				context,
			);
			if (record !== undefined) {
				const stored = await storage.document(record.id, frontier.commitSeq, context);
				if (stored === undefined) throw new Error(`Historical agent document ${record.id} cannot be read`);
				agent = materializeDocumentValue(AgentDoc.definition, record, stored.version, stored.value) as AgentState;
			}
			let cursor: Cursor | undefined;
			do {
				const page = await storage.scanEntries({ conversationId, maxEntryId: at }, SCAN_PAGE_SIZE, cursor, context);
				entries.push(...page.items);
				cursor = page.next;
			} while (cursor !== undefined);
			entries.reverse();
		}
		const selected = new Set(
			entries
				.filter((entry) => entry.model !== undefined || entry.head !== undefined || entry.edits !== undefined)
				.map((entry) => entry.id),
		);
		for (const entry of entries) {
			if (!selected.has(entry.id)) continue;
			if (entry.head !== undefined) selected.add(entry.head);
			for (const edit of entry.edits ?? []) selected.add(edit.target);
		}
		const history = {
			source: { conversationId, at },
			agent,
			entries: entries
				.filter((entry) => selected.has(entry.id))
				.map(
					(entry): ConversationHistoryEntry => ({
						id: entry.id,
						...(entry.kind === SystemEntry.kind ? { system: true } : {}),
						...(entry.model === undefined ? {} : { model: entry.model }),
						...(entry.head === undefined ? {} : { head: entry.head }),
						...(entry.edits === undefined ? {} : { edits: entry.edits }),
					}),
				),
		};
		return prepareConversationHistory(history);
	});
}

/** Detach and validate the knowledge graph before a receiving mutation is admitted. */
export function prepareConversationHistory(history: ConversationHistory): ConversationHistory {
	const value = copyJson(history, { omitUndefinedProperties: true }) as unknown as ConversationHistory;
	if (!Number.isSafeInteger(value.source.conversationId) || value.source.conversationId < 0) {
		throw new TypeError("History source requires a conversation ID");
	}
	const at = value.source.at;
	if (at !== null && (!Number.isSafeInteger(at) || at <= 0))
		throw new TypeError("History frontier requires an entry ID or null");
	if (!Array.isArray(value.entries)) throw new TypeError("History entries must be an array");
	if (at === null && value.entries.length !== 0) throw new Error("An empty history frontier cannot contain entries");
	const earlier = new Set<EntryId>();
	let previous = 0;
	for (const entry of value.entries) {
		if (!Number.isSafeInteger(entry.id) || entry.id <= previous || (at !== null && entry.id > at)) {
			throw new Error("History entries must be ordered unique IDs within the frontier");
		}
		if (entry.system !== undefined && entry.system !== true)
			throw new TypeError("History system marker must be true");
		if (entry.model !== undefined && !Array.isArray(entry.model))
			throw new TypeError("History model messages must be an array");
		if (entry.edits !== undefined && !Array.isArray(entry.edits))
			throw new TypeError("History edits must be an array");
		if (entry.head !== undefined && entry.head !== entry.id && !earlier.has(entry.head)) {
			throw new Error(`History head ${entry.head} is not an earlier transferred entry`);
		}
		for (const edit of entry.edits ?? []) {
			if (edit.action !== "omit" && edit.action !== "replace")
				throw new TypeError("History context edit action is invalid");
			if (edit.action === "replace" && !Array.isArray(edit.messages))
				throw new TypeError("History replacement messages must be an array");
			if (!earlier.has(edit.target))
				throw new Error(`History edit target ${edit.target} is not an earlier transferred entry`);
		}
		earlier.add(entry.id);
		previous = entry.id;
	}
	// Project fields explicitly even on imported data: arbitrary entry payloads and source control records cannot enter.
	return freezeJson(
		copyJson(
			{
				source: { conversationId: value.source.conversationId, at },
				agent: projectAgent(value.agent),
				entries: value.entries.map((entry: ConversationHistoryEntry) => ({
					id: entry.id,
					...(entry.system === true ? { system: true } : {}),
					...(entry.model === undefined ? {} : { model: entry.model }),
					...(entry.head === undefined ? {} : { head: entry.head }),
					...(entry.edits === undefined
						? {}
						: {
								edits: entry.edits.map((edit: ContextEdit) =>
									edit.action === "omit"
										? { target: edit.target, action: "omit" }
										: { target: edit.target, action: "replace", messages: edit.messages },
								),
							}),
				})),
			},
			{ omitUndefinedProperties: true },
		),
	) as unknown as ConversationHistory;
}

/** Apply knowledge to a new receiving conversation in its creating transaction; no source execution can be resumed. */
export async function importConversationHistory(
	tx: Tx,
	conversationId: ConversationId,
	history: ConversationHistory,
): Promise<ConversationHistoryEntryMap> {
	const agent = await tx.doc(AgentDoc, conversationId);
	for (const key of Object.keys(agent) as (keyof AgentState)[]) delete agent[key];
	Object.assign(agent, history.agent);
	const ids = new Map<EntryId, EntryId>();
	for (const entry of history.entries) {
		const head =
			entry.head === undefined ? {} : { head: entry.head === entry.id ? ("self" as const) : ids.get(entry.head)! };
		const edits =
			entry.edits === undefined
				? {}
				: { edits: entry.edits.map((edit) => ({ ...edit, target: ids.get(edit.target)! })) };
		const appended = await tx.appendEntry(conversationId, {
			kind: entry.system === true ? SystemEntry.kind : HISTORY_KIND,
			...(entry.model === undefined ? {} : { model: entry.model }),
			...head,
			...edits,
		});
		ids.set(entry.id, appended.id);
	}
	return Object.freeze(Object.fromEntries(ids)) as ConversationHistoryEntryMap;
}

/** Only agent configuration fields belong to transferred knowledge; executable state lives elsewhere. */
function projectAgent(agent: Readonly<AgentState>): AgentState {
	return {
		...(agent.model === undefined ? {} : { model: { provider: agent.model.provider, modelId: agent.model.modelId } }),
		...(agent.thinkingLevel === undefined ? {} : { thinkingLevel: agent.thinkingLevel }),
		...(agent.extensions === undefined
			? {}
			: {
					extensions: Array.isArray(agent.extensions)
						? agent.extensions
						: {
								...(agent.extensions.add === undefined ? {} : { add: agent.extensions.add }),
								...(agent.extensions.remove === undefined ? {} : { remove: agent.extensions.remove }),
							},
				}),
		...(agent.tools === undefined
			? {}
			: { tools: Array.isArray(agent.tools) ? agent.tools : { remove: agent.tools.remove } }),
		...(agent.instructions === undefined ? {} : { instructions: agent.instructions }),
		...(agent.cwd === undefined ? {} : { cwd: agent.cwd }),
	};
}

function freezeJson(value: JsonValue): JsonValue {
	if (value !== null && typeof value === "object") {
		for (const child of Object.values(value)) freezeJson(child);
		Object.freeze(value);
	}
	return value;
}
