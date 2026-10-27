import type { AttachedReplicatedState, Context, JsonValue, WatchHandle } from "@earendil-works/chord";
import type {
	AssistantMessage,
	CacheRetention,
	ClientIdentity,
	DeferredHandle,
	Message,
	Models,
	ModelsRequestTransforms,
	ModelThinkingLevel,
	OpenAICodexResponsesOptions,
	SimpleStreamOptions,
	Static,
	Tool,
	ToolCall,
	ToolResultMessage,
	Transport,
	TSchema,
	Usage,
	UserMessage,
} from "@earendil-works/pi-ai";
import type { TelemetryContext } from "@earendil-works/pi-telemetry";
import type { ExecutionEnv, ShellOutputSkip, ShellOutputWindow } from "../env/index.ts";
import type {
	CommitOptions,
	ConversationId,
	ConversationOwnership,
	ConversationRecord,
	Cursor,
	DocumentObserver,
	DocumentReader,
	EntryDraft,
	EntryId,
	EntryQuery,
	EntryRecord,
	JsonObject,
	Observation,
	Observer,
	Page,
	ReportInfo,
	Session,
	SetTimer,
	SubmissionId,
	SubmissionRecord,
	Task,
	TaskId,
	TaskOptions,
	TaskRecord,
	TaskState,
	TaskWaitCondition,
	Tx,
} from "../types.ts";
import type { PinnedMessages, PinnedModel } from "./json.ts";
import type { WakeSchedule } from "./live.ts";
import type { TaskGraph, TaskGraphWatch } from "./task-graph.ts";
import type { DirectToolCall, ToolTaskResult } from "./tool.ts";
import type { UsageState } from "./usage.ts";
import type { ConversationView } from "./view.ts";

/** Provider and model ID resolved through pi-ai `Models`. */
export type ModelRef = {
	readonly provider: string;
	readonly modelId: string;
};

export type UserInput = UserMessage["content"];

/**
 * The entry an input is placed as, instead of `pi.user`: an application kind, and data the transcript shows, such as its
 * author. Its model message is still the input's user message.
 */
export type InputEntry = { readonly kind: string; readonly data?: JsonValue };

/** Host submission: user input that may start a run, or a passive entry write. */
export type SubmissionDraft = {
	readonly requestId?: string;
} & (
	| {
			readonly type: "input";
			readonly content: UserInput | SubmissionPrepare<UserInput>;
			readonly whenBusy?: "steer" | "followUp" | "reject";
			/** Place the input as this entry; default `pi.user` without data. The kind must not start with `pi.`. */
			readonly entry?: InputEntry;
	  }
	| {
			readonly type: "write";
			readonly entry: EntryDraft | SubmissionPrepare<EntryDraft>;
			readonly content?: never;
			readonly whenBusy?: never;
	  }
);

export type InputSubmissionDraft = Extract<SubmissionDraft, { readonly type: "input" }>;

/**
 * Prepare a payload and local writes in the same admission commit, using the reserved submission identity. Runs only
 * for fresh admission, never for a request-ID replay or busy rejection. The payload follows the ordinary native
 * inbox/placement rules. A throw rolls back every native and local write. Document access remains available; table
 * reads throw `ReadAfterWrite`. `tx.createTask()` defaults to this conversation. Await every Tx operation; do not
 * call Session APIs or perform external effects here.
 */
export type SubmissionPrepare<T> = (tx: Tx, submissionId: SubmissionId) => T | Promise<T>;

export type SettledSubmissionRecord = SubmissionRecord & {
	readonly status: "done" | "unanswered";
};

/** Awaitable host object for one durably admitted submission. */
export interface Submission {
	readonly id: SubmissionId;
	status(context: Context): Promise<SubmissionRecord>;
	wait(context: Context): Promise<SettledSubmissionRecord>;
	abort(context: Context): Promise<"aborted" | "already_placed" | "settled">;
}

export type SettledTask<R> = TaskRecord<JsonValue, JsonValue, R> & {
	readonly state: Extract<TaskState<JsonValue, R>, { readonly status: "terminal" }>;
};

/** Erased executable task definition stored in the registry. */
export type AnyTask = {
	readonly definition: {
		readonly name: string;
		readonly version: number;
		readonly initial: unknown;
		readonly phases: Readonly<Record<string, unknown>>;
		readonly abort: unknown;
		readonly migrate?: unknown;
		readonly hooks?: object;
	};
};

/** Options of `Conversation.abort()`. */
export type ConversationAbortOptions = {
	/**
	 * Cross background boundaries: mark every live task reached ignoring the background flag when the abort is admitted,
	 * withdraw the queued inputs of every conversation reached, and wait until those tasks are terminal and the
	 * conversation is ordinarily idle. Background work created afterwards is neither marked nor awaited.
	 */
	readonly background?: boolean;
};

/** Hook handler map declared by a task definition. */
export type HooksOf<K> = K extends Task<infer _I, infer _S, infer _R, infer H> ? H : never;

/**
 * Invocation-bound conversation operations for tasks and tools. Rejects after the invocation ends; passive entries are
 * written with ordinary transaction writes instead.
 */
export interface ConversationHandle {
	readonly id: ConversationId;
	submit(submission: InputSubmissionDraft, context: Context): Promise<Submission>;
	/** `Conversation.abort()`: withdraw queued inputs, abort the ordinary ownership scope, and wait until it is idle. */
	abort(context: Context, options?: ConversationAbortOptions): Promise<void>;
	/** Resolve when the conversation's ordinary ownership scope has no live non-background task. */
	waitForIdle(context: Context): Promise<void>;
}

/** Post-tools controls requested by a tool result. */
export type ToolControl = {
	readonly addTools?: readonly string[];
	/** End the run without another model request when every result of the round sets it to `true`; `false` is a no-op. */
	readonly terminate?: boolean;
	/**
	 * Start a new context with this text as its user message, as `reset(handoff)` writes, and continue the run there.
	 * The last handoff in call order wins.
	 */
	readonly handoff?: string;
};

/** Remark about a call for the model and the UI, such as truncation or a spill path; never part of the tool's data. */
export type ToolDiagnostic = {
	readonly severity: "info" | "warn" | "error";
	readonly message: string;
	readonly code?: string;
};

/**
 * What a tool returns, on three channels: `output` for the model, `structuredOutput` for programs that call the tool
 * (`executeTool()` callers, code mode scripts), and `details` for UIs.
 */
export type ToolExecutionResult<TDetails extends JsonValue = JsonValue> = {
	/** For the model. Omitted: the retained `output()` text. */
	readonly output?: ToolResultMessage["content"];
	/**
	 * For programs. Required unless `isError` when the tool declares `structuredOutputSchema`, and validated against it;
	 * not allowed without one, where programs get the output itself (`NestedToolExecutionResult.structuredOutput`).
	 */
	readonly structuredOutput?: JsonValue;
	readonly isError?: boolean;
	/** Omitted: the last `details()` value becomes the details. */
	readonly details?: TDetails;
	/** Added after those recorded through `api.diagnostic()`. */
	readonly diagnostics?: readonly ToolDiagnostic[];
	/** Spend of the execution itself, such as a model call; stored on the result and in `pi.usage.tools`. */
	readonly usage?: Usage;
	readonly control?: ToolControl;
};

/** A committed external continuation; returning it releases this invocation. */
export type ToolExecutionWait = { readonly wait: TaskWaitCondition; readonly continuation: JsonValue };

/**
 * Result of a nested call, as `executeTool()` returns it. Programs read `structuredOutput`; the output for the model
 * is not kept, and `control` does not apply.
 */
export type NestedToolExecutionResult = {
	/** The nested call's tool task. */
	readonly taskId: TaskId;
	/**
	 * The tool's value for programs. A tool with `structuredOutputSchema`: its validated `structuredOutput`, also on an
	 * error result that has one; absent otherwise, and an error's text is then in an error diagnostic. A tool without:
	 * its bounded output, success or error, as a string for one text item, an `ImageContent` for one image, `""` for
	 * none, and the content list otherwise. Absent on an error result the Harness wrote itself (unavailable, invalid,
	 * blocked, interrupted, aborted); `diagnostics` say what went wrong.
	 */
	readonly structuredOutput?: JsonValue;
	readonly isError: boolean;
	readonly details?: JsonValue;
	readonly diagnostics: readonly ToolDiagnostic[];
	readonly usage?: Usage;
	/** Execution time of the attempt that produced the result; absent when it never executed. */
	readonly durationMs?: number;
};

/** Whether the tools of one round run at once or one after another in call order. */
export type ToolExecutionMode = "parallel" | "sequential";

/** How many queued items of one mode a boundary places: the first, or all of them. */
export type QueueMode = "all" | "one-at-a-time";

/**
 * Operations available to one tool invocation. A plain object, so a wrapper can spread it. Every operation rejects after
 * the invocation ends.
 */
export interface ToolExecutionApi<TDetails extends JsonValue = JsonValue> extends DocumentObserver, DocumentReader {
	readonly taskId: TaskId;
	readonly conversationId: ConversationId;
	readonly callId: string;
	/** Immutable JSON context captured from the original offered definition, independent of the current registry. Each read is detached. */
	readonly executionData: JsonValue | undefined;
	/** Committed continuation from the prior invocation, absent on first admission. */
	readonly continuation: JsonValue | undefined;
	/** The tool task's phase snapshot. */
	readonly registry: RegistrySnapshot;
	/** The calling conversation's agent, as the tool task's phase resolved it. */
	agent(context: Context): Promise<Agent>;
	/** The current `HarnessOptions.models`, read at each access: the catalog, credentials, and request transforms. */
	readonly models: Models;
	/** Built by `HarnessOptions.env` for this call; `undefined` without an environment. */
	readonly env: ExecutionEnv | undefined;
	/** The Harness clock (`HarnessOptions.now`), for times a rerun after a restart compares against. */
	now(): number;
	/**
	 * Append running output; it becomes the result's `output` when the result omits it. `skipped` counts output
	 * omitted before the chunk, as reported by an environment given `outputWindow` (`ShellOutputInfo.skipped`).
	 */
	output(chunk: string | Uint8Array, skipped?: ShellOutputSkip): void;
	/**
	 * The tail this call's output keeps and the pace of its progress commits, for `ShellExecOptions.window`; `undefined`
	 * when the tool keeps the head of its output, which cannot accept skips. A wrapper that replaces `output` and
	 * transforms text must also replace this with `undefined`, so skipped text cannot bypass its transform.
	 */
	readonly outputWindow: ShellOutputWindow | undefined;
	/**
	 * The output retained so far, as the model will see it when the result omits `output`, and whether earlier output was
	 * dropped; for a tool that also returns that text in its `structuredOutput`.
	 */
	retainedOutput(): { readonly text: string; readonly truncated: boolean };
	/** Record a model-visible remark about this call. */
	diagnostic(diagnostic: ToolDiagnostic): void;
	/** Replace running details; the last value becomes the result details when the result omits `details`. */
	details(value: TDetails, context: Context): Promise<void>;
	commit<T>(change: (tx: Tx) => T | Promise<T>, context: Context, options?: CommitOptions): Promise<T>;
	/** Commit an external operation's continuation and original binding before uncertain dispatch. Subsequent failures retain this task for exact repair or cancellation. */
	retainContinuation(
		continuation: JsonValue,
		change: (tx: Tx) => void | Promise<void>,
		context: Context,
	): Promise<void>;
	memo<T extends JsonValue>(name: string, context: Context): Promise<T | undefined>;
	memo<T extends JsonValue>(name: string, candidate: T, context: Context): Promise<T>;
	/**
	 * Create a task in one commit. A child task of a tool that is not replay-safe defaults to `abandonOnRestart`: the
	 * tool never resumes after a restart, so nothing awaits the child then.
	 */
	createTask<I, S extends { phase: string }, R, H extends object>(
		task: Task<I, S, R, H>,
		input: I,
		options: Omit<TaskOptions, "conversationId">,
		context: Context,
	): Promise<TaskId<R>>;
	getTask<R>(id: TaskId<R>, context: Context): Promise<TaskRecord<JsonValue, JsonValue, R> | undefined>;
	waitForTask<R>(id: TaskId<R>, context: Context): Promise<SettledTask<R>>;
	/** Invocation-bound handle of an existing conversation, such as one this tool created in `commit()`. */
	conversation(id: ConversationId, context: Context): Promise<ConversationHandle | undefined>;
	/**
	 * Run tool `name` as a nested call of this one and wait for its result. The nested call is its own `pi.tool` task,
	 * owned by this call: it resolves the tool among the conversation's tools, validates, runs the `ToolTask` hooks (which
	 * see `call.parent`), applies output limits and the tool's replay policy, and reports progress in `pi.live.nestedTools`.
	 * Its result returns here instead of entering the transcript. A call that is blocked, invalid, throws, is interrupted,
	 * or is aborted returns an `isError` result.
	 *
	 * The nested call's key names it within this call; its call ID is `<callId>/<key>`. By default the key is the call's
	 * position among this invocation's nested calls, `1`, `2`, ... An explicit `key` must be non-empty, without `/`, not
	 * `__proto__`, and not a positive integer, so it never collides with a default one. A rerun of a replay-safe tool
	 * that makes a call with a used key, by calling in the same order or by passing the same `key`, gets that nested call
	 * back, finished or still running, instead of starting another. A nested call a crash interrupted before its result
	 * committed follows its own replay policy, so this is not exactly-once execution. Reusing a key with another tool or
	 * other arguments throws. A tool that is not replay-safe never reruns: after a restart, its unfinished nested calls
	 * are abandoned (`TaskOptions.abandonOnRestart`) before they run again.
	 *
	 * `progress: false` commits no running output, details, or diagnostics to the nested call's slot, only its status;
	 * the result still carries its details and diagnostics, and a schema-less tool's output as `structuredOutput`. An
	 * interrupted or aborted call's result is built from its slot, so it then has none of them. Either way, programs never
	 * get the partial output of an interrupted or aborted call.
	 *
	 * Nested calls run as soon as they are made, whatever their `executionMode`; the caller decides what runs at once by
	 * what it awaits. Cancelling `context` stops only the wait; the nested call runs on. When this call settles, nested
	 * calls it left running are aborted first. A nested result's `usage` is already counted under its tool; a result
	 * that passes it on counts it again. Rejects, instead of returning a result, once `execute()` has returned, when this
	 * call is aborted, or when the Harness closes.
	 */
	executeTool(
		name: string,
		args: JsonObject,
		context: Context,
		options?: { readonly key?: string; readonly progress?: boolean },
	): Promise<NestedToolExecutionResult>;
}

/** Executable contract offered by one prepared request; independent of later agent settings. */
export type ToolBinding = {
	name: string;
	version: number;
	signature: string;
	executionMode: ToolExecutionMode;
	/** Original offered operation context; separate from the executable's semantic signature. */
	data?: JsonValue;
};

/**
 * Executable tool registered in a registry. Only pi-ai `Tool` fields enter the transcript. `args` are typed by
 * `parameters`, which the Harness validates them against before `execute()`; `defineTool()` infers both generics.
 */
export type ToolRegistration<
	TParameters extends TSchema = TSchema,
	TDetails extends JsonValue = JsonValue,
> = Tool<TParameters> & {
	/** Positive semantic version; default 1. Bump when execution/cancellation/hook semantics change. */
	readonly version?: number;
	/** JSON operation context captured when this definition is offered or directly bound, before execution. */
	readonly executionData?: JsonValue;
	/** Whether an interrupted execution may rerun on recovery. Default `unsafe`. */
	readonly replay?: "safe" | "unsafe";
	/** Default: the settings' `toolExecution`. A sequential call is a barrier between parallel-safe waves. */
	readonly executionMode?: ToolExecutionMode;
	/**
	 * Repair arguments models commonly get wrong before validation, such as a JSON string where an array belongs. Must be
	 * pure and must not mutate `args`: it runs again when a call is retried before its intent is recorded. Its result is
	 * still validated against `parameters`.
	 */
	prepareArguments?(args: unknown): Static<TParameters>;
	/**
	 * Schema of `structuredOutput`, which programs that call the tool receive. Omitted: they receive the result's bounded
	 * output, success or error, as `NestedToolExecutionResult.structuredOutput` describes.
	 */
	readonly structuredOutputSchema?: TSchema;
	/**
	 * Who may call the tool: the model, other tools through `executeTool()`, or both (default). A code mode tool is
	 * `["model"]`, so scripts cannot start scripts; tools behind code mode, such as MCP tools, can be `["tools"]`, so the
	 * model is never offered them.
	 */
	readonly callers?: readonly ("model" | "tools")[];
	/**
	 * Not offered to the model until loaded: the model gets the tool only when `modelTools` names it, in its array or its
	 * `add`, which `control.addTools` (from a tool such as `tool_search`) writes. Other tools can call it any time.
	 */
	readonly deferred?: true;
	readonly outputLimits?: {
		readonly maxBytes?: number;
		readonly maxLines?: number;
		readonly retain?: "head" | "tail";
	};
	execute(
		args: Static<TParameters>,
		api: ToolExecutionApi<TDetails>,
		context: Context,
	): Promise<ToolExecutionResult<TDetails> | ToolExecutionWait>;
	/** External tools cancel/attach by their immutable identity; cleanup may itself park. */
	cancel?(
		args: Static<TParameters>,
		api: ToolExecutionApi<TDetails>,
		context: Context,
	): Promise<ToolExecutionResult<TDetails> | ToolExecutionWait>;
};

/** Input to system prompt section rendering for one request preparation. */
export type PromptInput<Tool extends ToolRegistration = ToolRegistration> = {
	readonly conversationId: ConversationId;
	/** The request's resolution; `agent.tools` are the tools offered in this request. */
	readonly agent: Agent<Tool>;
	/** Built by `HarnessOptions.env` for this preparation; `undefined` without an environment. */
	readonly env: ExecutionEnv | undefined;
	/** Sections already in effect after replaying the active transcript. */
	readonly shown: Readonly<Record<string, string>>;
	/** Committed document reads. */
	readonly read: DocumentReader;
};

/** One system prompt section; the agent's sections render in order before each request. */
export type PromptSection<Tool extends ToolRegistration = ToolRegistration> = {
	readonly key: string;
	render(input: PromptInput<Tool>, context: Context): string | undefined | Promise<string | undefined>;
	/** Default true: wrap the text as `<key>\n...\n</key>`. */
	readonly tag?: boolean;
};

/** Built by `hook()`; matches tasks by name. */
export type HookRegistration = { readonly task: string; readonly handlers: object };

/** Built by `wrapTool()` and `wrapSection()`; targets a tool name or a section key. Wrappers are pure. */
export type Wrap<Tool extends ToolRegistration = ToolRegistration> =
	| { readonly tool: string; wrap(tool: Tool): Tool }
	| { readonly section: string; wrap(section: PromptSection<Tool>): PromptSection<Tool> };

/** Named bundle of code; installed in a registry and selected by conversations by name. */
export interface Extension<Tool extends ToolRegistration = ToolRegistration> {
	readonly name: string;
	readonly tools?: readonly Tool[];
	readonly sections?: readonly PromptSection<Tool>[];
	readonly hooks?: readonly HookRegistration[];
	/** Apply where this extension is selected, in order. */
	readonly wraps?: readonly Wrap<Tool>[];
	/** Resolved by name for every task, whichever conversations select this extension. */
	readonly tasks?: readonly AnyTask[];
}

/** Immutable view of one published registry state. */
export interface RegistrySnapshot<Tool extends ToolRegistration = ToolRegistration> {
	/** Activation-local publication identity; a different activation always rechecks parked bindings. */
	readonly revision: string;
	installed(): readonly Extension<Tool>[];
	extension(name: string): Extension<Tool> | undefined;
	/** Every installed tool with its extension, in install order. Names may repeat across extensions. */
	tools(): readonly { readonly extension: Extension<Tool>; readonly tool: Tool }[];
	sections(): readonly { readonly extension: Extension<Tool>; readonly section: PromptSection<Tool> }[];
	/** Built-in and installed task definitions. */
	tasks(): readonly AnyTask[];
	task(name: string): AnyTask | undefined;
}

/** Read side of a registry consumed by a Harness. */
export interface RegistryReader<Tool extends ToolRegistration = ToolRegistration> {
	/** Immutable view of the whole current registry. */
	snapshot(): RegistrySnapshot<Tool>;
	/** Called synchronously after every publication; wakes the scheduler to reconsider blocked tasks. */
	subscribe(listener: () => void): () => void;
}

/** Application-owned registry of extensions. */
export interface Registry<Tool extends ToolRegistration = ToolRegistration> extends RegistryReader<Tool> {
	/** Install `extension`, or replace the installed extension with its name in place. Publishes at once. */
	install(extension: Extension<Tool>): void;
	/** Remove the installed extension with `extension.name`, whichever object it is. A later install appends. */
	uninstall(extension: Extension): void;
}

/** Stored choices of one conversation; names, not objects. Unset fields follow the host. */
export type AgentState = {
	model?: ModelRef;
	thinkingLevel?: ModelThinkingLevel;
	stream?: ConversationStreamOptions;
	/** An array selects exactly these extensions, in order. An object edits the host default selection. */
	extensions?: string[] | { add?: string[]; remove?: string[] };
	/** Filters the selected extensions' tools: the enabled tools. An array enables exactly these, in order. */
	tools?: string[] | { remove: string[] };
	/**
	 * Filters the enabled tools the model may call (`callers` includes `model`): the tools offered to it. Never widens:
	 * other tools can still call the ones it leaves out.
	 * - Unset: every such tool except deferred ones.
	 * - An array: exactly these, in order, deferred ones too.
	 * - An object: every such tool except deferred ones and those in `remove`, in registry order, followed by the
	 *   deferred tools in `add` that are not in `remove`, in `add` order. Names of tools that are not deferred or not
	 *   registered have no effect in `add`.
	 */
	modelTools?: string[] | { remove?: string[]; add?: string[] };
	/** Rendered after every extension section, as the section `instructions`. */
	instructions?: string;
	/** Directory within the environment's file system, passed to `HarnessOptions.env`. */
	cwd?: string;
};

/** A change to `pi.agent`: a given field replaces the stored one, `null` clears it, `undefined` changes nothing. */
export type AgentChange = {
	readonly model?: ModelRef | null;
	readonly thinkingLevel?: ModelThinkingLevel | null;
	readonly stream?: ConversationStreamOptions | null;
	readonly extensions?:
		| readonly Extension[]
		| { readonly add?: readonly Extension[]; readonly remove?: readonly Extension[] }
		| null;
	readonly tools?: readonly ToolRegistration[] | { readonly remove: readonly ToolRegistration[] } | null;
	/** Replaces the stored filter as a whole; tools loaded with `control.addTools` are in its `add`. */
	readonly modelTools?:
		| readonly ToolRegistration[]
		| { readonly remove?: readonly ToolRegistration[]; readonly add?: readonly ToolRegistration[] }
		| null;
	readonly instructions?: string | null;
	readonly cwd?: string | null;
};

/** A conversation's agent resolved against a registry snapshot and the settings. */
export type Agent<Tool extends ToolRegistration = ToolRegistration> = {
	readonly model?: ModelRef;
	readonly thinkingLevel: ModelThinkingLevel;
	readonly stream: ConversationStreamOptions;
	readonly extensions: readonly Extension<Tool>[];
	/** The tools a request offers, in order: enabled, callable by the model, and selected by `modelTools`. */
	readonly tools: readonly Tool[];
	/** The tools nested calls (`executeTool()`) resolve among, in order: enabled and callable by tools. */
	readonly callable: readonly Tool[];
	/** Enabled deferred tools the model may call and is not offered yet, in registry order: what `tool_search` loads. */
	readonly deferred: readonly Tool[];
	/** Extension sections, then `instructions` when set. */
	readonly sections: readonly PromptSection<Tool>[];
	readonly instructions?: string;
	readonly cwd?: string;
};

/**
 * Runs inside the creating commit, after the creation hook and the `agent` change. The conversation creation is already
 * a table write, so table reads here throw `ReadAfterWrite`; document access remains available.
 */
export type ConversationInit = (tx: Tx, conversationId: ConversationId) => void | Promise<void>;

export type ConversationCreateOptions = {
	readonly ownership: ConversationOwnership;
	/** Applied in the creating commit after the creation hook's copy, before `init`. */
	readonly agent?: AgentChange;
	readonly init?: ConversationInit;
};

/** Curated pi-ai request options; absent fields use pi-ai defaults. */
export type ConversationStreamOptions = {
	serviceTier?: OpenAICodexResponsesOptions["serviceTier"];
	transport?: Transport;
	timeoutMs?: number;
	/** Provider/SDK retries inside one request attempt. */
	maxRetries?: number;
	maxRetryDelayMs?: number;
	headers?: Record<string, string>;
	metadata?: JsonObject;
	cacheRetention?: CacheRetention;
	deferred?: boolean | { window?: "15m" | "1h" | "24h" };
};

/** Durable input selected by the task before acquiring authentication or opening transport. */
export type ModelRequestInput = {
	readonly purpose: string;
	readonly attempt: number;
	readonly model: PinnedModel;
	readonly messages: PinnedMessages;
	readonly cutoff: EntryId;
	readonly options: ConversationStreamOptions & Pick<SimpleStreamOptions, "reasoning" | "maxTokens">;
} & (
	| { readonly operation: "stream" | "complete"; readonly handle?: never }
	| { readonly operation: "fetchDeferred" | "cancelDeferred"; readonly handle: DeferredHandle }
);

/** Identity stamped by the scheduler, never supplied by a model/tool caller. */
export type ModelRequestTarget = ModelRequestInput & {
	readonly taskId: TaskId;
	readonly conversationId: ConversationId;
	readonly taskKind: string;
	readonly taskVersion: number;
};

/** Activation-local authentication, attributed transports and observers; never checkpointed. */
export type ModelRequestCapabilities = Pick<
	SimpleStreamOptions,
	| "apiKey"
	| "authType"
	| "env"
	| "fetch"
	| "connectWebSocket"
	| "telemetry"
	| "onPayload"
	| "onResponse"
	| "onProviderStreamEvent"
	| "sessionId"
> &
	ModelsRequestTransforms;

/** Pending access has a real durable readiness source, with no resident approval wait. */
export type ModelRequestWait = {
	readonly status: "waiting";
	readonly condition: Extract<TaskWaitCondition, { readonly kind: "input" | "receipt" | "registry" }>;
};

/** A connection owns its transports until close joins them, including cancellation and late acquisition. */
export type ModelRequestConnection = {
	readonly status: "ready";
	readonly options: ModelRequestCapabilities;
	/** Retain ownership of uncertain resources if cleanup rejects; the product owner must still release them. */
	close(context: Context): Promise<void>;
};

/** The invoking task's mutation boundary; never open another Session from a model port. */
export interface ModelRequestApi {
	/** Detached committed effective target, absent until endpoint preparation; original intent remains unchanged. */
	readonly prepared: ModelRequestTarget | undefined;
	/**
	 * Commit the effective endpoint before provider-specific admission or transport. Only baseUrl may differ from the
	 * selected descriptor. Same preparation is idempotent; conflicting reuse rejects. Deferred observations and
	 * cancellation reuse this round's exact endpoint. Scoped to the invoking request and task lifecycle.
	 */
	prepare(model: ModelRequestTarget["model"], context: Context): Promise<ModelRequestTarget>;
	/** Scoped to this request and invocation. Commit admission rejects after either ends or is cancelled. */
	commit<T>(change: (tx: Tx) => T | Promise<T>, context: Context): Promise<T>;
}

/** The embedding binds authority, credentials and transport to each committed model invocation. */
export type ModelRequestPort = (
	request: ModelRequestTarget,
	api: ModelRequestApi,
	context: Context,
) => Promise<ModelRequestConnection | ModelRequestWait>;

export type ModelRequestResult<T> = { readonly status: "completed"; readonly result: T } | ModelRequestWait;

/** Durable generation attempt retries; the JSON shape of pi-ai `RetryPolicy`. */
export type ConversationRetryPolicy = {
	enabled: boolean;
	maxRetries: number;
	baseDelayMs: number;
	maxAgentDelayMs?: number;
};

/** Automatic compaction thresholds (spec §8.7); manual compaction ignores `enabled`. */
export type CompactionPolicy = {
	/** Threshold and overflow compaction. */
	enabled: boolean;
	/** Room kept free for the answer: generation blocks to compact above `contextWindow - reserveTokens`. */
	reserveTokens: number;
	/** Approximate size of the recent context a summary keeps verbatim. */
	keepRecentTokens: number;
	/** Background compaction starts `backgroundTokens` below the blocking threshold; `0` disables it. */
	backgroundTokens: number;
};

/**
 * Harness-wide compaction settings. `modelOverrides` is keyed by exact `"provider/modelId"`; each field it sets
 * replaces the harness-wide one for that model, so small- and large-window models can share a harness.
 */
export type CompactionSettings = Partial<CompactionPolicy> & {
	readonly modelOverrides?: Readonly<Record<string, Partial<CompactionPolicy>>>;
};

/**
 * How often running progress is committed. A host can commit less often, so live answers and tool output appear in larger steps.
 */
export type ProgressPolicy = {
	/** Minimum pause between commits of the answer being generated. */
	partialIntervalMs: number;
	/** Minimum pause between commits of running tool output; large commits also pause in proportion to their size. */
	outputIntervalMs: number;
	/** Maximum time deferred progress remains unstored. */
	persistWithinMs: number;
};

/** Why a compaction runs: `compact()`, a threshold in generation preparation, or a context overflow. */
export type CompactionReason = "manual" | "threshold" | "overflow";

/**
 * `entryId` of a blocking compaction's summary, or the `submissionId` of a conversation-owned compaction's summary
 * write; both absent when nothing was compacted.
 */
export type CompactionResult = { entryId?: EntryId; submissionId?: SubmissionId };

/** Harness-wide run policy. Read at every resolution and never copied; getters are fine. Synchronous: some readers run on the Session line. */
export type HarnessSettings = {
	/** Default extension selection; absent: every installed extension, in install order. */
	readonly extensions?: readonly Extension[];
	readonly stream?: ConversationStreamOptions;
	readonly retry?: Partial<ConversationRetryPolicy>;
	readonly compaction?: CompactionSettings;
	readonly progress?: Partial<ProgressPolicy>;
	readonly toolExecution?: ToolExecutionMode;
	readonly steeringMode?: QueueMode;
	readonly followUpMode?: QueueMode;
	/**
	 * How long an idle conversation keeps its last context read in memory, so its next run reads only newer entries.
	 * Busy conversations always keep it; `0` drops it once the conversation is idle.
	 */
	readonly contextRetentionMs?: number;
	/**
	 * How long a loaded document that no watch observes stays in memory after its last read or write;
	 * the next access loads it from Storage. Observed documents always stay. Default ten minutes; `0` drops one right
	 * after use; `Infinity` keeps them. Anything else that is not a number of milliseconds means the default.
	 */
	readonly documentRetentionMs?: number;
	/**
	 * What a run that fails (`model_error`, `no_model`) does with the inputs queued behind it: `keep` leaves them
	 * queued until the next input or queued write, `continue` starts the next run with them, as an answer would, and
	 * `withdraw` settles them `unanswered` with reason `run_failed`. Default `keep`.
	 */
	readonly afterFailedRun?: AfterFailedRun;
};

/** What a failed run does with the inputs queued behind it (`HarnessSettings.afterFailedRun`). */
export type AfterFailedRun = "keep" | "continue" | "withdraw";

/** Resolved settings: every field over its built-in default, object fields merged. */
export type Settings = {
	/** Absent: every installed extension, in install order. */
	readonly extensions?: readonly Extension[];
	readonly stream: ConversationStreamOptions;
	readonly retry: ConversationRetryPolicy;
	/** Harness-wide policy over the defaults; per-model overrides apply where the model is known. */
	readonly compaction: CompactionPolicy & Pick<CompactionSettings, "modelOverrides">;
	readonly progress: ProgressPolicy;
	readonly toolExecution: ToolExecutionMode;
	readonly steeringMode: QueueMode;
	readonly followUpMode: QueueMode;
	readonly contextRetentionMs: number;
	readonly documentRetentionMs: number;
	readonly afterFailedRun: AfterFailedRun;
};

/** What `HarnessOptions.env` builds an environment for. */
export type EnvTarget = {
	readonly conversationId: ConversationId;
	/** The conversation's agent `cwd`. */
	readonly cwd?: string;
	readonly read: DocumentReader;
};

/** Detached, recursively frozen candidates staged before the embedding's commit preparation. */
export type HarnessCommit = {
	readonly entries: readonly EntryRecord[];
	readonly submissions: readonly SubmissionRecord[];
	readonly tasks: readonly TaskRecord<JsonValue, JsonValue, JsonValue>[];
	/** Candidate task view on this mutation line, including unchanged dependencies. Not a committed table read. */
	readonly task: (id: TaskId) => Promise<TaskRecord<JsonValue, JsonValue, JsonValue> | undefined>;
};

export type HarnessOptions<Tool extends ToolRegistration = ToolRegistration> = {
	/** Publish durable schedules outside the mutation line. Omitted: a local clock adapter owns the timer. */
	readonly publishWake?: (schedule: WakeSchedule) => Promise<void>;
	/** Omitted: ordinary pi-ai authentication and transport. Product embeddings supply a protected port. */
	readonly modelRequests?: ModelRequestPort;
	/** Host-owned adapter for content-free admission/phase telemetry; propagation and metrics are optional. */
	readonly telemetry?: TelemetryContext;
	/**
	 * pi-ai model access used by generation, compaction, hooks and tools: a `Models`, or a function returning the current
	 * one, called at each use, so a host can replace its model configuration without reopening the Harness.
	 */
	readonly models: Models | (() => Models);
	/**
	 * How model requests identify the app to providers, merged field by field over `DEFAULT_HARNESS_IDENTITY`
	 * (`{ attribution: PI_ATTRIBUTION }`, provider attribution to pi). It applies to every request through the Harness's
	 * models, including hooks and tools, and overrides the identity the `Models` was created with; a request's own
	 * `identity` overrides it. Pass `{ attribution: null }` to send no attribution headers.
	 */
	readonly identity?: ClientIdentity;
	readonly registry: RegistryReader<Tool>;
	/** Run settings, or a function returning the current ones; read at each use either way. */
	readonly settings?: HarnessSettings | (() => HarnessSettings | undefined);
	/** Builds a conversation's environment at each use. Never called on the Session line; may be async. */
	readonly env?: (target: EnvTarget, context: Context) => ExecutionEnv | undefined | Promise<ExecutionEnv | undefined>;
	/**
	 * Runs in every commit that creates or forks a conversation, raw `tx.createConversation()` included, after the
	 * built-in `pi.*` documents and before the conveniences apply `agent` and run `init`. A fork already has its copies.
	 * Table reads throw `ReadAfterWrite`, as in `init`; a throw fails the creating commit.
	 */
	readonly conversationCreated?: (tx: Tx, conversation: ConversationRecord) => void | Promise<void>;
	/**
	 * Runs once after each commit callback settles, before native scheduler preparation and Storage admission. The
	 * snapshot includes appended entries, resolved submission candidates and created/replaced task candidates; writes made here are not fed back into
	 * this hook. Derived documents, entries and tasks join the same atomic batch and native wake schedule. A throw
	 * rolls back the whole commit. Await every Tx operation; table reads retain the ordinary ReadAfterWrite restriction.
	 * Do not call Session APIs or perform external effects here. Reopen may call this with no staged entries or tasks.
	 */
	readonly prepareCommit?: (tx: Tx, staged: HarnessCommit, context: Context) => void | Promise<void>;
	/**
	 * The Harness clock, epoch milliseconds; default `Date.now`. Sleeping tasks store deadlines on it, so every process
	 * that opens the same storage must use the same clock.
	 */
	readonly now?: () => number;
	/**
	 * Arms the timers of the Session, the scheduler, and its tasks: pacing progress commits, freeing idle memory
	 * (`background`), and pacing for `TaskRuntime.setTimer()`. The
	 * Node.js environment, file watcher, and image worker keep their own timers. Default `setTimeout`, with background
	 * timers unreferenced, or declined in Cloudflare Workers and where they cannot be unreferenced. Tests can pass fake
	 * timers; a host can keep timers off a process that should hibernate.
	 */
	readonly setTimer?: SetTimer;
	/**
	 * Receives failures that do not fail the calling operation, such as a throwing hook or listener, a task that faulted,
	 * and the error that fails the Session, once. `info` says what happened: a host can tell a failed Session
	 * (`session_failed`, often a platform reset) from a faulted task (`task_faulted`).
	 * Its own throw or rejection is dropped.
	 */
	readonly onReport?: (error: unknown, info: ReportInfo) => void;
	/**
	 * Open for reading only, as a viewer of a session that another process drives, for example a synced copy on a device
	 * that only shows it. Open recovers nothing: a `running` task stays as committed. Every commit with writes rejects
	 * with `ReadOnlySession` before it reaches Storage, and so does `resume()` and every call that enables scheduling;
	 * the Harness stays usable for reads and watches. Reopen without it to write.
	 */
	readonly readOnly?: boolean;
};

/** Live task and what the scheduler would do with it under the current registry. */
export type TaskInspection = {
	readonly record: TaskRecord<JsonValue, JsonValue, JsonValue>;
	readonly state: /** An invocation is active. */
		| { readonly kind: "running" }
		/** The next scheduling pass reserves it; `migrates` when its definition is newer and has `migrate`. */
		| { readonly kind: "ready"; readonly migrates: boolean }
		/**
		 * Waits for these live tasks: the live part of its `on`, or, when abort-marked, its live ordinary owned work, which
		 * must end before its abort handler starts.
		 */
		| { readonly kind: "waiting"; readonly on: readonly TaskId[]; readonly condition?: TaskWaitCondition }
		/** Outcome held until its ordinary owned work drains. */
		| { readonly kind: "completing" }
		/** Unavailable executable binding or a failed invocation; ownership remains live. */
		| {
				readonly kind: "blocked";
				readonly reason:
					| "missing_task"
					| "task_too_old"
					| "migration_failed"
					| "incompatible_binding"
					| "invocation_failed";
				readonly error?: unknown;
		  };
};

/** Point-in-time view of live work: unfinished tasks and submissions, read on the Session line. */
export type HarnessInspection = {
	readonly scheduling: "paused" | "running" | "closing";
	readonly tasks: readonly TaskInspection[];
	/** Queued and placed submissions, in ID order. */
	readonly submissions: readonly SubmissionRecord[];
};

/** Raw active transcript and derived model context. */
export type ContextView = {
	/** Newest applicable head marker, if any. */
	readonly head: EntryRecord | undefined;
	/** Raw active entries: the head marker followed by non-head entries from its head through the tail. */
	readonly entries: readonly EntryRecord[];
	/** Per entry of `entries`, its model messages after edits and excluded stop reasons, before tool result ordering. */
	readonly contributions: readonly (readonly Message[])[];
	/** Model context for the next provider request. */
	readonly messages: readonly Message[];
};

/** Knowledge-only entry graph: local source IDs locate heads and edits, never executable task attribution. */
export type ConversationHistoryEntry = Pick<EntryRecord, "id" | "model" | "head" | "edits"> & {
	/** Positional system entries retain their role in native prompt-baseline replay. */
	readonly system?: true;
};

/**
 * Detached immutable transcript knowledge at an exact visible frontier. Contains no tasks, submissions, receipts,
 * waits, authority, arbitrary entry data, or copied documents. A null frontier deliberately exports no entries and
 * pins the source's currently committed agent settings; an entry frontier pins settings as of that entry's commit.
 * The embedding authenticates the source owner separately; these IDs are local transcript provenance only.
 */
export type ConversationHistory = {
	readonly source: { readonly conversationId: ConversationId; readonly at: EntryId | null };
	readonly agent: Readonly<AgentState>;
	readonly entries: readonly ConversationHistoryEntry[];
};

/** Exact receiving entry IDs allocated by the native import, detached and frozen before initialization. */
export type ConversationHistoryEntryMap = Readonly<Record<EntryId, EntryId>>;

/** Bind receiving knowledge provenance in the same creating commit, without reading newly staged native tables. */
export type ConversationHistoryImportOptions = Omit<ConversationCreateOptions, "init"> & {
	readonly init?: (
		tx: Tx,
		conversationId: ConversationId,
		entryIds: ConversationHistoryEntryMap,
	) => void | Promise<void>;
};

/** Stateless handle for one conversation, bound to the Harness that returned it. Compare handles by `id`. */
export interface Conversation {
	readonly id: ConversationId;

	/** Resolved with the current registry snapshot and settings. */
	agent(context: Context): Promise<Agent>;
	/** `configure()` in its own commit. */
	configure(change: AgentChange, context: Context): Promise<void>;

	/**
	 * Durably admit user input or a passive entry write. A busy conversation queues input and writes that affect
	 * model context in `pi.inbox`; `whenBusy: "reject"` rejects input with `ConversationBusy` and writes nothing.
	 * A write of only `kind` and `data` is appended at once, even mid-run, as it changes nothing the model sees.
	 * A payload factory composes local writes and payload preparation with fresh admission atomically;
	 * request-ID replay skips it.
	 */
	submit(submission: SubmissionDraft, context: Context): Promise<Submission>;
	/** Invoke an actually selected tool without a model call, using the same native task lifecycle. */
	invokeTool(
		call: DirectToolCall,
		context: Context,
		options?: { readonly background?: boolean },
	): Promise<TaskId<ToolTaskResult>>;
	/**
	 * Admit a write of a `pi.reset` entry that starts a new context, carrying `handoff` as a user message when given.
	 * Resolves after admission; while busy, it is placed at the next boundary.
	 */
	reset(handoff: string | undefined, context: Context): Promise<void>;
	/**
	 * Admit a manual compaction task and return its ID. It summarizes while the conversation keeps working and places its
	 * summary through a write submission: at once when idle, otherwise at the next boundary (spec §8.7).
	 */
	compact(instructions: string | undefined, context: Context): Promise<TaskId<CompactionResult>>;

	/** Session commit whose `tx.createTask()` defaults to this conversation. */
	commit<T>(change: (tx: Tx) => T | Promise<T>, context: Context, options?: CommitOptions): Promise<T>;
	/**
	 * Committed raw active transcript and model context. With `at`, the context as of that visible entry: the same view
	 * `fork(at)` would start with, without creating a conversation.
	 */
	context(context: Context, options?: { readonly at?: EntryId }): Promise<ContextView>;
	/** Fork-aware history of this conversation, newest first unless `query.order` is `ascending`. */
	entries(
		query: Omit<EntryQuery, "conversationId">,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<EntryRecord, Cursor>>;
	fork(at: EntryId, options: ConversationCreateOptions, context: Context): Promise<Conversation>;
	/** Export knowledge through exactly this visible entry; null deliberately selects an empty prefix. */
	exportHistory(at: EntryId | null, context: Context): Promise<ConversationHistory>;
	/**
	 * Withdraw queued inputs (queued writes stay), mark every live non-background task of the ordinary ownership scope,
	 * signal them, and resolve once the scope is idle. Background subtrees survive unless `background` is set.
	 */
	abort(context: Context, options?: ConversationAbortOptions): Promise<void>;
	/**
	 * Interrupt the active generation and join its owned work without withdrawing queued inputs. Queued steers resume
	 * the original input run; otherwise exactly one follow-up starts a new run. An idle conversation advances its queue.
	 * Resolves once the old generation is terminal and the next input boundary is committed, not after the new answer.
	 */
	flush(context: Context): Promise<void>;
	/**
	 * Supersede exactly this generation and join its owned work, then prepare a new generation from current settings
	 * for the same admitted inputs. No input or transcript message is manufactured. A queued reset or explicit abort
	 * takes precedence. Returns false when the named generation no longer owns the run or is already cancelling.
	 */
	reprepare(taskId: TaskId, context: Context): Promise<boolean>;
	/**
	 * Resolve when the ordinary ownership scope has no live non-background task: this conversation and the conversations
	 * owned, transitively, by its non-background tasks.
	 */
	waitForIdle(context: Context): Promise<void>;
	/**
	 * The structural view (spec §9.3) as a serialized watch; commits made while its listener runs arrive folded. Built on
	 * the same mount as `Harness.observeConversation()`.
	 */
	viewState(context: Context): Promise<AttachedReplicatedState<ConversationView>>;
	watch(context: Context): Promise<ConversationWatch>;
}

export type ConversationWatch = WatchHandle<ConversationView>;

/** Durable agent harness over one Session. */
export interface Harness extends Session {
	/** Publish the current committed schedule again, e.g. after a host recovery scan. */
	flushWake(context: Context): Promise<WakeSchedule>;
	/** Run one finite pass and wait until its invocations have released. */
	runPass(context: Context): Promise<WakeSchedule>;
	/**
	 * Enable task scheduling. Idempotent; throws after close, and `ReadOnlySession` on a read-only Harness. Calls that
	 * ask for progress enable it too: `Conversation.submit()`, `Conversation.compact()`, `Conversation.abort()`,
	 * `Submission.wait()`, `waitForTask()`, `Harness.waitForIdle()`, and `Conversation.waitForIdle()`. Read-only viewers
	 * never do.
	 */
	resume(): void;

	/** Return the reserved root conversation, creating it with `agent` and `init` in one commit when absent. */
	root(
		context: Context,
		options?: { readonly agent?: AgentChange; readonly init?: ConversationInit },
	): Promise<Conversation>;
	conversation(id: ConversationId, context: Context): Promise<Conversation | undefined>;
	createConversation(options: ConversationCreateOptions, context: Context): Promise<Conversation>;
	/** Create fresh ownership and import only transcript knowledge atomically, then apply receiving agent/init options. */
	importHistory(
		history: ConversationHistory,
		options: ConversationHistoryImportOptions,
		context: Context,
	): Promise<Conversation>;

	getTask<R>(id: TaskId<R>, context: Context): Promise<TaskRecord<JsonValue, JsonValue, R> | undefined>;
	/** Live tasks and unsettled submissions. Writes nothing and runs no task code. */
	inspect(context: Context): Promise<HarnessInspection>;
	/** Reacquire a submission, for example after reopen. */
	submission(id: SubmissionId, context: Context): Promise<Submission | undefined>;
	/** `not_found` for an unknown submission or one of another conversation than `conversationId`. */
	abortSubmission(
		id: SubmissionId,
		context: Context,
		conversationId?: ConversationId,
	): Promise<"aborted" | "already_placed" | "settled" | "not_found">;
	/**
	 * Commit the abort mark, signal and join an active run invocation, and schedule cleanup. Unavailable definitions
	 * park on registry publication. Repeating cancellation never retries a failed cleanup attempt.
	 */
	abortTask(id: TaskId, context: Context): Promise<"marked" | "terminal">;
	/** Retry only the named failed invocation incident, in its retained run/abort mode. Stale/duplicate repair cannot retry a newer failure. */
	retryTask(id: TaskId, incident: EntryId, context: Context): Promise<"queued" | "stale" | "terminal">;
	/** Resolve with the terminal receipt; a retained invocation failure rejects while preserving ownership. Caller cancellation ends only this wait. */
	waitForTask<R>(id: TaskId<R>, context: Context): Promise<SettledTask<R>>;
	/** Resolve when the ordinary ownership scope of every ownerless conversation has no live non-background task. */
	waitForIdle(context: Context): Promise<void>;
	/** Session total: every conversation's `pi.usage` summed. */
	usage(context: Context): Promise<UsageState>;
	/**
	 * Every live task with its owner edge, status, and owned conversations (spec §9.5), as a serialized watch; commits
	 * made while its listener runs arrive folded.
	 */
	taskGraph(context: Context): Promise<AttachedReplicatedState<TaskGraph>>;
	watchTaskGraph(context: Context): Promise<TaskGraphWatch>;
	/**
	 * Attach a synchronous `Observer` to a conversation's structural view (spec §9.3), atomically on the Session line,
	 * for adapters: `advance` gets each next revision with its operations inside Storage's `adopt` after it is durable,
	 * so it only records it. `context` governs only
	 * the attachment.
	 */
	observeConversation(
		id: ConversationId,
		observer: Observer<ConversationView>,
		context: Context,
	): Promise<Observation<ConversationView>>;
	/**
	 * Attach a synchronous `Observer` to the task graph (spec §9.5), as `observeConversation()` does to a view. A
	 * `blocked` reason that changes without a commit (a registry change, a failed migration) advances it on the Session
	 * line outside any commit and `adopt`, with the background context.
	 */
	observeTaskGraph(observer: Observer<TaskGraph>, context: Context): Promise<Observation<TaskGraph>>;
}

/** What a hook may use: committed reads and the asking task's memos, which hooks and the task share. */
export interface HookApi extends DocumentReader {
	readonly taskId: TaskId;
	readonly conversationId: ConversationId;
	/** The current `HarnessOptions.models`, read at each access. */
	readonly models: Models;
	memo<T extends JsonValue>(name: string, context: Context): Promise<T | undefined>;
	memo<T extends JsonValue>(name: string, candidate: T, context: Context): Promise<T>;
}

export type HookResult<T> = T | undefined | Promise<T | undefined>;

/** A request-policy change for the same input run after an actual provider error. */
export type GenerationRetrySelection = {
	readonly model: ModelRef;
	readonly thinkingLevel?: ModelThinkingLevel;
	readonly stream?: ConversationStreamOptions;
};

export type GenerationResponseRequest = {
	readonly attempt: number;
	readonly model: PinnedModel;
	readonly thinkingLevel: ModelThinkingLevel;
	readonly streamOptions: ConversationStreamOptions;
	readonly cutoff: EntryId;
};

/** Hooks of the built-in generation task. */
export interface GenerationHooks {
	/** Before actual input binding commits; committed request recovery does not rerun this hook. */
	beforeRequest(
		request: { readonly messages: readonly Message[] },
		api: HookApi,
		context: Context,
	): HookResult<{ readonly messages: readonly Message[] }>;
	/** Every terminal provider message, before classification. */
	afterResponse(
		message: AssistantMessage,
		api: HookApi,
		context: Context,
		request: GenerationResponseRequest,
	): void | Promise<void> | HookResult<{ readonly retry: GenerationRetrySelection }>;
	/** A final answer; the first `continue` appends a user message and continues the run. */
	onYield(answer: AssistantMessage, api: HookApi, context: Context): HookResult<{ readonly continue: UserInput }>;
	/** After every tool of the round is terminal; `results` are the round's result entries in call order. */
	afterTools(assistant: EntryId, results: readonly EntryId[], api: HookApi, context: Context): void | Promise<void>;
}

/** A call as tool hooks see it; `parent` is set for a nested call, one a running tool made through `executeTool()`. */
export type ToolHookCall = ToolCall & {
	readonly parent?: { readonly taskId: TaskId; readonly callId: string };
};

/** Hooks of the built-in tool task, for model-issued and nested calls. */
export interface ToolHooks {
	/** Before intent; the first `block` wins, otherwise `arguments` replace the call's arguments. A throw blocks. */
	beforeTool(
		call: ToolHookCall,
		api: HookApi,
		context: Context,
	): HookResult<{ readonly arguments?: JsonObject; readonly block?: string }>;
	/** After execution, before the result is committed; replaces the result. */
	afterTool(
		call: ToolHookCall,
		result: ToolExecutionResult,
		api: HookApi,
		context: Context,
	): HookResult<ToolExecutionResult>;
}

/** Hooks of the built-in compaction task. */
export interface CompactionHooks {
	/**
	 * After range selection, before summarizing; the first decision wins. `entries` are the active entries the summary
	 * replaces, the head marker first, and `messages` their model context, the summarizer's source; `firstKept` is the
	 * first entry kept verbatim.
	 */
	beforeCompact(
		compaction: {
			readonly reason: CompactionReason;
			readonly entries: readonly EntryRecord[];
			readonly messages: readonly Message[];
			readonly firstKept: EntryId;
			readonly instructions?: string;
		},
		api: HookApi,
		context: Context,
	): HookResult<{ readonly decline: true } | { readonly summary: string }>;
}
