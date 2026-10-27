import type { AttachedReplicatedState, Context, JsonValue } from "@earendil-works/chord";
import { withAbortSignal, withoutAbortSignal } from "@earendil-works/chord/context";
import { copyJson } from "@earendil-works/chord/json";
import type { Models } from "@earendil-works/pi-ai";
import {
	type ClientIdentity,
	mergeClientIdentity,
	PI_ATTRIBUTION,
	withClientIdentity,
} from "@earendil-works/pi-ai/utils/identity";
import {
	createTypedMetrics,
	NOOP_TELEMETRY_CONTEXT,
	NOOP_TELEMETRY_METRICS,
	type SpanAttributes,
	type TelemetryContext,
} from "@earendil-works/pi-telemetry";
import { ResetEntry } from "../entries.ts";
import type { ExecutionEnv } from "../env/index.ts";
import { contained, SessionImpl } from "../session/session.ts";
import type { Transaction } from "../session/transaction.ts";
import { hostTimers } from "../timers.ts";
import type {
	CommitOptions,
	ConversationId,
	ConversationOwnership,
	ConversationRecord,
	Cursor,
	EntryId,
	EntryQuery,
	EntryRecord,
	Observation,
	Observer,
	Page,
	ReportInfo,
	Storage,
	SubmissionId,
	TaskId,
	TaskRecord,
	Tx,
} from "../types.ts";
import { ROOT_CONVERSATION_ID } from "../types.ts";
import { AgentDoc, configure, createAgent, resolveAgent, resolveSettings } from "./agent.ts";
import { bindTool } from "./binding.ts";
import { createCompaction } from "./compaction.ts";
import { readContext } from "./context.ts";
import { GenerationTask, startRun } from "./generation.ts";
import { exportConversationHistory, importConversationHistory, prepareConversationHistory } from "./history.ts";
import { applyBoundary, InboxDoc, prepareBoundary, withdrawQueuedInputs } from "./inbox.ts";
import { LiveDoc, settleSchedulerOutcome } from "./live.ts";
import { ProviderDoc } from "./provider.ts";
import { BUILTIN_TASKS } from "./registry.ts";
import { type InvocationBinding, TaskScheduler } from "./scheduler.ts";
import { Submissions } from "./submissions.ts";
import { type TaskGraph, TaskGraphView, type TaskGraphWatch } from "./task-graph.ts";
import { DURABLE_TELEMETRY_SCHEMA } from "./telemetry.ts";
import { createDirectToolTask, type DirectToolCall, type ToolTaskResult } from "./tool.ts";
import type {
	Agent,
	AgentChange,
	CompactionResult,
	ContextView,
	Conversation,
	ConversationAbortOptions,
	ConversationCreateOptions,
	ConversationHandle,
	ConversationHistory,
	ConversationHistoryImportOptions,
	ConversationInit,
	ConversationWatch,
	HarnessCommit,
	HarnessInspection,
	HarnessOptions,
	HarnessSettings,
	Harness as HarnessType,
	RegistrySnapshot,
	SettledTask,
	Submission,
	SubmissionDraft,
	ToolRegistration,
} from "./types.ts";
import { addUsageState, UsageDoc, type UsageState } from "./usage.ts";
import { scanAll } from "./util.ts";
import { type ConversationView, ConversationViews } from "./view.ts";

const SCAN_PAGE_SIZE = 256;

type CreateTarget =
	| { readonly kind: "root" }
	| { readonly kind: "independent"; readonly ownership: ConversationOwnership }
	| {
			readonly kind: "history";
			readonly ownership: ConversationOwnership;
			readonly history: ConversationHistory;
			readonly initialize?: ConversationHistoryImportOptions["init"];
	  }
	| {
			readonly kind: "fork";
			readonly parentId: ConversationId;
			readonly at: EntryId;
			readonly ownership: ConversationOwnership;
	  };

/** What the conveniences apply in the creating commit, after the creation hook. */
type CreateOptions = { readonly agent?: AgentChange; readonly init?: ConversationInit };

/** Harness-private services used by Conversation handles. */
type ConversationHost<Tool extends ToolRegistration> = {
	readonly harness: HarnessImpl<Tool>;
	readonly storage: Storage;
	readonly tasks: TaskScheduler;
	readonly submissions: Submissions;
	readonly views: ConversationViews;
	readonly now: () => number;
	create(target: CreateTarget, options: CreateOptions, context: Context): Promise<Conversation>;
};

/** Key of the Harness's internal index-size getter, for tests. */
export const SCHEDULER_INDEX_SIZES = Symbol.for("pi-durable.schedulerIndexSizes");

class ConversationImpl<Tool extends ToolRegistration> implements Conversation {
	readonly id: ConversationId;
	readonly #host: ConversationHost<Tool>;

	constructor(id: ConversationId, host: ConversationHost<Tool>) {
		this.id = id;
		this.#host = host;
	}

	agent(context: Context): Promise<Agent> {
		return this.#host.harness.resolveAgent(this.id, undefined, context);
	}

	configure(change: AgentChange, context: Context): Promise<void> {
		return this.#host.harness.commitWith((tx) => configure(tx, this.id, change), context);
	}

	submit(submission: SubmissionDraft, context: Context): Promise<Submission> {
		return this.#host.submissions.submit(this.id, submission, context);
	}

	invokeTool(
		call: DirectToolCall,
		context: Context,
		options?: { readonly background?: boolean },
	): Promise<TaskId<ToolTaskResult>> {
		this.#host.tasks.resume();
		return this.#host.harness.invokeTool(this.id, call, context, options);
	}

	compact(instructions: string | undefined, context: Context): Promise<TaskId<CompactionResult>> {
		this.#host.tasks.resume();
		const input = { reason: "manual", ...(instructions === undefined ? {} : { instructions }) } as const;
		return this.#host.harness.commitWith((tx) => createCompaction(tx, this.id, input), context);
	}

	async reset(handoff: string | undefined, context: Context): Promise<void> {
		const model =
			handoff === undefined
				? {}
				: { model: [{ role: "user", content: handoff, timestamp: this.#host.now() } as const] };
		const entry = { kind: ResetEntry.kind, head: "self", ...model } as const;
		await this.#host.submissions.submit(this.id, { type: "write", entry }, context);
	}

	commit<T>(change: (tx: Tx) => T | Promise<T>, context: Context, options?: CommitOptions): Promise<T> {
		return this.#host.harness.commitWith(change, context, { conversationId: this.id }, options);
	}

	context(context: Context, options?: { readonly at?: EntryId }): Promise<ContextView> {
		return readContext(this.#host.harness, this.#host.storage, this.id, context, options?.at);
	}

	entries(
		query: Omit<EntryQuery, "conversationId">,
		limit: number,
		cursor: Cursor | undefined,
		context: Context,
	): Promise<Page<EntryRecord, Cursor>> {
		const bounded: EntryQuery = {
			conversationId: this.id,
			...(query.minEntryId === undefined ? {} : { minEntryId: query.minEntryId }),
			...(query.maxEntryId === undefined ? {} : { maxEntryId: query.maxEntryId }),
			...(query.order === undefined ? {} : { order: query.order }),
		};
		return this.#host.harness.readOnLine(() => this.#host.storage.scanEntries(bounded, limit, cursor, context));
	}

	fork(at: EntryId, options: ConversationCreateOptions, context: Context): Promise<Conversation> {
		return this.#host.create({ kind: "fork", parentId: this.id, at, ownership: options.ownership }, options, context);
	}

	exportHistory(at: EntryId | null, context: Context): Promise<ConversationHistory> {
		return exportConversationHistory(this.#host.harness, this.#host.storage, this.id, at, context);
	}

	async abort(context: Context, options?: ConversationAbortOptions): Promise<void> {
		this.#host.tasks.resume();
		return this.#host.tasks.abortConversation(this.id, options?.background === true, context);
	}

	flush(context: Context): Promise<void> {
		return this.#host.harness.flushConversation(this.id, context);
	}

	reprepare(taskId: TaskId, context: Context): Promise<boolean> {
		return this.#host.harness.reprepareConversation(this.id, taskId, context);
	}

	waitForIdle(context: Context): Promise<void> {
		this.#host.tasks.resume();
		return this.#host.tasks.waitForIdle(this.id, context);
	}

	viewState(context: Context): Promise<AttachedReplicatedState<ConversationView>> {
		return this.#host.views.state(this.id, context);
	}

	watch(context: Context): Promise<ConversationWatch> {
		return this.#host.views.watch(this.id, context);
	}
}

/** Request identity of a Harness's model requests: pi-ai's defaults with provider attribution to pi. */
export const DEFAULT_HARNESS_IDENTITY: ClientIdentity = { attribution: PI_ATTRIBUTION };

/**
 * `HarnessOptions.models` as a function with `HarnessOptions.identity` applied. A plain `Models` is wrapped once; a
 * function's result is wrapped at each call.
 */
function currentModels(models: HarnessOptions["models"], identity: ClientIdentity | undefined): () => Models {
	const merged = mergeClientIdentity(DEFAULT_HARNESS_IDENTITY, identity) ?? DEFAULT_HARNESS_IDENTITY;
	if (typeof models === "function") return () => withClientIdentity(models(), merged);
	const wrapped = withClientIdentity(models, merged);
	return () => wrapped;
}

/** `HarnessOptions.settings` as a function: plain settings, or none, are returned at every call. */
function currentSettings(settings: HarnessOptions["settings"]): () => HarnessSettings | undefined {
	return typeof settings === "function" ? settings : () => settings;
}

/** Session kernel extended with conversation handles and a registry. */
class HarnessImpl<Tool extends ToolRegistration> extends SessionImpl implements HarnessType {
	readonly #options: HarnessOptions<Tool>;
	readonly #hostSettings: () => HarnessSettings | undefined;
	readonly #report: (error: unknown, info: ReportInfo, context?: Context) => void;
	readonly #host: ConversationHost<Tool>;
	readonly #tasks: TaskScheduler;
	readonly #submissions: Submissions;
	readonly #taskGraph: TaskGraphView;

	constructor(rawStorage: Storage, options: HarnessOptions<Tool>, context: Context) {
		const report = safeReport(options.onReport, options.telemetry ?? NOOP_TELEMETRY_CONTEXT);
		const now = safeNow(options.now, report);
		const hostSettings = currentSettings(options.settings);
		// Read at each use, as all settings are, but only these fields: they are read on every commit.
		super(
			rawStorage,
			now,
			() => hostSettings()?.documentRetentionMs,
			options.setTimer,
			options.readOnly === true,
			() => hostSettings()?.progress?.persistWithinMs,
		);
		// Every component reads and writes through the Session's failure guard: `rawStorage` serves only `super()`.
		const storage = this.storage;
		this.#options = options;
		this.#report = report;
		this.#hostSettings = hostSettings;
		const settings = () => resolveSettings(hostSettings());
		this.#tasks = new TaskScheduler({
			telemetry: options.telemetry,
			session: this,
			storage,
			registry: options.registry,
			modelRequests: options.modelRequests,
			models: currentModels(options.models, options.identity),
			agent: (id, snapshot, callContext) => this.resolveAgent(id, snapshot as RegistrySnapshot<Tool>, callContext),
			settings,
			env: (id, callContext) => this.buildEnv(id, callContext),
			now,
			timers: hostTimers(options.setTimer, (error) => report(error, { type: "host_error", callback: "setTimer" })),
			report: this.#report,
			publishWake: options.publishWake,
			settleOutcome: settleSchedulerOutcome,
			withdrawInputs: withdrawQueuedInputs,
			conversation: async (id, binding, callContext) => {
				const record = await this.readOnLine(() => storage.conversation(id, callContext));
				return record === undefined ? undefined : boundConversation(id, binding, this.#submissions, this.#tasks);
			},
			context: withoutAbortSignal(context),
			readOnly: options.readOnly === true,
		});
		this.#submissions = new Submissions(this, storage, now, settings, () => this.#tasks.resume(), options.telemetry);
		this.#taskGraph = new TaskGraphView(this, storage, {
			blocked: (record) => this.#tasks.blockedReason(record),
			subscribeBlocked: (listener) => this.#tasks.subscribeBlocked(listener),
		});
		this.#host = {
			harness: this,
			storage,
			tasks: this.#tasks,
			submissions: this.#submissions,
			views: new ConversationViews(this, storage),
			now,
			create: (target, createOptions, context) => this.#create(target, createOptions, context),
		};
	}

	/** Resolve a conversation's committed `pi.agent` against `snapshot`, or the current one, and the current settings. */
	async resolveAgent(
		id: ConversationId,
		snapshot: RegistrySnapshot<Tool> | undefined,
		context: Context,
	): Promise<Agent<Tool>> {
		const registry = snapshot ?? this.#options.registry.snapshot();
		const state = await this.snapshot(AgentDoc, id, context);
		return resolveAgent(state, registry, resolveSettings(this.#hostSettings()), (error, extension) =>
			this.#report(error, { type: "extension_error", conversationId: id, extension }, context),
		);
	}

	invokeTool(
		conversationId: ConversationId,
		call: DirectToolCall,
		context: Context,
		options?: { readonly background?: boolean },
	): Promise<TaskId<ToolTaskResult>> {
		const snapshot = this.#options.registry.snapshot();
		const settings = resolveSettings(this.#hostSettings());
		return this.commitWith(
			async (tx) => {
				if (!(await tx.conversation(conversationId)))
					throw new Error(`Conversation ${conversationId} does not exist`);
				const agent = resolveAgent(await tx.doc(AgentDoc, conversationId), snapshot, settings, (error, extension) =>
					this.#report(error, { type: "extension_error", conversationId, extension }, context),
				);
				const tool = agent.tools.find((tool) => tool.name === call.name);
				if (!tool) throw new Error(`Tool ${call.name} is not selected for conversation ${conversationId}`);
				return createDirectToolTask(tx, conversationId, call, bindTool(tool, settings.toolExecution), {
					ownership: { kind: "conversation" },
					...(options?.background === undefined ? {} : { background: options.background }),
				});
			},
			context,
			{ conversationId },
		);
	}

	/** Build a conversation's environment from its current `cwd`; `undefined` without an `env` option. */
	async buildEnv(id: ConversationId, context: Context): Promise<ExecutionEnv | undefined> {
		const build = this.#options.env;
		if (build === undefined) return undefined;
		const cwd = (await this.snapshot(AgentDoc, id, context))?.cwd;
		return build({ conversationId: id, ...(cwd === undefined ? {} : { cwd }), read: this }, context);
	}

	/** Reconcile surviving `running` tasks to `pending`; part of open. */
	openTasks(context: Context): Promise<void> {
		return this.#tasks.open(context);
	}

	resume(): void {
		this.assertUsable();
		this.#tasks.resume();
	}

	flushWake(context: Context) {
		return this.#tasks.flushWake(context);
	}

	runPass(context: Context) {
		return this.#tasks.runPass(context);
	}

	protected override async prepareCommit(tx: Transaction, context: Context): Promise<void> {
		if (this.#options.prepareCommit !== undefined) {
			const candidates = copyJson({
				entries: tx.stagedEntries(),
				submissions: await tx.stagedSubmissions(),
				tasks: tx.stagedTasks(),
			}) as unknown as Omit<HarnessCommit, "task">;
			const staged = {
				...candidates,
				task: async (id: TaskId) => {
					const record = await tx.prepareTask(id);
					if (record !== undefined) freezeCommit(record);
					return record;
				},
			} as unknown as HarnessCommit;
			freezeCommit(staged);
			await this.#options.prepareCommit(tx, staged, context);
			tx.assertCallbackSettled();
		}
		await this.#tasks.prepare(tx);
	}

	getTask<R>(id: TaskId<R>, context: Context): Promise<TaskRecord<JsonValue, JsonValue, R> | undefined> {
		return this.readOnLine(() => this.storage.task(id, context)) as Promise<
			TaskRecord<JsonValue, JsonValue, R> | undefined
		>;
	}

	/** Internal, for tests: sizes of the scheduler's ownership indexes. */
	get [SCHEDULER_INDEX_SIZES](): TaskScheduler["indexSizes"] {
		return this.#tasks.indexSizes;
	}

	inspect(context: Context): Promise<HarnessInspection> {
		return this.readOnLine(async () => {
			const { scheduling, tasks } = await this.#tasks.inspect(this.#options.registry.snapshot());
			const scan = (status: "queued" | "placed") =>
				scanAll((cursor) => this.storage.scanSubmissions({ status }, SCAN_PAGE_SIZE, cursor, context));
			const submissions = [...(await scan("queued")), ...(await scan("placed"))].sort((a, b) => a.id - b.id);
			return { scheduling, tasks, submissions };
		});
	}

	submission(id: SubmissionId, context: Context): Promise<Submission | undefined> {
		return this.#submissions.get(id, context);
	}

	abortSubmission(
		id: SubmissionId,
		context: Context,
		conversationId?: ConversationId,
	): Promise<"aborted" | "already_placed" | "settled" | "not_found"> {
		return this.#submissions.abort(id, context, conversationId);
	}

	abortTask(id: TaskId, context: Context): Promise<"marked" | "terminal"> {
		return this.#tasks.abort(id, context);
	}

	async flushConversation(conversationId: ConversationId, context: Context): Promise<void> {
		this.#tasks.resume();
		const taskId = await this.commitWith(
			async (tx) => {
				const live = await tx.doc(LiveDoc, conversationId);
				if (live.run === undefined) {
					const boundary = await prepareBoundary(tx, conversationId, {
						...resolveSettings(this.#hostSettings()),
						followUpMode: "one-at-a-time",
					});
					const { users } = await applyBoundary(tx, boundary, "final", this.#host.now());
					if (users.length > 0) await startRun(tx, conversationId, live, users);
					return undefined;
				}
				const task = await tx.task(live.run.taskId);
				if (task === undefined || task.state.status === "terminal")
					throw new Error("Active input run has no live generation");
				if (!task.abortRequested) {
					live.run.interruption = { kind: "flush" };
					tx.setTask({ ...task, abortRequested: true });
				}
				return task.id;
			},
			context,
			{ conversationId },
		);
		if (taskId !== undefined) await this.#tasks.waitForTask(taskId, context);
	}

	async reprepareConversation(conversationId: ConversationId, taskId: TaskId, context: Context): Promise<boolean> {
		this.#tasks.resume();
		const admitted = await this.commitWith(
			async (tx) => {
				const live = await tx.doc(LiveDoc, conversationId);
				if (live.run?.taskId !== taskId) return false;
				const task = await tx.task(taskId);
				if (
					!task ||
					task.kind !== GenerationTask.definition.name ||
					task.state.status === "terminal" ||
					task.state.status === "completing" ||
					(task.state.checkpoint as { phase: string }).phase === "tools" ||
					task.abortRequested
				)
					return false;
				live.run.interruption = { kind: "reprepare" };
				tx.setTask({ ...task, abortRequested: true });
				return true;
			},
			context,
			{ conversationId },
		);
		if (admitted) await this.#tasks.waitForTask(taskId, context);
		return admitted;
	}

	retryTask(id: TaskId, incident: EntryId, context: Context): Promise<"queued" | "stale" | "terminal"> {
		return this.#tasks.retry(id, incident, context);
	}

	waitForTask<R>(id: TaskId<R>, context: Context): Promise<SettledTask<R>> {
		this.#tasks.resume();
		return this.#tasks.waitForTask(id, context) as Promise<SettledTask<R>>;
	}

	async waitForIdle(context: Context): Promise<void> {
		this.#tasks.resume();
		return this.#tasks.waitForIdle(undefined, context);
	}

	/** Sum every conversation's committed `pi.usage`. Each document is read at its own point; totals only grow. */
	async usage(context: Context): Promise<UsageState> {
		const conversations = await this.readOnLine(() =>
			scanAll((cursor) => this.storage.scanConversations({}, SCAN_PAGE_SIZE, cursor, context)),
		);
		const total = UsageDoc.definition.initial();
		for (const { id } of conversations) {
			const state = await this.snapshot(UsageDoc, id, context);
			if (state !== undefined) addUsageState(total, state);
		}
		return total;
	}

	taskGraph(context: Context): Promise<AttachedReplicatedState<TaskGraph>> {
		return this.#taskGraph.state(context);
	}

	watchTaskGraph(context: Context): Promise<TaskGraphWatch> {
		return this.#taskGraph.watch(context);
	}

	async observeTaskGraph(observer: Observer<TaskGraph>, context: Context): Promise<Observation<TaskGraph>> {
		this.assertUsable();
		return this.#taskGraph.observe(observer, context);
	}

	async observeConversation(
		id: ConversationId,
		observer: Observer<ConversationView>,
		context: Context,
	): Promise<Observation<ConversationView>> {
		this.assertUsable();
		return this.#host.views.observe(id, observer, context);
	}

	root(
		context: Context,
		options?: { readonly agent?: AgentChange; readonly init?: ConversationInit },
	): Promise<Conversation> {
		return this.#create({ kind: "root" }, options ?? {}, context);
	}

	async conversation(id: ConversationId, context: Context): Promise<Conversation | undefined> {
		this.assertUsable();
		const record = await this.readOnLine(() => this.storage.conversation(id, context));
		return record === undefined ? undefined : new ConversationImpl(record.id, this.#host);
	}

	createConversation(options: ConversationCreateOptions, context: Context): Promise<Conversation> {
		return this.#create({ kind: "independent", ownership: options.ownership }, options, context);
	}

	async importHistory(
		history: ConversationHistory,
		options: ConversationHistoryImportOptions,
		context: Context,
	): Promise<Conversation> {
		const prepared = prepareConversationHistory(history);
		return this.#create(
			{ kind: "history", ownership: options.ownership, history: prepared, initialize: options.init },
			{ agent: options.agent },
			context,
		);
	}

	override report(error: unknown, info: ReportInfo = { type: "listener_error" }, context?: Context): void {
		this.#report(error, info, context);
	}

	/** Signal and join task invocations after the close listeners and before Storage closes; writes no task outcome. */
	protected override beforeClose(): Promise<void> {
		return this.#tasks.join();
	}

	async #create(target: CreateTarget, options: CreateOptions, context: Context): Promise<Conversation> {
		this.assertUsable();
		const id = await this.commitWith(async (tx) => {
			if (target.kind === "root" && (await tx.conversation(ROOT_CONVERSATION_ID)) !== undefined) {
				return ROOT_CONVERSATION_ID;
			}
			const record =
				target.kind === "root"
					? await tx.createRootConversation()
					: target.kind === "fork"
						? await tx.forkConversation(target.parentId, target.at, { ownership: target.ownership })
						: await tx.createConversation({ ownership: target.ownership });
			const entryIds =
				target.kind === "history" ? await importConversationHistory(tx, record.id, target.history) : undefined;
			if (options.agent !== undefined) await configure(tx, record.id, options.agent);
			if (target.kind === "history") await target.initialize?.(tx, record.id, entryIds!);
			if (options.init !== undefined) await options.init(tx, record.id);
			return record.id;
		}, context);
		return new ConversationImpl(id, this.#host);
	}

	/**
	 * The built-in creation hook, in every commit that creates or forks a conversation: empty `pi.live`, `pi.inbox`, and
	 * `pi.usage`, a fresh `pi.provider`, the conversation's `pi.agent` (see `createAgent()`), then
	 * `HarnessOptions.conversationCreated`.
	 */
	protected override async conversationCreated(tx: Transaction, record: ConversationRecord): Promise<void> {
		await tx.doc(LiveDoc, record.id);
		await tx.doc(InboxDoc, record.id);
		await tx.doc(UsageDoc, record.id);
		await tx.doc(ProviderDoc, record.id);
		await createAgent(tx, record);
		await this.#options.conversationCreated?.(tx, record);
	}
}

/**
 * Report to `onReport`, then count it and record its event. Never throws, since reports also come from inside Storage's
 * `adopt`: a throwing `onReport` is dropped, and so is a throwing metric or event, as reporting it would report again.
 */
function safeReport(
	onReport: ((error: unknown, info: ReportInfo) => void) | undefined,
	telemetry: TelemetryContext,
): (error: unknown, info: ReportInfo, context?: Context) => void {
	const reports = createTypedMetrics(telemetry.metrics ?? NOOP_TELEMETRY_METRICS, [DURABLE_TELEMETRY_SCHEMA]).counter(
		"pi.durable.reports",
	);
	return (error, info, context) => {
		contained(
			() => onReport?.(error, info),
			() => {},
		);
		const attributes: SpanAttributes = {
			type: info.type,
		};
		switch (info.type) {
			case "task_faulted":
			case "task_outdated":
				attributes.reason = info.reason;
				attributes.extension = info.extension;
				attributes["task.id"] = info.task.id;
				break;
			case "migration_failed":
			case "task_error":
			case "extension_error":
				attributes.extension = info.extension;
				attributes["task.id"] = info.task?.id;
				break;
			case "host_error":
				attributes["task.id"] = info.task?.id;
				break;
		}
		try {
			reports.add(1, { type: info.type });
			context?.telemetry.addEvent("pi.durable.report", attributes);
		} catch {
			// Dropped: see above.
		}
	};
}

/**
 * The host clock, falling back to `Date.now` when it throws: a broken clock must not stop commits. The first throw is
 * reported; the clock is read in every task change, so later ones are not.
 */
export function safeNow(
	now: (() => number) | undefined,
	report: (error: unknown, info: ReportInfo) => void,
): () => number {
	if (now === undefined) return Date.now;
	let reported = false;
	return () => {
		try {
			return now();
		} catch (error) {
			// Set first: a report handler that reads the clock again must not report again.
			if (!reported) {
				reported = true;
				report(error, { type: "host_error", callback: "now" });
			}
			return Date.now();
		}
	};
}

/**
 * Invocation-bound handle for tasks and tools. Every operation, and every operation of a submission it returns, first
 * checks the invocation and runs under its signal, so it rejects once the invocation ends; admitted work stays durable.
 */
function boundConversation(
	id: ConversationId,
	binding: InvocationBinding,
	submissions: Submissions,
	tasks: TaskScheduler,
): ConversationHandle {
	const bind = (context: Context): Context => withAbortSignal(binding.signal, context);
	const bound = <T>(operation: (context: Context) => Promise<T>) => {
		return async (context: Context): Promise<T> => {
			binding.check();
			return operation(bind(context));
		};
	};
	return {
		id,
		submit: async (draft, context) => {
			binding.check();
			const submission = await submissions.submit(id, draft, bind(context));
			return {
				id: submission.id,
				status: bound((callContext) => submission.status(callContext)),
				wait: bound((callContext) => submission.wait(callContext)),
				abort: bound((callContext) => submission.abort(callContext)),
			};
		},
		abort: async (context, options) => {
			binding.check();
			return tasks.abortConversation(id, options?.background === true, bind(context));
		},
		waitForIdle: bound((callContext) => tasks.waitForIdle(id, callContext)),
	};
}

/** Durable agent harness over one Session. */
export type Harness = HarnessType;

export const Harness = {
	/** Open a Harness over storage. The registry may keep changing while the Harness runs. */
	async open<Tool extends ToolRegistration>(
		storage: Storage,
		options: HarnessOptions<Tool>,
		context: Context,
	): Promise<Harness> {
		context.abortSignal?.throwIfAborted();
		const snapshot = options.registry.snapshot();
		const missing = BUILTIN_TASKS.filter((task) => snapshot.task(task.definition.name) === undefined);
		if (missing.length > 0) {
			const names = missing.map((task) => task.definition.name).join(", ");
			throw new Error(`Registry lacks built-in tasks ${names}; create it with createRegistry()`);
		}
		const harness = new HarnessImpl(storage, options, context);
		try {
			await harness.openTasks(context);
		} catch (error) {
			// The caller's context may be what failed open: close without it, and rethrow the open error.
			// A failing Storage close fails the Session, which reports it.
			await harness.close(withoutAbortSignal(context)).catch(() => {});
			throw error;
		}
		return harness;
	},
};

/** Freeze a detached JSON snapshot, including nested model content and task checkpoints. */
function freezeCommit(value: unknown): void {
	if (typeof value !== "object" || value === null) return;
	for (const child of Object.values(value)) freezeCommit(child);
	Object.freeze(value);
}
