import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { copyJson } from "@earendil-works/chord";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import {
	acceptReceipt,
	bindReceipt,
	defineExtension,
	defineTask,
	type EntryId,
	GenerationTask,
	Harness,
	LiveDoc,
	MemoryStorage,
	type ModelRequestApi,
	type ModelRequestInput,
	type ModelRequestPort,
	type ModelRequestTarget,
	ReceiptDoc,
	type Storage,
	type TaskRuntime,
} from "@earendil-works/pi-durable";
import { describe, expect, it } from "vitest";
import { pinModel } from "../src/harness/json.ts";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, type ChatSetup, chatSetup, waitFor } from "./chat-support.ts";
import { addHooks } from "./harness-support.ts";
import { context } from "./session-support.ts";
import { aborted, deferred, settled } from "./task-support.ts";

async function open(storage: Storage, setup: ChatSetup, modelRequests: ModelRequestPort) {
	const harness = await Harness.open(
		storage,
		{
			models: setup.models,
			registry: setup.registry,
			settings: setup.settings,
			modelRequests,
			now: () => setup.now(),
			onReport: (error) => setup.reports.push(error),
		},
		context,
	);
	const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
	return { harness, root };
}

describe("invocation model request ownership", () => {
	it("preserves dispatch and cleanup failures and rejects acquisition after the invocation ends", async () => {
		const setup = chatSetup();
		const dispatchError = new Error("original dispatch failure");
		const cleanupError = new Error("original cleanup failure");
		let observed: unknown;
		let captured: TaskRuntime<{ cutoff: EntryId }, { phase: "request" }, null, object> | undefined;
		let input: ModelRequestInput | undefined;
		let acquisitions = 0;
		let requestApi: ModelRequestApi | undefined;
		const task = defineTask<{ cutoff: EntryId }, { phase: "request" }, null>({
			name: "app.request",
			version: 1,
			initial: () => ({ phase: "request" }),
			phases: {
				request: async (task, runtime, ctx) => {
					captured = runtime;
					input = {
						purpose: "test",
						operation: "complete",
						attempt: 1,
						model: pinModel(setup.models.getModel("faux", "faux-1")!),
						messages: [],
						cutoff: task.input.cutoff,
						options: {},
					};
					try {
						await runtime.withModelRequest(
							input,
							async () => {
								throw dispatchError;
							},
							ctx,
						);
					} catch (error) {
						observed = error;
					}
					await expect(requestApi!.commit(() => null, ctx)).rejects.toThrow("Model request has ended");
					await runtime.commit(
						() => ({ status: "terminal", outcome: { status: "completed", result: null } }),
						ctx,
					);
				},
			},
			abort: async (_task, runtime, ctx) => {
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		setup.registry.install(defineExtension({ name: "request-test", tasks: [task] }));
		const opened = await open(new MemoryStorage(), setup, async (_request, api) => {
			acquisitions++;
			requestApi = api;
			return {
				status: "ready",
				options: {},
				close: async () => {
					throw cleanupError;
				},
			};
		});
		try {
			const id = await opened.root.commit(async (tx) => {
				const entry = await tx.appendEntry(opened.root.id, { kind: "app.request" });
				return tx.createTask(task, { cutoff: entry.id }, { ownership: { kind: "conversation" } });
			}, context);
			await opened.harness.runPass(context);
			expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
			expect(observed).toBeInstanceOf(AggregateError);
			expect((observed as AggregateError).errors).toEqual([dispatchError, cleanupError]);
			await expect(captured!.withModelRequest(input!, async () => null, context)).rejects.toThrow(
				"invocation has ended",
			);
			expect(acquisitions).toBe(1);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("refuses premature settlement and joins an unawaited owned request during shutdown", async () => {
		const setup = chatSetup();
		const cleanupStarted = deferred();
		const release = deferred();
		const task = defineTask<{ cutoff: EntryId }, { phase: "request" }, null>({
			name: "app.detached",
			version: 1,
			initial: () => ({ phase: "request" }),
			phases: {
				request: async (task, runtime, ctx) => {
					void runtime.withModelRequest(
						{
							purpose: "test",
							operation: "complete",
							attempt: 1,
							model: pinModel(setup.models.getModel("faux", "faux-1")!),
							messages: [],
							cutoff: task.input.cutoff,
							options: {},
						},
						async (_options, signal) => aborted(signal),
						ctx,
					);
					await runtime.commit(
						() => ({ status: "terminal", outcome: { status: "completed", result: null } }),
						ctx,
					);
				},
			},
			abort: async (_task, runtime, ctx) => {
				await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), ctx);
			},
		});
		setup.registry.install(defineExtension({ name: "detached-test", tasks: [task] }));
		const opened = await open(new MemoryStorage(), setup, async () => ({
			status: "ready",
			options: {},
			close: async () => {
				cleanupStarted.resolve();
				await release.promise;
			},
		}));
		try {
			const id = await opened.root.commit(async (tx) => {
				const entry = await tx.appendEntry(opened.root.id, { kind: "app.request" });
				return tx.createTask(task, { cutoff: entry.id }, { ownership: { kind: "conversation" } });
			}, context);
			opened.harness.resume();
			await cleanupStarted.promise;
			expect((await opened.harness.waitForTask(id, context)).state.outcome).toMatchObject({
				status: "faulted",
				error: { message: expect.stringContaining("cannot settle or suspend") },
			});
			const closing = opened.harness.close(context);
			expect(await settled(closing)).toBe(false);
			release.resolve();
			await closing;
		} finally {
			release.resolve();
			await opened.harness.close(context);
		}
	});

	it("stamps kernel identity, forwards live capabilities through Models and joins cleanup before classification", async () => {
		const setup = chatSetup();
		setup.settings.stream = { headers: { "x-pinned": "original" } };
		const cleanupStarted = deferred();
		const release = deferred();
		const requests: ModelRequestTarget[] = [];
		let classified = 0;
		addHooks(setup.registry, GenerationTask, {
			afterResponse: () => {
				classified++;
			},
		});
		const fetch: typeof globalThis.fetch = async () => new Response("owned");
		setup.faux.setResponses([
			(_messages, options) => {
				expect(options).toMatchObject({
					apiKey: "request-token",
					authType: "oauth",
					sessionId: "owned-session",
					fetch,
				});
				expect(options?.headers).toEqual({ "x-pinned": "original", "x-owned": "true" });
				expect(options?.signal).toBeInstanceOf(AbortSignal);
				return fauxAssistantMessage("done");
			},
		]);
		const opened = await open(new MemoryStorage(), setup, async (request) => {
			requests.push(request);
			return {
				status: "ready",
				options: {
					apiKey: "request-token",
					authType: "oauth",
					fetch,
					sessionId: "owned-session",
					transformHeaders: (headers) => ({ ...headers, "x-owned": "true" }),
				},
				close: async (ctx) => {
					expect(ctx.abortSignal).toBeUndefined();
					cleanupStarted.resolve();
					await release.promise;
				},
			};
		});
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await cleanupStarted.promise;
			const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
			expect(requests).toHaveLength(1);
			expect(requests[0]).toMatchObject({
				taskId,
				conversationId: opened.root.id,
				taskKind: "pi.generation",
				taskVersion: GenerationTask.definition.version,
				purpose: "generation",
				operation: "stream",
				attempt: 1,
				model: { provider: "faux", id: "faux-1" },
				options: { headers: { "x-pinned": "original" } },
			});
			expect(JSON.stringify(requests[0])).not.toContain("request-token");
			expect(classified).toBe(0);
			expect((await opened.harness.getTask(taskId, context))?.state.status).toBe("running");
			release.resolve();
			expect((await submission.wait(context)).status).toBe("done");
			expect(classified).toBe(1);
		} finally {
			release.resolve();
			await opened.harness.close(context);
		}
	});

	it("joins summary transport before committing a compaction result", async () => {
		const setup = chatSetup();
		setup.settings.compaction = { enabled: false, keepRecentTokens: 1 };
		setup.faux.setResponses([fauxAssistantMessage("summary")]);
		const cleanupStarted = deferred();
		const release = deferred();
		const requests: ModelRequestTarget[] = [];
		const opened = await open(new MemoryStorage(), setup, async (request) => {
			requests.push(request);
			return {
				status: "ready",
				options: {},
				close: async () => {
					cleanupStarted.resolve();
					await release.promise;
				},
			};
		});
		try {
			await opened.root.commit(async (tx) => {
				for (const content of ["summarize this history", "keep"])
					await tx.appendEntry(opened.root.id, {
						kind: "app.history",
						model: [{ role: "user", content, timestamp: 1 }],
					});
			}, context);
			const id = await opened.root.compact(undefined, context);
			await cleanupStarted.promise;
			expect(requests[0]).toMatchObject({
				taskId: id,
				taskKind: "pi.compaction",
				purpose: "compaction",
				operation: "complete",
			});
			expect((await opened.harness.getTask(id, context))?.state.status).toBe("running");
			expect((await allEntries(opened.root)).some((entry) => entry.kind === "pi.compaction")).toBe(false);
			release.resolve();
			expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
		} finally {
			release.resolve();
			await opened.harness.close(context);
		}
	});

	it("keeps Harness close pending until a cancelled dispatch and its cleanup have joined", async () => {
		const setup = chatSetup();
		const dispatched = deferred();
		const cleanupStarted = deferred();
		const release = deferred();
		setup.faux.setResponses([
			async (_request, options) => {
				dispatched.resolve();
				return aborted(options!.signal!);
			},
		]);
		const opened = await open(new MemoryStorage(), setup, async () => ({
			status: "ready",
			options: {},
			close: async (ctx) => {
				expect(ctx.abortSignal).toBeUndefined();
				cleanupStarted.resolve();
				await release.promise;
			},
		}));
		try {
			await opened.root.submit({ type: "input", content: "go" }, context);
			await dispatched.promise;
			const closing = opened.harness.close(context);
			await cleanupStarted.promise;
			expect(await settled(closing)).toBe(false);
			release.resolve();
			await closing;
		} finally {
			release.resolve();
			await opened.harness.close(context);
		}
	});

	it("closes late acquired capabilities after shutdown without dispatching them", async () => {
		const setup = chatSetup();
		const acquiring = deferred();
		const cancelled = deferred();
		const acquired = deferred();
		let closed = 0;
		let requestApi: ModelRequestApi | undefined;
		setup.faux.setResponses([fauxAssistantMessage("must not dispatch")]);
		const opened = await open(new MemoryStorage(), setup, async (_request, api, ctx) => {
			requestApi = api;
			acquiring.resolve();
			void aborted(ctx.abortSignal!).catch(() => cancelled.resolve());
			await acquired.promise;
			return {
				status: "ready",
				options: {},
				close: async () => {
					closed++;
				},
			};
		});
		try {
			await opened.root.submit({ type: "input", content: "go" }, context);
			await acquiring.promise;
			const closing = opened.harness.close(context);
			await cancelled.promise;
			expect(await settled(closing)).toBe(false);
			let mutated = false;
			await expect(
				requestApi!.commit(() => {
					mutated = true;
				}, context),
			).rejects.toThrow();
			expect(mutated).toBe(false);
			acquired.resolve();
			await closing;
			expect(closed).toBe(1);
			expect(setup.faux.state.callCount).toBe(0);
		} finally {
			acquired.resolve();
			await opened.harness.close(context);
		}
	});

	it("binds an approval receipt through the invoking task and resumes the same request after SQLite replacement", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-model-receipt-"));
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		const requests: ModelRequestTarget[] = [];
		const apis: ModelRequestApi[] = [];
		const binding = "sealed-acquisition";
		const port: ModelRequestPort = async (request, api, ctx) => {
			requests.push(request);
			apis.push(api);
			const key = `approval:${request.taskId}`;
			const result = await api.commit(async (tx) => {
				const existing = await tx.doc(ReceiptDoc, key, null);
				await bindReceipt(tx, key, binding);
				return existing.result === undefined ? undefined : copyJson(existing.result);
			}, ctx);
			if (result !== undefined) {
				expect(result).toEqual({ decision: "approved" });
				return { status: "ready", options: {}, close: async () => {} };
			}
			return { status: "waiting", condition: { kind: "receipt", key, binding } };
		};
		setup.faux.setResponses([fauxAssistantMessage("done")]);
		let opened = await open(await openNodeSqliteStorage(path), setup, port);
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await opened.harness.runPass(context);
			const original = copyJson(requests[0]!);
			const taskId = requests[0]!.taskId;
			const key = `approval:${taskId}`;
			expect(await opened.harness.snapshot(ReceiptDoc, key, context)).toEqual({ admitted: true, binding });
			expect((await opened.harness.getTask(taskId, context))?.state).toMatchObject({
				status: "waiting",
				checkpoint: { phase: "request" },
				condition: { kind: "receipt", key, binding },
			});
			expect(setup.faux.state.callCount).toBe(0);
			await expect(apis[0]!.commit(() => null, context)).rejects.toThrow("Model request has ended");
			await opened.harness.close(context);
			opened = await open(await openNodeSqliteStorage(path), setup, port);
			await opened.harness.runPass(context);
			expect(requests).toHaveLength(1);
			await opened.harness.commit((tx) => acceptReceipt(tx, key, binding, { decision: "approved" }), context);
			expect((await (await opened.harness.submission(submission.id, context))!.wait(context)).status).toBe("done");
			expect(requests).toEqual([original, original]);
			expect(setup.faux.state.callCount).toBe(1);
			await expect(apis[1]!.commit(() => null, context)).rejects.toThrow("Model request has ended");
		} finally {
			await opened.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});

	it("retains a terminal receipt arriving before parking and attributes port writes to the invoking task", async () => {
		const setup = chatSetup();
		let requests = 0;
		let target: ModelRequestTarget | undefined;
		const opened = await open(new MemoryStorage(), setup, async (request, api, ctx) => {
			target = request;
			requests++;
			const key = `approval:${request.taskId}`;
			const binding = "approved-acquisition";
			const ready = await api.commit(async (tx) => {
				const existing = await tx.doc(ReceiptDoc, key, null);
				if (existing?.result !== undefined) return true;
				await bindReceipt(tx, key, binding);
				await acceptReceipt(tx, key, binding, null);
				await tx.appendEntry(request.conversationId, { kind: "app.approval" });
				return false;
			}, ctx);
			return ready
				? { status: "ready", options: {}, close: async () => {} }
				: { status: "waiting", condition: { kind: "receipt", key, binding } };
		});
		setup.faux.setResponses([fauxAssistantMessage("done")]);
		try {
			expect(
				(await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context)).status,
			).toBe("done");
			expect(requests).toBe(2);
			expect(setup.faux.state.callCount).toBe(1);
			const entry = (await allEntries(opened.root)).find((entry) => entry.kind === "app.approval");
			expect(entry).toMatchObject({ conversationId: target!.conversationId, byTaskId: target!.taskId });
		} finally {
			await opened.harness.close(context);
		}
	});

	it("propagates a port commit failure without retaining partial admission or dispatching the provider", async () => {
		const setup = chatSetup();
		const failure = new Error("original approval admission failure");
		let observed: unknown;
		const opened = await open(new MemoryStorage(), setup, async (_request, api, ctx) => {
			try {
				await api.commit(async (tx) => {
					await bindReceipt(tx, "rejected-admission", "binding");
					throw failure;
				}, ctx);
			} catch (error) {
				observed = error;
				throw error;
			}
			throw new Error("unreachable");
		});
		try {
			expect(
				await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context),
			).toMatchObject({
				status: "unanswered",
				reason: "faulted",
				detail: failure.message,
			});
			expect(observed).toBe(failure);
			expect(await opened.harness.snapshot(ReceiptDoc, "rejected-admission", context)).toBeUndefined();
			expect(setup.faux.state.callCount).toBe(0);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("refuses a queued port commit once its request has ended without mutating a still-running task", async () => {
		const setup = chatSetup();
		const release = deferred();
		const closed = deferred();
		let api: ModelRequestApi | undefined;
		let queued: Promise<void> | undefined;
		let blocker: Promise<void> | undefined;
		const opened = await open(new MemoryStorage(), setup, async (_request, requestApi) => {
			api = requestApi;
			return { status: "ready", options: {}, close: async () => closed.resolve() };
		});
		setup.faux.setResponses([
			() => {
				blocker = opened.harness.commit(async () => release.promise, context);
				queued = api!.commit((tx) => bindReceipt(tx, "late-port-commit", "binding"), context);
				void queued.catch(() => {});
				return fauxAssistantMessage("done");
			},
		]);
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await closed.promise;
			expect(await settled(queued!)).toBe(false);
			release.resolve();
			await blocker;
			await expect(queued).rejects.toThrow("Model request has ended");
			expect((await submission.wait(context)).status).toBe("done");
			expect(await opened.harness.snapshot(ReceiptDoc, "late-port-commit", context)).toBeUndefined();
		} finally {
			release.resolve();
			await blocker;
			await opened.harness.close(context);
		}
	});

	it("refuses a conflicting approval binding without replacing the admitted receipt", async () => {
		const setup = chatSetup();
		const opened = await open(new MemoryStorage(), setup, async (_request, api, ctx) => {
			await api.commit((tx) => bindReceipt(tx, "approval", "changed-binding"), ctx);
			return { status: "waiting", condition: { kind: "receipt", key: "approval", binding: "changed-binding" } };
		});
		try {
			await opened.harness.commit((tx) => bindReceipt(tx, "approval", "original-binding"), context);
			expect(
				await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context),
			).toMatchObject({
				status: "unanswered",
				reason: "faulted",
				detail: "Receipt identity conflicts with its admission",
			});
			expect(await opened.harness.snapshot(ReceiptDoc, "approval", context)).toEqual({
				admitted: true,
				binding: "original-binding",
			});
			expect(setup.faux.state.callCount).toBe(0);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("faults rather than accepting a response whose transport cleanup failed", async () => {
		const setup = chatSetup();
		let classified = false;
		addHooks(setup.registry, GenerationTask, {
			afterResponse: () => {
				classified = true;
			},
		});
		setup.faux.setResponses([fauxAssistantMessage("not yet safe")]);
		const opened = await open(new MemoryStorage(), setup, async () => ({
			status: "ready",
			options: {},
			close: async () => {
				throw new Error("owned transport remains uncertain");
			},
		}));
		try {
			expect(
				await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context),
			).toMatchObject({
				status: "unanswered",
				reason: "faulted",
				detail: "owned transport remains uncertain",
			});
			expect(classified).toBe(false);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("parks access durably and reuses committed input after SQLite replacement", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-model-access-"));
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		let hooks = 0;
		addHooks(setup.registry, GenerationTask, {
			beforeRequest: ({ messages }) => {
				hooks++;
				return { messages: [...messages, { role: "user", content: "committed hook", timestamp: 123 }] };
			},
		});
		const requests: ModelRequestTarget[] = [];
		let ready = false;
		const port: ModelRequestPort = async (request) => {
			requests.push(request);
			if (!ready)
				return {
					status: "waiting",
					condition: {
						kind: "input",
						conversationId: request.conversationId,
						after: request.cutoff,
						kinds: ["app.credential-ready"],
					},
				};
			return { status: "ready", options: { apiKey: "fresh-token" }, close: async () => {} };
		};
		setup.faux.setResponses([fauxAssistantMessage("done")]);
		let opened = await open(await openNodeSqliteStorage(path), setup, port);
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await opened.harness.runPass(context);
			const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
			expect((await opened.harness.getTask(taskId, context))?.state).toMatchObject({
				status: "waiting",
				mode: "run",
				checkpoint: { phase: "request" },
				condition: { kind: "input", kinds: ["app.credential-ready"] },
			});
			expect(setup.faux.state.callCount).toBe(0);
			const original = copyJson(requests[0]!);
			await opened.harness.close(context);
			setup.models.getModel("faux", "faux-1")!.baseUrl = "https://changed.example";
			opened = await open(await openNodeSqliteStorage(path), setup, port);
			ready = true;
			await opened.root.commit(
				(tx) => tx.appendEntry(opened.root.id, { kind: "app.credential-ready" }).then(() => {}),
				context,
			);
			opened.harness.resume();
			expect((await (await opened.harness.submission(submission.id, context))!.wait(context)).status).toBe("done");
			expect(hooks).toBe(1);
			expect(requests).toHaveLength(2);
			expect(requests[1]).toEqual(original);
		} finally {
			await opened.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});

	it("retains deferred cancellation intent and its handle while access is pending", async () => {
		const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
		setup.settings.stream = { deferred: true };
		setup.faux.setResponses([fauxAssistantMessage("done")]);
		const pending = deferred();
		let ready = false;
		const requests: ModelRequestTarget[] = [];
		const closed: string[] = [];
		const opened = await open(new MemoryStorage(), setup, async (request) => {
			requests.push(request);
			if (request.operation === "cancelDeferred" && !ready) {
				pending.resolve();
				return {
					status: "waiting",
					condition: {
						kind: "input",
						conversationId: request.conversationId,
						after: request.cutoff,
						kinds: ["app.credential-ready"],
					},
				};
			}
			return {
				status: "ready",
				options: {},
				close: async () => {
					closed.push(request.operation);
				},
			};
		});
		let aborting: Promise<void> | undefined;
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await waitFor(
				async () =>
					(await opened.harness.snapshot(LiveDoc, opened.root.id, context))?.generation?.deferred !== undefined,
			);
			aborting = opened.root.abort(context);
			await pending.promise;
			await waitFor(async () => {
				const task = await opened.harness.getTask(requests[0]!.taskId, context);
				return task?.state.status === "waiting" && task.state.mode === "abort";
			});
			expect(await settled(aborting)).toBe(false);
			expect(await opened.harness.getTask(requests[0]!.taskId, context)).toMatchObject({
				abortRequested: true,
				state: { status: "waiting", mode: "abort", checkpoint: { phase: "poll", handle: expect.any(Object) } },
			});
			expect(closed).toEqual(["stream"]);
			ready = true;
			await opened.root.commit(async (tx) => {
				await tx.appendEntry(opened.root.id, { kind: "app.credential-ready" });
			}, context);
			await aborting;
			expect(await submission.wait(context)).toMatchObject({ status: "unanswered", reason: "aborted" });
			expect(requests.map((request) => request.operation)).toEqual(["stream", "cancelDeferred", "cancelDeferred"]);
			expect(requests[2]).toEqual(requests[1]);
			expect(closed).toEqual(["stream", "cancelDeferred"]);
		} finally {
			// Observe a close rejection if the assertion failed while cancellation was still waiting.
			void aborting?.catch(() => {});
			await opened.harness.close(context);
		}
	});

	for (const operation of ["fetchDeferred", "cancelDeferred"] as const) {
		it(`acquires and joins a separate connection for ${operation}`, async () => {
			const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
			setup.settings.stream = { deferred: true };
			let now = 1000;
			setup.now = () => now;
			setup.faux.setResponses([fauxAssistantMessage("done")]);
			const requests: ModelRequestTarget[] = [];
			const closed: string[] = [];
			const opened = await open(new MemoryStorage(), setup, async (request) => {
				requests.push(request);
				return {
					status: "ready",
					options: {},
					close: async () => {
						closed.push(request.operation);
					},
				};
			});
			try {
				const submission = await opened.root.submit({ type: "input", content: "go" }, context);
				await waitFor(
					async () =>
						(await opened.harness.snapshot(LiveDoc, opened.root.id, context))?.generation?.deferred !== undefined,
				);
				if (operation === "cancelDeferred") await opened.root.abort(context);
				else {
					now = 61_000;
					opened.harness.resume();
				}
				expect((await submission.wait(context)).status).toBe(
					operation === "cancelDeferred" ? "unanswered" : "done",
				);
				expect(requests.map((request) => request.operation)).toEqual(["stream", operation]);
				expect(closed).toEqual(["stream", operation]);
				expect(requests[1]).toMatchObject({
					taskId: requests[0]!.taskId,
					model: requests[0]!.model,
					messages: requests[0]!.messages,
				});
			} finally {
				await opened.harness.close(context);
			}
		});
	}
});
