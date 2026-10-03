import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { copyJson } from "@earendil-works/chord";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import {
	Harness,
	LiveDoc,
	MemoryStorage,
	type ModelRequestApi,
	type ModelRequestPort,
	type ModelRequestTarget,
	type Storage,
	StorageRejected,
	type StorageWrite,
} from "@earendil-works/pi-durable";
import { describe, expect, it, vi } from "vitest";
import { admitModelRequest } from "../src/harness/model-request.ts";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { type ChatSetup, chatSetup, waitFor } from "./chat-support.ts";
import { context } from "./session-support.ts";
import { aborted, deferred, settled } from "./task-support.ts";

const endpoint = "http://127.0.0.1:32123/v1";

async function open(storage: Storage, setup: ChatSetup, port: ModelRequestPort) {
	const harness = await Harness.open(
		storage,
		{
			models: setup.models,
			registry: setup.registry,
			settings: setup.settings,
			modelRequests: port,
			now: () => setup.now(),
			onReport: (error) => setup.reports.push(error),
		},
		context,
	);
	const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
	return { harness, root };
}

describe("durable model endpoint preparation", () => {
	it("commits the effective endpoint before dispatch while preserving original intent and catalog metadata", async () => {
		const setup = chatSetup();
		const original = copyJson(setup.models.getModel("faux", "faux-1")!, { omitUndefinedProperties: true });
		let api: ModelRequestApi | undefined;
		let request: ModelRequestTarget | undefined;
		setup.faux.setResponses([
			(_messages, _options, _state, model) => {
				expect(model).toEqual({ ...(original as object), baseUrl: endpoint });
				expect(api!.prepared!.model).toEqual(model);
				return fauxAssistantMessage("done");
			},
		]);
		const opened = await open(new MemoryStorage(), setup, async (target, boundary, ctx) => {
			request = target;
			api = boundary;
			expect(api.prepared).toBeUndefined();
			const effective = { ...target.model, baseUrl: endpoint };
			const prepared = await api.prepare(effective, ctx);
			expect(await api.prepare(effective, ctx)).toEqual(prepared);
			expect(prepared).toEqual({ ...target, model: effective });
			// Each read and return is detached from the private committed model.
			prepared.model.baseUrl = "https://mutated-return.test";
			api.prepared!.model.baseUrl = "https://mutated-accessor.test";
			effective.baseUrl = "https://mutated-candidate.test";
			expect(api.prepared!.model.baseUrl).toBe(endpoint);
			return { status: "ready", options: {}, close: async () => {} };
		});
		try {
			expect(
				(await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context)).status,
			).toBe("done");
			expect(request!.model).toEqual(original);
			expect(setup.models.getModel("faux", "faux-1")).toEqual(original);
			await expect(api!.prepare({ ...request!.model, baseUrl: endpoint }, context)).rejects.toThrow(
				"Model request has ended",
			);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("uses the committed effective model for compaction", async () => {
		const setup = chatSetup();
		setup.settings.compaction = { enabled: false, keepRecentTokens: 1 };
		const dispatched: string[] = [];
		setup.faux.setResponses([
			(_messages, _options, _state, model) => {
				dispatched.push(model.baseUrl);
				return fauxAssistantMessage("summary");
			},
		]);
		const opened = await open(new MemoryStorage(), setup, async (target, api, ctx) => {
			expect(target.operation).toBe("complete");
			await api.prepare({ ...target.model, baseUrl: endpoint }, ctx);
			return { status: "ready", options: {}, close: async () => {} };
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
			expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
			expect(dispatched).toEqual([endpoint]);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("reopens the same prepared model and original input after SQLite replacement", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-model-preparation-"));
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		const requests: ModelRequestTarget[] = [];
		let preparations = 0;
		let ready = false;
		setup.faux.setResponses([
			(_messages, _options, _state, model) => {
				expect(model.baseUrl).toBe(endpoint);
				return fauxAssistantMessage("done");
			},
		]);
		const port: ModelRequestPort = async (target, api, ctx) => {
			requests.push(target);
			if (api.prepared === undefined) {
				preparations++;
				await api.prepare({ ...target.model, baseUrl: endpoint }, ctx);
			}
			expect(api.prepared!.model.baseUrl).toBe(endpoint);
			return ready
				? { status: "ready", options: {}, close: async () => {} }
				: {
						status: "waiting",
						condition: {
							kind: "input",
							conversationId: target.conversationId,
							after: target.cutoff,
							kinds: ["app.model-ready"],
						},
					};
		};
		let opened = await open(await openNodeSqliteStorage(path), setup, port);
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await opened.harness.runPass(context);
			const original = copyJson(requests[0]);
			await opened.harness.close(context);
			setup.models.getModel("faux", "faux-1")!.baseUrl = "https://changed-catalog.test";
			opened = await open(await openNodeSqliteStorage(path), setup, port);
			ready = true;
			await opened.root.commit(async (tx) => {
				await tx.appendEntry(opened.root.id, { kind: "app.model-ready" });
			}, context);
			opened.harness.resume();
			expect((await (await opened.harness.submission(submission.id, context))!.wait(context)).status).toBe("done");
			expect(requests).toEqual([original, original]);
			expect(preparations).toBe(1);
			expect(setup.faux.state.callCount).toBe(1);
		} finally {
			await opened.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});

	for (const field of ["provider", "id", "api", "contextWindow"] as const) {
		it(`refuses changing ${field} during endpoint preparation`, async () => {
			const setup = chatSetup();
			const opened = await open(new MemoryStorage(), setup, async (target, api, ctx) => {
				const model = { ...target.model, [field]: field === "contextWindow" ? 17 : "foreign" };
				await expect(api.prepare(model, ctx)).rejects.toThrow("cannot change the selected model");
				expect(api.prepared).toBeUndefined();
				await api.prepare({ ...target.model, baseUrl: endpoint }, ctx);
				await expect(api.prepare({ ...target.model, baseUrl: "https://replacement.test" }, ctx)).rejects.toThrow(
					"committed endpoint",
				);
				expect(api.prepared!.model.baseUrl).toBe(endpoint);
				return { status: "ready", options: {}, close: async () => {} };
			});
			setup.faux.setResponses([fauxAssistantMessage("done")]);
			try {
				expect(
					(await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context)).status,
				).toBe("done");
			} finally {
				await opened.harness.close(context);
			}
		});
	}

	it("propagates an authoritative rejected prepare commit and retains no partial effective model", async () => {
		class FailingStorage extends MemoryStorage {
			failure: Error | undefined;
			override async commit(writes: readonly StorageWrite[], ctx: Parameters<Storage["commit"]>[1]) {
				if (
					this.failure &&
					writes.some((write) => {
						if (
							(write.type !== "document.create" && write.type !== "document.change") ||
							write.content.kind !== "base"
						)
							return false;
						const model = write.content.value.model;
						return (
							model !== null && typeof model === "object" && !Array.isArray(model) && model.baseUrl === endpoint
						);
					})
				) {
					const error = this.failure;
					this.failure = undefined;
					throw error;
				}
				return super.commit(writes, ctx);
			}
		}
		const setup = chatSetup();
		const storage = new FailingStorage();
		const failure = new StorageRejected("original model preparation storage refusal");
		setup.faux.setResponses([fauxAssistantMessage("done")]);
		const opened = await open(storage, setup, async (target, api, ctx) => {
			storage.failure = failure;
			await expect(api.prepare({ ...target.model, baseUrl: endpoint }, ctx)).rejects.toBe(failure);
			expect(api.prepared).toBeUndefined();
			await api.prepare({ ...target.model, baseUrl: endpoint }, ctx);
			return { status: "ready", options: {}, close: async () => {} };
		});
		try {
			expect(
				(await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context)).status,
			).toBe("done");
			expect(setup.faux.state.callCount).toBe(1);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("rejects a preparation queued after the request ends without replacing committed identity", async () => {
		const setup = chatSetup();
		const release = deferred();
		const closed = deferred();
		let api: ModelRequestApi | undefined;
		let queued: Promise<ModelRequestTarget> | undefined;
		let blocker: Promise<void> | undefined;
		const opened = await open(new MemoryStorage(), setup, async (target, boundary, ctx) => {
			api = boundary;
			await api.prepare({ ...target.model, baseUrl: endpoint }, ctx);
			return { status: "ready", options: {}, close: async () => closed.resolve() };
		});
		setup.faux.setResponses([
			() => {
				blocker = opened.harness.commit(async () => release.promise, context);
				queued = api!.prepare({ ...api!.prepared!.model, baseUrl: "https://late.test" }, context);
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
			expect(api!.prepared!.model.baseUrl).toBe(endpoint);
			expect((await submission.wait(context)).status).toBe("done");
		} finally {
			release.resolve();
			await blocker;
			await opened.harness.close(context);
		}
	});

	it("fences late preparation after authoritative shutdown and joins late acquired cleanup", async () => {
		const setup = chatSetup();
		const acquiring = deferred();
		const cancelled = deferred();
		const acquired = deferred();
		let api: ModelRequestApi | undefined;
		let target: ModelRequestTarget | undefined;
		let closed = 0;
		const opened = await open(new MemoryStorage(), setup, async (request, boundary, ctx) => {
			api = boundary;
			target = request;
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
			await expect(api!.prepare({ ...target!.model, baseUrl: endpoint }, context)).rejects.toThrow();
			expect(api!.prepared).toBeUndefined();
			expect(await settled(closing)).toBe(false);
			acquired.resolve();
			await closing;
			expect(closed).toBe(1);
			expect(setup.faux.state.callCount).toBe(0);
		} finally {
			acquired.resolve();
			await opened.harness.close(context);
		}
	});

	for (const operation of ["fetchDeferred", "cancelDeferred"] as const) {
		it(`reuses stream preparation for ${operation} instead of resolving a new endpoint`, async () => {
			const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
			setup.settings.stream = { deferred: true };
			let now = 1000;
			setup.now = () => now;
			setup.faux.setResponses([fauxAssistantMessage("done")]);
			const observe = vi.spyOn(setup.models, operation);
			const calls: string[] = [];
			const opened = await open(new MemoryStorage(), setup, async (target, api, ctx) => {
				calls.push(target.operation);
				if (target.operation === "stream") await api.prepare({ ...target.model, baseUrl: endpoint }, ctx);
				else {
					expect(api.prepared!.model.baseUrl).toBe(endpoint);
					await expect(
						api.commit(
							(tx) => admitModelRequest(tx, { ...target, options: { ...target.options, maxTokens: 99 } }),
							ctx,
						),
					).rejects.toThrow("Model operation conflicts with its original request");
					if (target.operation === "fetchDeferred" || target.operation === "cancelDeferred") {
						await expect(
							api.commit(
								(tx) =>
									admitModelRequest(tx, {
										...target,
										handle: { ...target.handle, id: "foreign-request" },
									}),
								ctx,
							),
						).rejects.toThrow("Model operation conflicts with its original request");
					}
					await expect(
						api.commit(
							(tx) =>
								admitModelRequest(tx, {
									...target,
									messages: [{ role: "user", content: "foreign prompt", timestamp: 2 }],
								}),
							ctx,
						),
					).rejects.toThrow("Model request conflicts with its original intent");
				}
				return { status: "ready", options: {}, close: async () => {} };
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
				expect(calls).toEqual(["stream", operation]);
				expect(observe.mock.calls[0]![0].baseUrl).toBe(endpoint);
			} finally {
				observe.mockRestore();
				await opened.harness.close(context);
			}
		});
	}
});
