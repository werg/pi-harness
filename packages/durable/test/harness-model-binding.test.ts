import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { copyJson, type JsonValue } from "@earendil-works/chord";
import { type Api, fauxAssistantMessage, type Model } from "@earendil-works/pi-ai";
import {
	CompactionTask,
	GenerationTask,
	LiveDoc,
	MemoryStorage,
	StorageRejected,
	type Task,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, type ChatSetup, chatSetup, openChat, waitFor } from "./chat-support.ts";
import { addHooks, addSection } from "./harness-support.ts";
import { ControlledStorage, context } from "./session-support.ts";
import { aborted, deferred } from "./task-support.ts";

const directories = new Set<string>();
afterEach(async () => {
	for (const directory of directories) await rm(directory, { recursive: true, force: true });
	directories.clear();
});

async function sqlitePath(): Promise<string> {
	const directory = await mkdtemp(join(tmpdir(), "pi-model-binding-"));
	directories.add(directory);
	return join(directory, "session.sqlite");
}

function catalogModel(setup: ChatSetup): Model<Api> {
	const model = setup.models.getModel("faux", "faux-1");
	if (model === undefined) throw new Error("Missing test model");
	return model;
}

describe("committed model request binding", () => {
	it("replays the actual hook input and full descriptor after catalog, hook and settings changes", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-model-binding-"));
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		const descriptor = catalogModel(setup);
		descriptor.baseUrl = "https://original.example/api";
		descriptor.samplingParams = { nested: { custom: [1, true, null] } };
		descriptor.inputLimits = { images: { maxPerRequest: 4 } };
		descriptor.headers = { "x-model-route": "original" };
		descriptor.cost.tiers = [{ ...descriptor.cost, inputTokensAbove: 1000 }];
		const pinned = copyJson(descriptor);
		let hooks = 0;
		addHooks(
			setup.registry,
			GenerationTask,
			{
				beforeRequest: ({ messages }) => {
					hooks++;
					return {
						messages: [
							...messages,
							fauxAssistantMessage([
								{ type: "thinking", thinking: "opaque", thinkingSignature: "replay-secret-shape" },
								{ type: "text", text: "prepared", textSignature: "provider-item-identity" },
							]),
						],
					};
				},
			},
			"binding-hook",
		);
		const reached = deferred();
		const sent: JsonValue[] = [];
		setup.faux.setResponses([
			async (request, options, _state, model) => {
				sent.push(copyJson({ request, model, headers: options?.headers }, { omitUndefinedProperties: true }));
				reached.resolve();
				return aborted(options!.signal!);
			},
			(request, options, _state, model) => {
				sent.push(copyJson({ request, model, headers: options?.headers }, { omitUndefinedProperties: true }));
				return fauxAssistantMessage("done");
			},
		]);
		setup.settings.stream = { headers: { "x-request-route": "original" } };
		let opened = await openChat(await openNodeSqliteStorage(path), setup);
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await reached.promise;
			const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.run!.taskId;
			const task = await opened.harness.getTask(taskId, context);
			expect(task?.state).toMatchObject({ checkpoint: { phase: "request", model: pinned } });
			await opened.harness.close(context);
			descriptor.baseUrl = "https://replacement.example/api";
			descriptor.samplingParams = { changed: true };
			descriptor.headers!["x-model-route"] = "replacement";
			setup.faux.provider.getModels = () => [];
			setup.settings.stream = { headers: { "x-request-route": "replacement" } };
			addHooks(
				setup.registry,
				GenerationTask,
				{
					beforeRequest: () => {
						hooks++;
						return { messages: [{ role: "user", content: "changed hook", timestamp: 99 }] };
					},
				},
				"binding-hook",
			);
			opened = await openChat(await openNodeSqliteStorage(path), setup);
			opened.harness.resume();
			expect((await (await opened.harness.submission(submission.id, context))!.wait(context)).status).toBe("done");
			expect(hooks).toBe(1);
			expect(sent).toHaveLength(2);
			expect(sent[1]).toEqual(sent[0]);
			expect(sent[0]).toMatchObject({ model: pinned, headers: { "x-request-route": "original" } });
		} finally {
			await opened.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});

	it("restarts interrupted binding without rerendering preparation or retargeting its model", async () => {
		const path = await sqlitePath();
		const setup = chatSetup();
		let rendered = 0;
		addSection(setup.registry, "fixed", () => {
			rendered++;
			return "prepared";
		});
		const reached = deferred();
		let hooks = 0;
		addHooks(setup.registry, GenerationTask, {
			beforeRequest: async ({ messages }, _runtime, ctx) => {
				if (++hooks === 1) {
					reached.resolve();
					await aborted(ctx.abortSignal!);
				}
				return { messages };
			},
		});
		const original = catalogModel(setup).baseUrl;
		const endpoints: string[] = [];
		setup.faux.setResponses([
			(_request, _options, _state, model) => {
				endpoints.push(model!.baseUrl);
				return fauxAssistantMessage("done");
			},
		]);
		let opened = await openChat(await openNodeSqliteStorage(path), setup);
		try {
			const submission = await opened.root.submit({ type: "input", content: "go" }, context);
			await reached.promise;
			await opened.harness.close(context);
			catalogModel(setup).baseUrl = "https://replacement.example";
			opened = await openChat(await openNodeSqliteStorage(path), setup);
			opened.harness.resume();
			expect((await (await opened.harness.submission(submission.id, context))!.wait(context)).status).toBe("done");
			expect(hooks).toBe(2);
			expect(rendered).toBe(1);
			expect(endpoints).toEqual([original]);
		} finally {
			await opened.harness.close(context);
		}
	});

	for (const action of ["poll", "cancel"] as const) {
		it(`uses the admitted descriptor and headers for deferred ${action} after replacement`, async () => {
			const setup = chatSetup({ deferred: { pollAfterMs: 60_000 } });
			let now = 1000;
			setup.now = () => now;
			setup.settings.stream = { deferred: true, headers: { "x-request-route": "original" } };
			const original = catalogModel(setup).baseUrl;
			const endpoints: string[] = [];
			const headers: unknown[] = [];
			const provider = setup.faux.provider;
			const fetch = provider.fetchDeferred!.bind(provider);
			const cancel = provider.cancelDeferred!.bind(provider);
			provider.fetchDeferred = (model, handle, options) => {
				endpoints.push(model.baseUrl);
				headers.push(options?.headers);
				return fetch(model, handle, options);
			};
			provider.cancelDeferred = async (model, handle, options) => {
				endpoints.push(model.baseUrl);
				headers.push(options?.headers);
				await cancel(model, handle, options);
			};
			setup.faux.setResponses([fauxAssistantMessage("done")]);
			const path = await sqlitePath();
			let opened = await openChat(await openNodeSqliteStorage(path), setup);
			try {
				const submission = await opened.root.submit({ type: "input", content: "go" }, context);
				await waitFor(
					async () =>
						(await opened.harness.snapshot(LiveDoc, opened.root.id, context))?.generation?.deferred !== undefined,
				);
				await opened.harness.close(context);
				catalogModel(setup).baseUrl = "https://replacement.example";
				setup.settings.stream = { headers: { "x-request-route": "replacement" } };
				opened = await openChat(await openNodeSqliteStorage(path), setup);
				if (action === "poll") {
					now = 61_000;
					opened.harness.resume();
				} else await opened.root.abort(context);
				const outcome = await (await opened.harness.submission(submission.id, context))!.wait(context);
				expect(outcome.status).toBe(action === "poll" ? "done" : "unanswered");
				expect(endpoints).toEqual([original]);
				expect(headers).toEqual([{ "x-request-route": "original" }]);
			} finally {
				await opened.harness.close(context);
			}
		});
	}

	it("replays a summary's exact payload and descriptor after catalog removal, a history edit and clock change", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-summary-binding-"));
		const path = join(directory, "session.sqlite");
		const setup = chatSetup();
		setup.now = () => 1000;
		setup.settings.compaction = { enabled: false, keepRecentTokens: 1 };
		const original = copyJson(catalogModel(setup), { omitUndefinedProperties: true });
		const reached = deferred();
		const sent: JsonValue[] = [];
		setup.faux.setResponses([
			async (request, _options, _state, model) => {
				sent.push(copyJson({ request, model }, { omitUndefinedProperties: true }));
				reached.resolve();
				return aborted(_options!.signal!);
			},
			(request, _options, _state, model) => {
				sent.push(copyJson({ request, model }, { omitUndefinedProperties: true }));
				return fauxAssistantMessage("summary");
			},
		]);
		let opened = await openChat(await openNodeSqliteStorage(path), setup);
		try {
			await opened.root.commit(async (tx) => {
				await tx.appendEntry(opened.root.id, {
					kind: "app.history",
					model: [{ role: "user", content: "original history", timestamp: 1 }],
				});
				await tx.appendEntry(opened.root.id, {
					kind: "app.history",
					model: [{ role: "user", content: "keep", timestamp: 2 }],
				});
			}, context);
			const [first] = await allEntries(opened.root);
			const id = await opened.root.compact("retain details", context);
			await reached.promise;
			await opened.harness.close(context);
			setup.faux.provider.getModels = () => [];
			setup.now = () => 9000;
			opened = await openChat(await openNodeSqliteStorage(path), setup);
			await opened.root.submit(
				{
					type: "write",
					entry: {
						kind: "app.edit",
						edits: [
							{
								target: first!.id,
								action: "replace",
								messages: [{ role: "user", content: "changed history", timestamp: 3 }],
							},
						],
					},
				},
				context,
			);
			opened.harness.resume();
			expect((await opened.harness.waitForTask(id, context)).state.outcome.status).toBe("completed");
			expect(sent).toHaveLength(2);
			expect(sent[1]).toEqual(sent[0]);
			expect(sent[0]).toMatchObject({ model: original });
		} finally {
			await opened.harness.close(context);
			await rm(directory, { recursive: true, force: true });
		}
	});

	it("does not send a model request if storing the actual hook input fails", async () => {
		const setup = chatSetup();
		const storage = new ControlledStorage();
		addHooks(setup.registry, GenerationTask, {
			beforeRequest: () => {
				storage.failNextCommit(new StorageRejected("binding commit rejected"));
				return undefined;
			},
		});
		setup.faux.setResponses([fauxAssistantMessage("must not run")]);
		const opened = await openChat(storage, setup);
		try {
			expect(
				await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context),
			).toMatchObject({
				status: "unanswered",
				reason: "faulted",
				detail: "binding commit rejected",
			});
			expect(setup.faux.state.callCount).toBe(0);
		} finally {
			await opened.harness.close(context);
		}
	});

	it("refuses non-durable descriptor data before any provider dispatch", async () => {
		const setup = chatSetup();
		catalogModel(setup).samplingParams = { callback: () => "cannot survive replacement" };
		setup.faux.setResponses([fauxAssistantMessage("must not run")]);
		const opened = await openChat(new MemoryStorage(), setup);
		try {
			expect(
				await (await opened.root.submit({ type: "input", content: "go" }, context)).wait(context),
			).toMatchObject({
				status: "unanswered",
				reason: "faulted",
				detail: expect.stringContaining("non-JSON function"),
			});
			expect(setup.faux.state.callCount).toBe(0);
		} finally {
			await opened.harness.close(context);
		}
	});

	function refusesPreviousContract<I, S extends { phase: string }, R, H extends object>(
		definition: Task<I, S, R, H>,
		input: I,
	): void {
		it(`refuses the previous ${definition.definition.name} checkpoint contract before dispatch`, async () => {
			const setup = chatSetup();
			const opened = await openChat(new MemoryStorage(), setup);
			try {
				const id = await opened.root.commit(
					(tx) =>
						tx.createTask({ definition: { ...definition.definition, version: 1 } }, input, {
							ownership: { kind: "conversation" },
						}),
					context,
				);
				await opened.harness.runPass(context);
				const inspection = await opened.harness.inspect(context);
				expect(inspection.tasks.find((task) => task.record.id === id)?.state).toMatchObject({
					kind: "blocked",
					reason: "migration_failed",
				});
				expect(setup.faux.state.callCount).toBe(0);
			} finally {
				await opened.harness.close(context);
			}
		});
	}
	refusesPreviousContract(GenerationTask, {});
	refusesPreviousContract(CompactionTask, { reason: "manual" });
});
