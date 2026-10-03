import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { getCurrentTools } from "@earendil-works/pi-ai/utils/transcript";
import {
	bindReceipt,
	bindTool,
	createDirectToolTask,
	DirectToolCallEntry,
	DirectToolResultEntry,
	defineDoc,
	defineExtension,
	defineTool,
	Harness,
	hook,
	MemoryStorage,
	type ModelRequestTarget,
	type Storage,
	StorageRejected,
	ToolProgressDoc,
	ToolTask,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { allEntries, chatSetup } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";

const sessions: Harness[] = [];
const directories: string[] = [];
afterEach(async () => {
	await Promise.all(sessions.splice(0).map((session) => session.close(context)));
	await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe("genuine direct native tool invocation", () => {
	it("publishes owned progress without flushing a live split UTF-8 sequence", async () => {
		const setup = chatSetup();
		const tool = defineTool({
			name: "eval",
			description: "Direct",
			parameters: Type.Object({}),
			execute: async (_args, api, ctx) => {
				api.output(new Uint8Array([0xe2]));
				await api.details({ phase: "partial" }, ctx);
				expect(await api.snapshot(ToolProgressDoc, api.taskId, ctx)).toMatchObject({
					output: { text: "" },
					details: { phase: "partial" },
				});
				api.output(new Uint8Array([0x82, 0xac]));
				await api.details({ phase: "complete" }, ctx);
				return {};
			},
		});
		setup.registry.install(defineExtension({ name: "direct", tools: [tool] }));
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { tools: [tool] } });
		const id = await root.invokeTool({ id: "utf8:one", name: tool.name, arguments: {} }, context);
		await harness.waitForTask(id, context);
		expect((await allEntries(root)).filter(DirectToolResultEntry.is)).toMatchObject([
			{ data: { result: { content: [{ type: "text", text: "€" }], details: { phase: "complete" } } } },
		]);
	});
	it("retains direct execution definitions while incapable model rounds pin no tool offerings", async () => {
		const setup = chatSetup();
		const descriptor = setup.models.getModel("faux", "faux-1")!;
		descriptor.capabilities = { tools: false };
		let dispatches = 0;
		const tool = defineTool({
			name: "eval",
			description: "Direct",
			parameters: Type.Object({}),
			execute: async () => {
				dispatches++;
				return { content: [] };
			},
		});
		setup.registry.install(defineExtension({ name: "direct", tools: [tool] }));
		const requests: ModelRequestTarget[] = [];
		const harness = await Harness.open(
			new MemoryStorage(),
			{
				...setup,
				modelRequests: async (request, api, ctx) => {
					requests.push(request);
					const key = `test:model:${request.taskId}`;
					await api.commit((tx) => bindReceipt(tx, key, "test"), ctx);
					return { status: "waiting", condition: { kind: "receipt", key, binding: "test" } };
				},
			},
			context,
		);
		sessions.push(harness);
		const root = await harness.root(context, {
			agent: { model: { provider: "faux", modelId: "faux-1" }, tools: [tool] },
		});
		await root.submit({ type: "input", content: "model incapable" }, context);
		await harness.runPass(context);
		expect(getCurrentTools(requests[0]!.messages)).toEqual([]);
		expect(await harness.getTask(requests[0]!.taskId, context)).toMatchObject({
			state: { checkpoint: { offered: [], model: { capabilities: { tools: false } } } },
		});
		const direct = await root.invokeTool({ id: "actual direct", name: tool.name, arguments: {} }, context);
		expect((await harness.waitForTask(direct, context)).state.outcome.status).toBe("completed");
		expect(dispatches).toBe(1);
		expect(setup.faux.state.callCount).toBe(0);
		await root.abort(context);
		descriptor.capabilities.tools = true;
		await root.submit({ type: "input", content: "now capable" }, context);
		await harness.runPass(context);
		expect(getCurrentTools(requests[1]!.messages).map((tool) => tool.name)).toEqual(["eval"]);
		expect(await harness.getTask(requests[1]!.taskId, context)).toMatchObject({
			state: { checkpoint: { offered: [{ name: "eval" }], model: { capabilities: { tools: true } } } },
		});
		// A changed catalog cannot change the original admitted model capability.
		expect(requests[0]!.model.capabilities?.tools).toBe(false);
	});
	it("uses the actual ToolTask validation and hooks without provider messages or model configuration", async () => {
		const setup = chatSetup();
		const observed: unknown[] = [];
		const tool = defineTool({
			name: "eval",
			description: "Execute an actual direct call",
			parameters: Type.Object({ value: Type.String() }),
			execute: async (args, api) => {
				observed.push({ args, callId: api.callId, taskId: api.taskId });
				return { content: [{ type: "text" as const, text: args.value }], details: { value: args.value } };
			},
		});
		setup.registry.install(
			defineExtension({
				name: "direct",
				tools: [tool],
				hooks: [
					hook(ToolTask, {
						beforeTool: (call) => ({ arguments: { ...call.arguments, value: "actually prepared" } }),
					}),
				],
			}),
		);
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { tools: [tool] } });
		const id = await root.invokeTool(
			{ id: "domain-call", name: tool.name, arguments: { value: "original" } },
			context,
		);
		const ended = await harness.waitForTask(id, context);
		expect(ended.state.outcome.status).toBe("completed");
		expect(observed).toEqual([{ args: { value: "actually prepared" }, callId: "domain-call", taskId: id }]);
		const entries = await allEntries(root);
		expect(entries.every((entry) => entry.model === undefined)).toBe(true);
		expect(entries.filter(DirectToolCallEntry.is)).toMatchObject([
			{ data: { call: { id: "domain-call", arguments: { value: "original" } } } },
		]);
		expect(entries.filter(DirectToolResultEntry.is)).toMatchObject([
			{
				data: {
					sourceEntryId: entries[0]!.id,
					callId: "domain-call",
					result: { details: { value: "actually prepared" } },
				},
			},
		]);
		expect((await root.context(context)).messages).toEqual([]);
		expect(setup.faux.state.callCount).toBe(0);
	});

	it("atomically binds product state, the original source and the actual task; rejection dispatches nothing", async () => {
		const setup = chatSetup();
		const storage = new ControlledStorage();
		let dispatches = 0;
		const tool = defineTool({
			name: "eval",
			description: "Direct",
			parameters: Type.Object({}),
			execute: async () => {
				dispatches++;
				return { content: [] };
			},
		});
		setup.registry.install(defineExtension({ name: "direct", tools: [tool] }));
		const harness = await Harness.open(storage, setup, context);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { tools: [tool] } });
		const Product = defineDoc<{ taskId: number | null }>({
			kind: "test.direct-product",
			version: 1,
			scope: "conversation",
			history: "latest",
			fork: "initial",
			initial: () => ({ taskId: null }),
		});
		const error = new StorageRejected("actual direct admission rejected");
		storage.failNextCommit(error);
		const admit = () =>
			root.commit(async (tx) => {
				const id = await createDirectToolTask(
					tx,
					root.id,
					{ id: "run:one", name: tool.name, arguments: {} },
					bindTool(tool, "parallel"),
					{ ownership: { kind: "conversation" } },
				);
				(await tx.doc(Product, root.id)).taskId = id;
				return id;
			}, context);
		await expect(admit()).rejects.toBe(error);
		expect(await allEntries(root)).toEqual([]);
		expect(await harness.snapshot(Product, root.id, context)).toBeUndefined();
		expect(dispatches).toBe(0);
		const id = await admit();
		expect((await harness.snapshot(Product, root.id, context))?.taskId).toBe(id);
		await harness.waitForTask(id, context);
		expect(dispatches).toBe(1);
	});

	it("refuses an unselected definition and validates direct arguments through the original execution contract", async () => {
		const setup = chatSetup();
		let dispatches = 0;
		const tool = defineTool({
			name: "eval",
			description: "Direct",
			parameters: Type.Object({ value: Type.String() }),
			execute: async () => {
				dispatches++;
				return { content: [] };
			},
		});
		setup.registry.install(defineExtension({ name: "direct", tools: [tool] }));
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { tools: [] } });
		await expect(
			root.invokeTool({ id: "not-selected", name: tool.name, arguments: { value: "one" } }, context),
		).rejects.toThrow("not selected");
		expect(await allEntries(root)).toEqual([]);
		await root.configure({ tools: [tool] }, context);
		const id = await root.invokeTool({ id: "invalid", name: tool.name, arguments: {} }, context);
		await harness.waitForTask(id, context);
		expect(dispatches).toBe(0);
		expect((await allEntries(root)).filter(DirectToolResultEntry.is)).toMatchObject([
			{ data: { result: { isError: true, diagnostics: [{ code: "invalid_arguments" }] } } },
		]);
	});

	it("retains lost initial acknowledgement and failed direct cancellation in SQLite under the same original source", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-direct-tool-"));
		directories.push(directory);
		const file = join(directory, "session.sqlite");
		const setup = chatSetup();
		const original = new Error("direct start response lost");
		const cleanup = new Error("direct cleanup readback lost");
		let starts = 0;
		let cancels = 0;
		let repaired = false;
		const identities: unknown[] = [];
		const tool = defineTool({
			name: "eval",
			description: "Direct",
			parameters: Type.Object({ value: Type.String() }),
			replay: "unsafe",
			executionData: { provider: "original" },
			execute: async (args, api, ctx) => {
				identities.push({ taskId: api.taskId, callId: api.callId, args, data: api.executionData });
				await api.retainContinuation({ operation: api.callId }, () => {}, ctx);
				api.output("actual direct partial output\n");
				api.diagnostic({ severity: "warn", code: "original", message: "Original direct progress" });
				await api.details({ original: args.value }, ctx);
				starts++;
				throw original;
			},
			cancel: async (args, api) => {
				identities.push({ taskId: api.taskId, callId: api.callId, args, data: api.executionData });
				cancels++;
				if (!repaired) throw cleanup;
				return { content: [] };
			},
		});
		setup.registry.install(defineExtension({ name: "direct", tools: [tool] }));
		const open = async (storage: Storage) => {
			const harness = await Harness.open(storage, setup, context);
			sessions.push(harness);
			const root = await harness.root(context, { agent: { tools: [tool] } });
			return { harness, root };
		};
		let f = await open(await openNodeSqliteStorage(file));
		const id = await f.root.invokeTool(
			{ id: "domain:one", name: tool.name, arguments: { value: "pinned" } },
			context,
		);
		await f.harness.runPass(context);
		await expect(f.root.waitForIdle(context)).rejects.toBe(original);
		await expect(f.root.abort(context)).rejects.toBe(cleanup);
		const source = (await allEntries(f.root)).filter(DirectToolCallEntry.is);
		expect(await f.harness.snapshot(ToolProgressDoc, id, context)).toMatchObject({
			output: { text: "actual direct partial output\n" },
			details: { original: "pinned" },
			diagnostics: [{ code: "original" }],
		});
		await f.harness.close(context);
		setup.registry.install(
			defineExtension({ name: "direct", tools: [{ ...tool, executionData: { provider: "replacement" } }] }),
		);
		f = await open(await openNodeSqliteStorage(file));
		const failed = await f.harness.getTask(id, context);
		if (failed?.state.status !== "waiting" || failed.state.condition.kind !== "failure")
			throw new Error("Direct cancellation has no retained incident");
		expect(failed).toMatchObject({ abortRequested: true, state: { mode: "abort" } });
		expect(await f.harness.snapshot(ToolProgressDoc, id, context)).toMatchObject({
			output: { text: "actual direct partial output\n" },
			details: { original: "pinned" },
			diagnostics: [{ code: "original" }],
		});
		repaired = true;
		await f.harness.retryTask(id, failed.state.condition.incident, context);
		expect((await f.harness.waitForTask(id, context)).state.outcome.status).toBe("aborted");
		expect({ starts, cancels }).toEqual({ starts: 1, cancels: 2 });
		expect(identities).toEqual([identities[0], identities[0], identities[0]]);
		expect(identities[0]).toMatchObject({ data: { provider: "original" } });
		expect((await allEntries(f.root)).filter(DirectToolCallEntry.is)).toEqual(source);
		expect((await allEntries(f.root)).filter(DirectToolResultEntry.is)).toMatchObject([
			{ data: { result: { details: { original: "pinned" }, diagnostics: [{ code: "original" }] } } },
		]);
		expect(setup.faux.state.callCount).toBe(0);
	});
});
