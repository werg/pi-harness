import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	acceptReceipt,
	bindReceipt,
	createRegistry,
	defineTool,
	Harness,
	MemoryStorage,
	ReceiptDoc,
	type Registry,
	type ToolRegistration,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { chatSetup } from "./chat-support.ts";
import { addTool } from "./harness-support.ts";
import { context } from "./session-support.ts";

const directories: string[] = [];
const handles: Harness[] = [];
afterEach(async () => {
	await Promise.all(handles.splice(0).map((harness) => harness.close(context)));
	await Promise.all(directories.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

async function fixture() {
	const directory = await mkdtemp(join(tmpdir(), "pi-tool-binding-"));
	directories.push(directory);
	const setup = chatSetup();
	let admissions = 0;
	let resumes = 0;
	let cancellations = 0;
	let changedCalls = 0;
	const original = defineTool({
		name: "external",
		version: 1,
		description: "owned operation",
		parameters: Type.Object({}),
		execute: async (_args, api, ctx) => {
			if (api.continuation !== undefined) {
				resumes++;
				expect((await api.snapshot(ReceiptDoc, "operation", ctx))?.result).toEqual({ answer: 42 });
				return { details: { answer: 42 } };
			}
			admissions++;
			await api.commit((tx) => bindReceipt(tx, "operation", "owner"), ctx);
			api.output("original output");
			return { wait: { kind: "receipt", key: "operation", binding: "owner" }, continuation: "operation" };
		},
		cancel: async () => {
			cancellations++;
			return { content: [] };
		},
	});
	const changed: ToolRegistration = {
		...original,
		version: 2,
		execute: async () => {
			changedCalls++;
			return { content: [] };
		},
		cancel: async () => {
			changedCalls++;
			return { content: [] };
		},
	};
	addTool(setup.registry, original);
	setup.faux.setResponses([
		fauxAssistantMessage([fauxToolCall("external", {}, { id: "call" })], { stopReason: "toolUse" }),
		fauxAssistantMessage("done"),
	]);
	const open = async (registry: Registry = setup.registry) => {
		const harness = await Harness.open(
			await openNodeSqliteStorage(join(directory, "state.sqlite")),
			{
				models: setup.models,
				registry,
				publishWake: async () => {},
				onReport: (error) => setup.reports.push(error),
			},
			context,
		);
		handles.push(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		return { harness, root };
	};
	const counts = () => ({ admissions, resumes, cancellations, changedCalls });
	const toolTask = async (harness: Harness) =>
		(await harness.inspect(context)).tasks.find((task) => task.record.kind === "pi.tool")!;
	return { setup, original, changed, open, counts, toolTask };
}

describe("prepared tool contracts", () => {
	it("executes original offered context after a same-contract registry refresh without exposing mutable binding aliases", async () => {
		const setup = chatSetup();
		const data = { participants: ["original-human"], provider: "original-provider" };
		const observed: unknown[] = [];
		const original = defineTool({
			name: "ask",
			description: "Ask the offered people",
			parameters: Type.Object({}),
			executionData: data,
			execute: async (_args, api) => {
				const first = api.executionData;
				observed.push(first);
				if (first === null || typeof first !== "object" || Array.isArray(first))
					throw new Error("Missing offered context");
				expect(() => {
					first.provider = "mutated consumer copy";
				}).toThrow(TypeError);
				return { details: api.executionData };
			},
		});
		addTool(setup.registry, original);
		setup.faux.setResponses([
			() => {
				data.participants[0] = "mutated registration alias";
				addTool(setup.registry, {
					...original,
					executionData: { participants: ["replacement-human"], provider: "replacement-provider" },
				});
				return fauxAssistantMessage([fauxToolCall("ask", {}, { id: "original-call" })], { stopReason: "toolUse" });
			},
			fauxAssistantMessage("done"),
		]);
		const harness = await Harness.open(new MemoryStorage(), setup, context);
		handles.push(harness);
		const root = await harness.root(context, {
			agent: { model: { provider: "faux", modelId: "faux-1" }, tools: [original] },
		});
		await root.submit({ type: "input", content: "ask" }, context);
		await root.waitForIdle(context);
		expect(observed).toEqual([{ participants: ["original-human"], provider: "original-provider" }]);
		expect((await root.context(context)).messages.find((message) => message.role === "toolResult")).toMatchObject({
			details: { participants: ["original-human"], provider: "original-provider" },
		});
		expect(setup.reports).toEqual([]);
	});
	it("blocks changed code before admission when publication changes during the model request", async () => {
		const f = await fixture();
		f.setup.faux.setResponses([
			() => {
				addTool(f.setup.registry, f.changed);
				return fauxAssistantMessage([fauxToolCall("external", {}, { id: "call" })], { stopReason: "toolUse" });
			},
			fauxAssistantMessage("done"),
		]);
		const { harness, root } = await f.open();
		await root.submit({ type: "input", content: "go" }, context);
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		expect(f.counts()).toEqual({ admissions: 0, resumes: 0, cancellations: 0, changedCalls: 0 });
		expect((await f.toolTask(harness)).state).toMatchObject({ kind: "blocked", reason: "incompatible_binding" });
		addTool(f.setup.registry, f.original);
		await harness.runPass(context);
		expect(f.counts().admissions).toBe(1);
		expect(f.setup.reports).toEqual([]);
	});

	it("retains the accepted receipt, checkpoint and output across incompatible replacement and resumes exactly once", async () => {
		const f = await fixture();
		let { harness, root } = await f.open();
		const submission = await root.submit({ type: "input", content: "go" }, context);
		await harness.runPass(context);
		const task = await f.toolTask(harness);
		const checkpoint = task.record.state.checkpoint;
		await harness.close(context);
		const registry = createRegistry();
		addTool(registry, f.changed);
		({ harness, root } = await f.open(registry));
		await harness.commit((tx) => acceptReceipt(tx, "operation", "owner", { answer: 42 }), context);
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		expect((await f.toolTask(harness)).record.state.checkpoint).toEqual(checkpoint);
		expect((await f.toolTask(harness)).state).toMatchObject({ kind: "blocked", reason: "incompatible_binding" });
		await harness.runPass(context);
		expect(f.counts()).toEqual({ admissions: 1, resumes: 0, cancellations: 0, changedCalls: 0 });
		await harness.close(context);
		// A new activation rechecks once, then remains dormant while still incompatible.
		const nextRegistry = createRegistry();
		addTool(nextRegistry, f.changed);
		({ harness, root } = await f.open(nextRegistry));
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		expect(f.counts().changedCalls).toBe(0);
		addTool(nextRegistry, f.original);
		await harness.runPass(context);
		await harness.runPass(context);
		expect(f.counts()).toEqual({ admissions: 1, resumes: 1, cancellations: 0, changedCalls: 0 });
		expect((await (await harness.submission(submission.id, context))?.status(context))?.status).toBe("done");
		const view = await root.context(context);
		expect(view.messages.find((message) => message.role === "toolResult")).toMatchObject({
			content: [{ type: "text", text: "original output" }],
		});
		expect(f.setup.reports).toEqual([]);
	});

	it("keeps abort cleanup live and never delegates it to an incompatible cancellation handler", async () => {
		const f = await fixture();
		let { harness, root } = await f.open();
		await root.submit({ type: "input", content: "go" }, context);
		await harness.runPass(context);
		const id = (await f.toolTask(harness)).record.id;
		addTool(f.setup.registry, f.changed);
		await harness.abortTask(id, context);
		expect((await harness.runPass(context)).wakeAt).toBeNull();
		expect((await f.toolTask(harness)).record.state).toMatchObject({
			status: "waiting",
			mode: "abort",
			checkpoint: { continuation: "operation" },
		});
		expect((await f.toolTask(harness)).state).toMatchObject({ kind: "blocked", reason: "incompatible_binding" });
		await harness.close(context);
		({ harness, root } = await f.open());
		await harness.runPass(context);
		expect(f.counts()).toEqual({ admissions: 1, resumes: 0, cancellations: 0, changedCalls: 0 });
		addTool(f.setup.registry, f.original);
		await harness.runPass(context);
		expect((await harness.getTask(id, context))?.state.status).toBe("terminal");
		expect(f.counts()).toEqual({ admissions: 1, resumes: 0, cancellations: 1, changedCalls: 0 });
		expect(f.setup.reports).toEqual([]);
	});

	it("detects wire-schema drift even without a version bump and rejects an invalid publication atomically", async () => {
		const f = await fixture();
		const { harness, root } = await f.open();
		await root.submit({ type: "input", content: "go" }, context);
		await harness.runPass(context);
		const snapshot = f.setup.registry.snapshot();
		expect(() => addTool(f.setup.registry, { ...f.original, version: 0 })).toThrow(/Invalid version/);
		expect(f.setup.registry.snapshot()).toBe(snapshot);
		addTool(f.setup.registry, { ...f.original, parameters: Type.Object({ marker: Type.Optional(Type.String()) }) });
		await harness.commit((tx) => acceptReceipt(tx, "operation", "owner", { answer: 42 }), context);
		await harness.runPass(context);
		expect((await f.toolTask(harness)).state).toMatchObject({ kind: "blocked", reason: "incompatible_binding" });
		expect(f.counts().resumes).toBe(0);
		addTool(f.setup.registry, f.original);
		await harness.runPass(context);
		expect(f.counts().resumes).toBe(1);
	});
});
