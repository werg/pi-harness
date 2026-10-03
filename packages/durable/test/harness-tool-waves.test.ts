import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	acceptReceipt,
	bindReceipt,
	defineTool,
	Harness,
	ReceiptDoc,
	type ToolExecutionMode,
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
	const directory = await mkdtemp(join(tmpdir(), "pi-tool-waves-"));
	directories.push(directory);
	const path = join(directory, "state.sqlite");
	const setup = chatSetup();
	const admitted: string[] = [];
	for (const [name, executionMode] of [
		["a", "parallel"],
		["b", "parallel"],
		["c", "sequential"],
		["d", "parallel"],
		["e", "parallel"],
	] as const satisfies readonly (readonly [string, ToolExecutionMode])[]) {
		addTool(
			setup.registry,
			defineTool({
				name,
				description: name,
				parameters: Type.Object({}),
				replay: "safe",
				executionMode,
				execute: async (_args, api, ctx) => {
					const key = `wave:${api.callId}`;
					if (api.continuation !== undefined) {
						const result = await api.snapshot(ReceiptDoc, key, ctx);
						expect(result?.result).toEqual({ done: name });
						return { content: [{ type: "text", text: name }] };
					}
					await api.commit((tx) => bindReceipt(tx, key, "test-owner"), ctx);
					admitted.push(name);
					return { wait: { kind: "receipt", key, binding: "test-owner" }, continuation: name };
				},
				cancel: async () => ({ content: [{ type: "text", text: "cancelled" }] }),
			}),
		);
	}
	setup.faux.setResponses([
		fauxAssistantMessage(
			["a", "b", "c", "d", "e"].map((name) => fauxToolCall(name, {}, { id: name })),
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage("done"),
	]);
	const open = async () => {
		const harness = await Harness.open(
			await openNodeSqliteStorage(path),
			{
				models: setup.models,
				registry: setup.registry,
				settings: setup.settings,
				publishWake: async () => {},
				onReport: (error) => setup.reports.push(error),
			},
			context,
		);
		handles.push(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		return { harness, root };
	};
	const deliver = (harness: Harness, name: string) =>
		harness.commit((tx) => acceptReceipt(tx, `wave:${name}`, "test-owner", { done: name }), context);
	return { open, deliver, admitted, setup };
}

describe("ordered parallel tool waves", () => {
	it("holds each sequential barrier until its preceding wave settles, across owner reopen", async () => {
		const { open, deliver, admitted, setup } = await fixture();
		let { harness, root } = await open();
		const submission = await root.submit({ type: "input", content: "go" }, context);
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b"]);
		await deliver(harness, "a");
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b"]);
		await harness.close(context);
		({ harness, root } = await open());
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b"]);
		await deliver(harness, "b");
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b", "c"]);
		await harness.close(context);
		({ harness, root } = await open());
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b", "c"]);
		await deliver(harness, "c");
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b", "c", "d", "e"]);
		await deliver(harness, "d");
		await deliver(harness, "e");
		await harness.runPass(context);
		expect((await (await harness.submission(submission.id, context))?.status(context))?.status).toBe("done");
		expect(setup.reports).toEqual([]);
	});

	it("cancels the admitted wave and answers later calls without ever admitting them", async () => {
		const { open, admitted, setup } = await fixture();
		const { harness, root } = await open();
		const submission = await root.submit({ type: "input", content: "go" }, context);
		await harness.runPass(context);
		expect(admitted).toEqual(["a", "b"]);
		await root.abort(context);
		expect(admitted).toEqual(["a", "b"]);
		expect((await submission.status(context)).status).toBe("unanswered");
		const view = await root.context(context);
		expect(
			view.messages.filter((message) => message.role === "toolResult").map((message) => message.toolCallId),
		).toEqual(["a", "b", "c", "d", "e"]);
		expect(setup.reports).toEqual([]);
	});
});
