import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, fauxToolCall, Type } from "@earendil-works/pi-ai";
import {
	bindReceipt,
	defineExtension,
	defineTool,
	Harness,
	LiveDoc,
	ReceiptDoc,
	type Storage,
	StorageRejected,
	type ToolExecutionApi,
} from "@earendil-works/pi-durable";
import { afterEach, describe, expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { chatSetup, toolsNamed, waitFor } from "./chat-support.ts";
import { ControlledStorage, context } from "./session-support.ts";

const sessions: Harness[] = [];
const directories: string[] = [];
afterEach(async () => {
	await Promise.all(sessions.splice(0).map((session) => session.close(context)));
	await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});
function scenario(storage?: ControlledStorage, loseHeap = false) {
	const setup = chatSetup();
	const original = new Error("original start acknowledgement lost");
	const cleanup = new Error("original cancellation readback failed");
	const state = { dispatches: 0, cancellations: 0, preparations: 0, repaired: false };
	const identities: unknown[] = [];
	let captured: ToolExecutionApi | undefined;
	const tool = defineTool({
		name: "external",
		description: "Retain one real external operation",
		parameters: Type.Object({ value: Type.String() }),
		replay: "unsafe",
		prepareArguments: (args) => {
			state.preparations++;
			return args as { value: string };
		},
		execute: async (args, api, ctx) => {
			captured = api;
			if (api.continuation === undefined) {
				storage?.failNextCommit(new StorageRejected("original continuation batch rejected"));
				await api.retainContinuation(
					{ operation: api.callId },
					(tx) => bindReceipt(tx, api.callId, `binding:${args.value}`),
					ctx,
				);
				state.dispatches++;
				identities.push({ taskId: api.taskId, callId: api.callId, args });
				api.output("domain start attempted\n");
				await api.details({ operation: api.callId }, ctx);
				if (loseHeap)
					await new Promise<void>((_resolve, reject) => {
						const signal = ctx.abortSignal!;
						signal.addEventListener("abort", () => reject(signal.reason), { once: true });
						if (signal.aborted) reject(signal.reason);
					});
				throw original;
			}
			identities.push({ taskId: api.taskId, callId: api.callId, args, continuation: api.continuation });
			return { content: [{ type: "text" as const, text: "actual domain result" }] };
		},
		cancel: async (args, api) => {
			state.cancellations++;
			identities.push({ taskId: api.taskId, callId: api.callId, args, continuation: api.continuation });
			if (!state.repaired) throw cleanup;
			return { content: [] };
		},
	});
	setup.registry.install(defineExtension({ name: "external", tools: [tool] }));
	setup.faux.setResponses([
		fauxAssistantMessage([fauxToolCall(tool.name, { value: "pinned" }, { id: "run:one" })], {
			stopReason: "toolUse",
		}),
		fauxAssistantMessage("done"),
	]);
	async function open(storage: Storage) {
		const harness = await Harness.open(
			storage,
			{ models: setup.models, registry: setup.registry, onReport: (error) => setup.reports.push(error) },
			context,
		);
		sessions.push(harness);
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		await root.configure({ tools: toolsNamed(setup, tool.name) }, context);
		return { harness, root };
	}
	return { open, state, original, cleanup, identities, captured: () => captured };
}
describe("external tool initial admission", () => {
	it("reattaches the committed original operation after losing the dispatching heap", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-tool-admission-"));
		directories.push(directory);
		const path = join(directory, "session.sqlite");
		const f = scenario(undefined, true);
		let opened = await f.open(await openNodeSqliteStorage(path));
		const submission = await opened.root.submit({ type: "input", content: "go" }, context);
		await waitFor(() => f.state.dispatches === 1);
		await opened.harness.close(context);
		opened = await f.open(await openNodeSqliteStorage(path));
		expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
			status: "done",
		});
		expect(f.state).toMatchObject({ dispatches: 1, preparations: 1, cancellations: 0 });
		expect(f.identities[1]).toEqual({ ...(f.identities[0] as object), continuation: { operation: "run:one" } });
	});
	it("retains the initial lost acknowledgement in SQLite and repairs the same operation without rerunning preparation", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-tool-admission-"));
		directories.push(directory);
		const path = join(directory, "session.sqlite");
		const f = scenario();
		let opened = await f.open(await openNodeSqliteStorage(path));
		const submission = await opened.root.submit({ type: "input", content: "go" }, context);
		await opened.harness.runPass(context);
		const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.tools![0]!.taskId!;
		await expect(opened.root.waitForIdle(context)).rejects.toBe(f.original);
		const failed = await opened.harness.getTask(taskId, context);
		if (failed?.state.status !== "waiting" || failed.state.condition.kind !== "failure")
			throw Error("missing original retained failure");
		expect(failed.state.checkpoint).toMatchObject({
			phase: "execute",
			arguments: { value: "pinned" },
			replay: "unsafe",
			continuation: { operation: "run:one" },
		});
		await opened.harness.close(context);
		opened = await f.open(await openNodeSqliteStorage(path));
		await expect(opened.harness.waitForTask(taskId, context)).rejects.toThrow(f.original.message);
		expect(f.state.dispatches).toBe(1);
		await opened.harness.retryTask(taskId, failed.state.condition.incident, context);
		await opened.harness.runPass(context);
		expect(await (await opened.harness.submission(submission.id, context))!.wait(context)).toMatchObject({
			status: "done",
		});
		expect(f.state).toMatchObject({ dispatches: 1, preparations: 1, cancellations: 0 });
		expect(f.identities[1]).toEqual({ ...(f.identities[0] as object), continuation: { operation: "run:one" } });
	});
	it("keeps a second cancellation failure owned in abort mode across replacement and exact repair", async () => {
		const directory = await mkdtemp(join(tmpdir(), "pi-tool-admission-"));
		directories.push(directory);
		const path = join(directory, "session.sqlite");
		const f = scenario();
		let opened = await f.open(await openNodeSqliteStorage(path));
		await opened.root.submit({ type: "input", content: "go" }, context);
		await opened.harness.runPass(context);
		const taskId = (await opened.harness.snapshot(LiveDoc, opened.root.id, context))!.tools![0]!.taskId!;
		await expect(opened.root.abort(context)).rejects.toBe(f.cleanup);
		const failed = await opened.harness.getTask(taskId, context);
		expect(failed).toMatchObject({
			abortRequested: true,
			state: { status: "waiting", mode: "abort", checkpoint: { continuation: { operation: "run:one" } } },
		});
		if (failed?.state.status !== "waiting" || failed.state.condition.kind !== "failure")
			throw Error("missing retained cancellation incident");
		await opened.harness.close(context);
		opened = await f.open(await openNodeSqliteStorage(path));
		await expect(opened.harness.waitForTask(taskId, context)).rejects.toThrow(f.cleanup.message);
		f.state.repaired = true;
		await opened.harness.retryTask(taskId, failed.state.condition.incident, context);
		expect((await opened.harness.waitForTask(taskId, context)).state.outcome.status).toBe("aborted");
		expect(f.state).toMatchObject({ dispatches: 1, cancellations: 2, preparations: 1 });
	});
	it("rolls original identity and continuation back before dispatch, and refuses a settled capability", async () => {
		const storage = new ControlledStorage();
		const f = scenario(storage);
		const opened = await f.open(storage);
		const submission = await opened.root.submit({ type: "input", content: "go" }, context);
		await opened.harness.runPass(context);
		expect(await submission.wait(context)).toMatchObject({ status: "done" });
		expect(f.state.dispatches).toBe(0);
		expect(await opened.harness.snapshot(ReceiptDoc, "run:one", context)).toBeUndefined();
		await expect(f.captured()!.retainContinuation({ operation: "late" }, async () => {}, context)).rejects.toThrow(
			"settled",
		);
	});
});
