import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Harness, type ModelRequestPort } from "@earendil-works/pi-durable";
import { expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { chatSetup } from "./chat-support.ts";
import { context } from "./session-support.ts";

it("retains both protected-port and withdrawal failures in a terminal native task through SQLite reopen", async () => {
	const directory = await mkdtemp(join(tmpdir(), "pi-terminal-error-"));
	const path = join(directory, "session.sqlite");
	const setup = chatSetup();
	const original = new Error("original protected provider admission failed");
	const closure = new Error("canonical acquisition withdrawal failed");
	const failure = new AggregateError([original, closure], "Model admission and withdrawal failed", {
		cause: original,
	});
	const modelRequests: ModelRequestPort = async () => {
		throw failure;
	};
	const options = { models: setup.models, registry: setup.registry, modelRequests, publishWake: async () => {} };
	let harness = await Harness.open(await openNodeSqliteStorage(path), options, context);
	try {
		const root = await harness.root(context, { agent: { model: { provider: "faux", modelId: "faux-1" } } });
		const submission = await root.submit({ type: "input", content: "hello" }, context);
		await harness.runPass(context);
		const receipt = await submission.status(context);
		expect(receipt).toMatchObject({ status: "unanswered", reason: "faulted" });
		const tasks = await root.commit((tx) => tx.scanTasks({ conversationId: root.id }, 10), context);
		expect(tasks.items).toHaveLength(1);
		const taskId = tasks.items[0]!.id;
		const expected = {
			status: "terminal",
			outcome: {
				status: "faulted",
				error: {
					message: failure.message,
					detail: { errors: [original.message, closure.message], cause: { message: original.message } },
				},
			},
		};
		expect(tasks.items[0]!.state).toEqual(expected);
		await harness.close(context);
		harness = await Harness.open(await openNodeSqliteStorage(path), options, context);
		expect((await harness.getTask(taskId, context))!.state).toEqual(expected);
		expect((await harness.waitForTask(taskId, context)).state).toEqual(expected);
	} finally {
		await harness.close(context);
		await rm(directory, { recursive: true, force: true });
	}
});
