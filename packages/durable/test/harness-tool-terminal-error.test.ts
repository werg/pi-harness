import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { DirectToolResultEntry, defineExtension, defineTool, Harness, MemoryStorage } from "@earendil-works/pi-durable";
import { expect, it } from "vitest";
import { openNodeSqliteStorage } from "../src/storage/sqlite/node.ts";
import { chatSetup } from "./chat-support.ts";
import { context } from "./session-support.ts";

it.each(["memory", "sqlite"] as const)("retains original tool terminal errors in %s", async (backend) => {
	const directory = backend === "sqlite" ? await mkdtemp(join(tmpdir(), "pi-tool-terminal-error-")) : undefined;
	const original = new Error("original tool admission refused", {
		cause: { operation: "real-operation", reason: "canonical write rejected" },
	});
	const closure = new Error("original error publication refused");
	const failure = new AggregateError([original, closure], "Tool admission and error publication failed", {
		cause: original,
	});
	const setup = chatSetup();
	const tool = defineTool({
		name: "failed-tool",
		description: "Preserve original failure",
		parameters: Type.Object({}),
		execute: async () => {
			throw failure;
		},
	});
	setup.registry.install(defineExtension({ name: "failed-tool", tools: [tool] }));
	let harness = await Harness.open(
		directory ? await openNodeSqliteStorage(join(directory, "session.sqlite")) : new MemoryStorage(),
		setup,
		context,
	);
	try {
		const root = await harness.root(context, { agent: { tools: [tool] } });
		const id = await root.invokeTool({ id: "actual-source", name: tool.name, arguments: {} }, context);
		const expected = {
			status: "failed",
			error: {
				message: failure.message,
				detail: {
					errors: [original.message, closure.message],
					cause: { message: original.message, detail: { cause: original.cause } },
				},
			},
		};
		expect((await harness.waitForTask(id, context)).state.outcome).toMatchObject(expected);
		const result = (await root.entries({}, 100, undefined, context)).items.find(DirectToolResultEntry.is);
		expect(result?.data?.result).toMatchObject({
			isError: true,
			diagnostics: [{ code: "tool_error", message: failure.message }],
		});
		if (directory) {
			await harness.close(context);
			harness = await Harness.open(await openNodeSqliteStorage(join(directory, "session.sqlite")), setup, context);
			expect((await harness.waitForTask(id, context)).state.outcome).toMatchObject(expected);
		}
	} finally {
		await harness.close(context);
		if (directory) await rm(directory, { recursive: true, force: true });
	}
});
