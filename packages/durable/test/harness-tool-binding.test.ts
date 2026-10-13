import { Type } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { bindTool, toolMatches } from "../src/harness/binding.ts";
import type { ToolRegistration } from "../src/harness/types.ts";

const tool: ToolRegistration = {
	name: "bound",
	description: "Binding contract",
	parameters: Type.Object({}),
	execute: async () => ({ output: [] }),
};

describe("tool admission binding", () => {
	it("rejects a definition whose structured result schema changes", () => {
		const original = { ...tool, structuredOutputSchema: Type.String() };
		const binding = bindTool(original, "parallel");
		expect(toolMatches({ ...original, structuredOutputSchema: Type.Number() }, binding)).toBe(false);
		expect(toolMatches({ ...original, structuredOutputSchema: Type.String() }, binding)).toBe(true);
	});
	it("rejects a definition whose callable audiences change", () => {
		const original = { ...tool, callers: ["tools"] as const };
		const binding = bindTool(original, "parallel");
		expect(toolMatches({ ...original, callers: ["model"] }, binding)).toBe(false);
	});
});
