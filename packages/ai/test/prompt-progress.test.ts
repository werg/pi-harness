import { describe, expect, it } from "vitest";
import { stream } from "../src/api/openai-completions.ts";
import { AssistantMessageFrameEncoder, reduceAssistantMessageFrames } from "../src/utils/assistant-message-frame.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const model = {
	id: "bundled",
	name: "bundled",
	provider: "local",
	api: "openai-completions",
	baseUrl: "http://127.0.0.1:9000/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 131072,
	maxTokens: 256,
} as const;

async function collect(chunks: Record<string, unknown>[], provider = "local") {
	let payload: Record<string, unknown> = {};
	const nativeFetch = async (_url: unknown, init?: RequestInit) => {
		payload = JSON.parse(String(init?.body));
		const text = `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`;
		return new Response(text, { headers: { "content-type": "text/event-stream" } });
	};
	const response = stream(
		{ ...model, provider, input: ["text"] },
		normalizeContext({
			messages: [{ role: "user", content: "Read this prompt", timestamp: 0 }],
		}),
		{ apiKey: "native-test-key", fetch: nativeFetch as typeof fetch },
	);
	const events = [];
	for await (const event of response) events.push(event);
	return { payload, events, result: await response.result() };
}

const progress = (processed: number, overrides: Record<string, unknown> = {}) => ({
	id: "request-owned",
	model: "bundled",
	choices: [{ index: 0, delta: { role: "assistant", content: null } }],
	prompt_progress: { total: 100, cache: 10, processed, ...overrides },
});
const completed = [
	{ id: "request-owned", choices: [{ index: 0, delta: { content: "Real answer" } }] },
	{ id: "request-owned", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
];

describe("native completion prompt progress", () => {
	it("retains monotonic request-specific work without turning it into assistant content", async () => {
		const { payload, events, result } = await collect([
			progress(20),
			progress(20),
			progress(19),
			{ ...progress(40), id: "other-request" },
			progress(30, { total: 200 }),
			progress(30, { cache: 11 }),
			progress(101),
			progress(Number.NaN),
			progress(30),
			...completed,
		]);
		expect(payload.return_progress).toBe(true);
		expect(events.filter((event) => event.type === "prompt_progress")).toMatchObject([
			{ total: 100, cache: 10, processed: 20 },
			{ total: 100, cache: 10, processed: 30 },
		]);
		expect(events[0]?.type).toBe("start");
		expect(result.content).toEqual([{ type: "text", text: "Real answer" }]);
		expect(result.stopReason).toBe("stop");
		const encoder = new AssistantMessageFrameEncoder();
		const frames = events.map((event) => encoder.encode(event)).filter((frame) => frame !== undefined);
		expect(frames.filter((frame) => frame.type === "prompt_progress")).toEqual([
			{ type: "prompt_progress", total: 100, cache: 10, processed: 20 },
			{ type: "prompt_progress", total: 100, cache: 10, processed: 30 },
		]);
		expect(reduceAssistantMessageFrames(frames)?.content).toEqual(result.content);
	});

	it("does not request llama-specific progress from another provider", async () => {
		const { payload, result } = await collect(completed, "openai");
		expect(payload).not.toHaveProperty("return_progress");
		expect(result.stopReason).toBe("stop");
	});
});
