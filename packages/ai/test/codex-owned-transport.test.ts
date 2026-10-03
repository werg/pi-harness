import { afterEach, describe, expect, it, vi } from "vitest";
import {
	getOpenAICodexWebSocketDebugStats,
	releaseOpenAICodexWebSocketSession,
	stream,
	streamSimple,
} from "../src/api/openai-codex-responses.ts";
import { cleanupSessionResources } from "../src/session-resources.ts";
import type { Model, ProviderWebSocket, StreamOptions } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const model: Model<"openai-codex-responses"> = {
	id: "test-model",
	name: "Test",
	api: "openai-codex-responses",
	provider: "openai-codex",
	baseUrl: "https://provider.invalid/backend-api",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 100000,
	maxTokens: 1000,
};
const token = `aaa.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "test-account" } })).toString("base64")}.bbb`;
const context = normalizeContext({ messages: [{ role: "user", content: "hello", timestamp: 0 }] });
const terminal = {
	type: "response.completed",
	response: {
		id: "response-owned",
		status: "completed",
		output: [],
		usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8, input_tokens_details: { cached_tokens: 0 } },
	},
};

class OwnedSocket implements ProviderWebSocket {
	readonly listeners = new Map<string, Set<(event: unknown) => void>>();
	readonly close = vi.fn();
	readonly sent: unknown[] = [];
	readonly complete: boolean;
	constructor(complete = true) {
		this.complete = complete;
	}
	send(data: string): void {
		this.sent.push(JSON.parse(data));
		queueMicrotask(() => {
			this.emit({ type: "response.created", response: { id: "response-owned", status: "in_progress" } });
			if (this.complete) this.emit(terminal);
		});
	}
	emit(value: unknown): void {
		for (const listener of this.listeners.get("message") ?? []) listener({ data: JSON.stringify(value) });
	}
	addEventListener(type: string, listener: (event: unknown) => void): void {
		const listeners = this.listeners.get(type) ?? new Set();
		listeners.add(listener);
		this.listeners.set(type, listeners);
	}
	removeEventListener(type: string, listener: (event: unknown) => void): void {
		this.listeners.get(type)?.delete(listener);
	}
}

afterEach(() => {
	vi.useRealTimers();
	vi.restoreAllMocks();
	cleanupSessionResources();
	vi.unstubAllGlobals();
});

describe("Codex invocation-owned transport", () => {
	it("accounts for the Codex fast response tier using priority pricing", async () => {
		const priced = { ...model, cost: { input: 1, output: 2, cacheRead: 0.5, cacheWrite: 0 } };
		const response = { ...terminal, response: { ...terminal.response, service_tier: "fast" } };
		const result = await stream(priced, context, {
			apiKey: token,
			transport: "sse",
			serviceTier: "priority",
			fetch: async () => new Response(`data: ${JSON.stringify(response)}\n\n`),
		}).result();
		expect(result.stopReason).toBe("stop");
		expect(result.usage.cost.input).toBeCloseTo(10 / 1_000_000, 12);
		expect(result.usage.cost.output).toBeCloseTo(12 / 1_000_000, 12);
		expect(result.usage.cost.total).toBeCloseTo(22 / 1_000_000, 12);
	});

	it("uses the attributed connector, gives each attempt an identity and closes every socket", async () => {
		const ambient = vi.fn(() => {
			throw new Error("ambient transport escaped attribution");
		});
		vi.stubGlobal("WebSocket", ambient);
		const sockets: OwnedSocket[] = [];
		const headers: Headers[] = [];
		const connectWebSocket: StreamOptions["connectWebSocket"] = async (url, options) => {
			expect(url).toBe("wss://provider.invalid/backend-api/codex/responses");
			headers.push(options.headers);
			const socket = new OwnedSocket();
			sockets.push(socket);
			return socket;
		};
		for (let attempt = 0; attempt < 2; attempt++) {
			const result = await streamSimple(model, context, {
				apiKey: token,
				sessionId: "owner-session",
				connectWebSocket,
			}).result();
			expect(result.stopReason).toBe("stop");
			expect(result.usage.totalTokens).toBe(8);
		}
		expect(ambient).not.toHaveBeenCalled();
		expect(sockets).toHaveLength(2);
		for (const socket of sockets) {
			expect(socket.close).toHaveBeenCalledOnce();
			expect([...socket.listeners.values()].every((listeners) => listeners.size === 0)).toBe(true);
		}
		expect(headers.map((value) => value.get("session-id"))).toEqual(["owner-session", "owner-session"]);
		expect(headers[0]?.get("x-client-request-id")).not.toBe("owner-session");
		expect(headers[0]?.get("x-client-request-id")).not.toBe(headers[1]?.get("x-client-request-id"));
		expect(getOpenAICodexWebSocketDebugStats("owner-session")).toMatchObject({
			requests: 2,
			connectionsCreated: 2,
			connectionsReused: 0,
		});
		cleanupSessionResources("owner-session");
		expect(getOpenAICodexWebSocketDebugStats("owner-session")).toBeUndefined();
	});

	it("settles explicit cancellation and releases the open socket and its listeners", async () => {
		const controller = new AbortController();
		const socket = new OwnedSocket(false);
		const response = streamSimple(model, context, {
			apiKey: token,
			sessionId: "cancel-session",
			signal: controller.signal,
			connectWebSocket: async () => socket,
		});
		for await (const event of response) if (event.type === "start") controller.abort(new Error("user interrupted"));
		expect((await response.result()).stopReason).toBe("aborted");
		expect(socket.close).toHaveBeenCalledOnce();
		expect([...socket.listeners.values()].every((listeners) => listeners.size === 0)).toBe(true);
	});

	it("closes a late connection instead of admitting a response after cancellation", async () => {
		const controller = new AbortController();
		const socket = new OwnedSocket();
		let finish!: (socket: ProviderWebSocket) => void;
		let connected!: () => void;
		const connectionStarted = new Promise<void>((resolve) => {
			connected = resolve;
		});
		const response = streamSimple(model, context, {
			apiKey: token,
			signal: controller.signal,
			connectWebSocket: () => {
				connected();
				return new Promise((resolve) => {
					finish = resolve;
				});
			},
		});
		await connectionStarted;
		controller.abort();
		finish(socket);
		expect((await response.result()).stopReason).toBe("aborted");
		expect(socket.sent).toEqual([]);
		expect(socket.close).toHaveBeenCalledWith(1000, "aborted");
	});

	it("retains one logical attempt identity across attributed WebSocket failure and HTTP fallback", async () => {
		let socketHeaders!: Headers;
		let httpHeaders!: Headers;
		const response = streamSimple(model, context, {
			apiKey: token,
			sessionId: "fallback-session",
			connectWebSocket: async (_url, options) => {
				socketHeaders = options.headers;
				throw new Error("transport disconnected");
			},
			fetch: async (_url, init) => {
				httpHeaders = new Headers(init?.headers);
				return new Response(`data: ${JSON.stringify(terminal)}\n\n`);
			},
		});
		const result = await response.result();
		expect(result.stopReason).toBe("stop");
		expect(httpHeaders.get("x-client-request-id")).toBe(socketHeaders.get("x-client-request-id"));
		expect(httpHeaders.get("session-id")).toBe("fallback-session");
		expect(result.diagnostics).toContainEqual(
			expect.objectContaining({
				type: "provider_transport_failure",
				details: expect.objectContaining({ requestId: socketHeaders.get("x-client-request-id") }),
			}),
		);
		releaseOpenAICodexWebSocketSession("fallback-session");
		expect(getOpenAICodexWebSocketDebugStats("fallback-session")).toBeUndefined();
	});
	it("records response and transport retries without losing request identity or abort listeners", async () => {
		vi.useFakeTimers();
		const controller = new AbortController();
		const add = vi.spyOn(controller.signal, "addEventListener");
		const remove = vi.spyOn(controller.signal, "removeEventListener");
		const requestIds: (string | null)[] = [];
		let calls = 0;
		const response = streamSimple(model, context, {
			apiKey: token,
			transport: "sse",
			maxRetries: 2,
			signal: controller.signal,
			fetch: async (_url, init) => {
				requestIds.push(new Headers(init?.headers).get("x-client-request-id"));
				if (++calls === 1)
					return new Response("rate limited", { status: 429, headers: { "retry-after-ms": "10" } });
				if (calls === 2) throw new Error("connection reset");
				return new Response(`data: ${JSON.stringify(terminal)}\n\n`);
			},
		});
		await vi.runAllTimersAsync();
		const result = await response.result();
		expect(result.stopReason).toBe("stop");
		expect(calls).toBe(3);
		expect(new Set(requestIds).size).toBe(1);
		expect(requestIds[0]).toMatch(/^[0-9a-f-]{36}$/);
		expect(result.diagnostics?.map((entry) => entry.details)).toEqual([
			expect.objectContaining({
				requestId: requestIds[0],
				phase: "sse_response_status",
				attempt: 1,
				nextAttempt: 2,
				status: 429,
				delayMs: 10,
			}),
			expect.objectContaining({
				requestId: requestIds[0],
				phase: "sse_fetch",
				attempt: 2,
				nextAttempt: 3,
				delayMs: 2000,
			}),
		]);
		for (const [type, listener] of add.mock.calls) expect(remove.mock.calls).toContainEqual([type, listener]);
		vi.useRealTimers();
	});

	it.each([400, 401, 403, 422])("does not retry a terminal HTTP %s response", async (status) => {
		const fetch = vi.fn(
			async () => new Response(JSON.stringify({ error: { message: "permanent refusal" } }), { status }),
		);
		const result = await streamSimple(model, context, {
			apiKey: token,
			transport: "sse",
			maxRetries: 3,
			fetch,
		}).result();
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toBe("permanent refusal");
		expect(result.diagnostics).toBeUndefined();
		expect(fetch).toHaveBeenCalledOnce();
	});

	it("cancels the response body and preserves a response observer failure without retry", async () => {
		const cancel = vi.fn();
		const fetch = vi.fn(async () => new Response(new ReadableStream({ cancel })));
		const result = await streamSimple(model, context, {
			apiKey: token,
			transport: "sse",
			maxRetries: 3,
			fetch,
			onResponse: () => {
				throw new Error("observer rejected");
			},
		}).result();
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toBe("observer rejected");
		expect(fetch).toHaveBeenCalledOnce();
		expect(cancel).toHaveBeenCalledOnce();
	});
});
