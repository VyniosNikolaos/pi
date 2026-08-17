import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type Context,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type Model,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { generateBranchSummary } from "../src/core/compaction/index.ts";
import type { SessionEntry } from "../src/core/session-manager.ts";

const model: Model<"anthropic-messages"> = {
	id: "test-model",
	name: "Test Model",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200000,
	maxTokens: 8192,
};

function branchEntries(): SessionEntry[] {
	return [
		{
			type: "message",
			id: "branch-user",
			parentId: "shared-assistant",
			timestamp: new Date(1).toISOString(),
			message: { role: "user", content: "Abandoned request", timestamp: 1 },
		},
		{
			type: "message",
			id: "branch-assistant",
			parentId: "branch-user",
			timestamp: new Date(2).toISOString(),
			message: fauxAssistantMessage("Abandoned answer", { timestamp: 2 }),
		},
	];
}

function response(content: AssistantMessage["content"]): AssistantMessage {
	return {
		...fauxAssistantMessage(""),
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
	};
}

describe("branch summarization contexts", () => {
	it("appends a branch-bounded instruction to an active provider context", async () => {
		const sourceContext: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [
				{ role: "user", content: "Shared request", timestamp: 1 },
				fauxAssistantMessage("Shared answer", { timestamp: 2 }),
				{ role: "user", content: "Abandoned request", timestamp: 3 },
				fauxAssistantMessage("Abandoned answer", { timestamp: 4 }),
			],
			tools: [],
		};
		let requestContext: Context | undefined;
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, context, options) => {
			requestContext = context;
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};

		const result = await generateBranchSummary(
			branchEntries(),
			{
				model,
				signal: new AbortController().signal,
				streamFn,
				sourceContext,
				sourceBranchMessageCount: 2,
				transport: "websocket",
				thinkingBudgets: { low: 1234 },
				maxRetryDelayMs: 4321,
			},
			"active-session",
		);

		expect(result.summary).toContain("summary");
		expect(requestContext?.systemPrompt).toBe(sourceContext.systemPrompt);
		expect(requestContext?.tools).toBe(sourceContext.tools);
		expect(requestContext?.messages.slice(0, -1)).toEqual(sourceContext.messages);
		const instruction = JSON.stringify(requestContext?.messages.at(-1));
		expect(instruction).toContain("final 2 conversation messages");
		expect(instruction).toContain("Earlier messages are shared context");
		expect(instruction).toContain("Do not call tools. Return only the summary.");
		expect(instruction).not.toContain("<conversation>");
		expect(requestOptions).toMatchObject({
			cacheRetention: "short",
			sessionId: "active-session",
			transport: "websocket",
			thinkingBudgets: { low: 1234 },
			maxRetryDelayMs: 4321,
		});
	});

	it("keeps standalone branch summaries isolated from prompt caching", async () => {
		let requestContext: Context | undefined;
		let requestOptions: SimpleStreamOptions | undefined;
		const streamFn: StreamFn = (_model, context, options) => {
			requestContext = context;
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({ type: "done", reason: "stop", message: response([{ type: "text", text: "summary" }]) }),
			);
			return stream;
		};

		await generateBranchSummary(branchEntries(), {
			model,
			signal: new AbortController().signal,
			streamFn,
		});

		expect(requestContext?.systemPrompt).not.toBe("You are a coding agent.");
		expect(JSON.stringify(requestContext?.messages)).toContain("<conversation>");
		expect(requestOptions?.cacheRetention).toBe("none");
	});

	it("rejects tool calls from cache-friendly branch summaries", async () => {
		const sourceContext: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [{ role: "user", content: "Abandoned request", timestamp: 1 }],
			tools: [],
		};
		const streamFn: StreamFn = () => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() =>
				stream.push({
					type: "done",
					reason: "toolUse",
					message: response([{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "README.md" } }]),
				}),
			);
			return stream;
		};

		const result = await generateBranchSummary(branchEntries(), {
			model,
			signal: new AbortController().signal,
			streamFn,
			sourceContext,
			sourceBranchMessageCount: 1,
		});

		expect(result.error).toBe("Branch summarization attempted to call a tool");
	});
});
