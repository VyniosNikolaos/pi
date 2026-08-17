import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Context, Model } from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
	type CompactionPreparation,
	compact,
	completeSummarization,
	generateSummary,
	generateSummaryWithUsage,
} from "../src/core/compaction/index.ts";

const { completeSimpleMock } = vi.hoisted(() => ({
	completeSimpleMock: vi.fn(),
}));

vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@earendil-works/pi-ai/compat")>();
	return {
		...actual,
		completeSimple: completeSimpleMock,
	};
});

function createModel(reasoning: boolean, maxTokens = 8192): Model<"anthropic-messages"> {
	return {
		id: reasoning ? "reasoning-model" : "non-reasoning-model",
		name: reasoning ? "Reasoning Model" : "Non-reasoning Model",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens,
	};
}

const mockSummaryResponse: AssistantMessage = {
	role: "assistant",
	content: [{ type: "text", text: "## Goal\nTest summary" }],
	api: "anthropic-messages",
	provider: "anthropic",
	model: "claude-sonnet-4-5",
	usage: {
		input: 10,
		output: 10,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 20,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	},
	stopReason: "stop",
	timestamp: Date.now(),
};

const mockToolCallResponse: AssistantMessage = {
	...mockSummaryResponse,
	content: [{ type: "toolCall", id: "tool-call-1", name: "read", arguments: { path: "README.md" } }],
	stopReason: "toolUse",
};

const messages: AgentMessage[] = [{ role: "user", content: "Summarize this.", timestamp: Date.now() }];

describe("generateSummary reasoning options", () => {
	beforeEach(() => {
		completeSimpleMock.mockReset();
		completeSimpleMock.mockResolvedValue(mockSummaryResponse);
	});

	it("uses the provided thinking level for reasoning-capable models", async () => {
		const result = await generateSummaryWithUsage(
			messages,
			createModel(true),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			"medium",
		);

		expect(result.text).toBe("## Goal\nTest summary");
		expect(result.usage).toEqual(mockSummaryResponse.usage);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			reasoning: "medium",
			apiKey: "test-key",
		});
	});

	it("preserves the string result from generateSummary", async () => {
		await expect(generateSummary(messages, createModel(false), 2000, "test-key")).resolves.toBe(
			"## Goal\nTest summary",
		);
	});

	it("uses fresh routing sessions without prompt caching", async () => {
		await generateSummary(messages, createModel(false), 2000, "test-key");
		await generateSummary(messages, createModel(false), 2000, "test-key");

		const requestOptions = completeSimpleMock.mock.calls.map((call) => call[2]);
		expect(requestOptions).toHaveLength(2);
		expect(requestOptions.every((options) => options?.cacheRetention === "none")).toBe(true);

		const sessionIds = requestOptions.map((options) => options?.sessionId);
		expect(sessionIds[0]).not.toBe(sessionIds[1]);
	});

	it("honors a caller-supplied routing session without prompt caching", async () => {
		await completeSummarization(
			createModel(false),
			{ systemPrompt: "Summarize", messages: [] },
			{ sessionId: "current-routing-session", cacheRetention: "long" },
		);

		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			sessionId: "current-routing-session",
			cacheRetention: "none",
		});
	});

	it("forwards optional provider request arguments", async () => {
		const onPayload = async (payload: unknown) => payload;
		const onResponse = async () => {};
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: messages,
			turnPrefixMessages: [],
			isSplitTurn: false,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await compact(
			preparation,
			createModel(false),
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			"routing-session",
			onPayload,
			onResponse,
			"websocket",
			{ low: 1234, medium: 5678 },
			4321,
		);

		const requestOptions = completeSimpleMock.mock.calls[0][2];
		expect(requestOptions).toMatchObject({
			cacheRetention: "none",
			sessionId: "routing-session",
			transport: "websocket",
			thinkingBudgets: { low: 1234, medium: 5678 },
			maxRetryDelayMs: 4321,
		});
		expect(requestOptions?.onPayload).toBe(onPayload);
		expect(requestOptions?.onResponse).toBe(onResponse);
	});

	it("appends the summarization instruction to a cache-friendly source context", async () => {
		const sourceContext: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [{ role: "user", content: "Previous summary and original request", timestamp: 1 }],
			tools: [],
		};

		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: messages,
			sourceMessages: messages,
			turnPrefixMessages: [],
			turnPrefixSourceMessages: [],
			isSplitTurn: false,
			tokensBefore: 100,
			previousSummary: "Previous summary",
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await compact(
			preparation,
			createModel(false),
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined, // sessionId
			undefined, // onPayload
			undefined, // onResponse
			undefined, // transport
			undefined, // thinkingBudgets
			undefined, // maxRetryDelayMs
			sourceContext,
		);

		const requestContext = completeSimpleMock.mock.calls[0][1] as Context;
		expect(requestContext.systemPrompt).toBe(sourceContext.systemPrompt);
		expect(requestContext.tools).toBe(sourceContext.tools);
		expect(requestContext.messages.slice(0, -1)).toEqual(sourceContext.messages);
		expect(requestContext.messages).toHaveLength(sourceContext.messages.length + 1);
		expect(requestContext.messages.at(-1)).toMatchObject({ role: "user" });
		expect(JSON.stringify(requestContext.messages.at(-1))).toContain(
			"Create a structured context checkpoint summary",
		);
		expect(JSON.stringify(requestContext.messages.at(-1))).not.toContain("context summarization assistant");
		expect(JSON.stringify(requestContext.messages.at(-1))).not.toContain("<conversation>");
		expect(JSON.stringify(requestContext.messages.at(-1))).not.toContain("<previous-summary>");
		expect(JSON.stringify(requestContext.messages.at(-1))).not.toContain("NEW conversation messages");
		expect(sourceContext.messages).toHaveLength(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({ cacheRetention: "none" });
	});

	it("rejects tool calls from a source-context history summary", async () => {
		completeSimpleMock.mockResolvedValueOnce(mockToolCallResponse);
		const sourceContext: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [{ role: "user", content: "Inspect README.md", timestamp: 1 }],
			tools: [],
		};
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: messages,
			turnPrefixMessages: [],
			isSplitTurn: false,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await expect(
			compact(
				preparation,
				createModel(false),
				"test-key",
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined, // sessionId
				undefined, // onPayload
				undefined, // onResponse
				undefined, // transport
				undefined, // thinkingBudgets
				undefined, // maxRetryDelayMs
				sourceContext,
			),
		).rejects.toThrow("Summarization attempted to call a tool");
	});

	it("limits a source-context turn-prefix summary to the final incomplete turn", async () => {
		const earlierUser = { role: "user" as const, content: "Earlier request", timestamp: 1 };
		const earlierAssistant: AssistantMessage = {
			...mockSummaryResponse,
			content: [{ type: "text", text: "Earlier work completed" }],
			timestamp: 2,
		};
		const splitUser = { role: "user" as const, content: "Large final request", timestamp: 3 };
		const earlyAssistant: AssistantMessage = {
			...mockSummaryResponse,
			content: [{ type: "text", text: "Early work in final turn" }],
			timestamp: 4,
		};
		const turnPrefixSourceContext: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [earlierUser, earlierAssistant, splitUser, earlyAssistant],
			tools: [],
		};
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: [],
			turnPrefixMessages: [splitUser, earlyAssistant],
			isSplitTurn: true,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await compact(
			preparation,
			createModel(false),
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined, // sessionId
			undefined, // onPayload
			undefined, // onResponse
			undefined, // transport
			undefined, // thinkingBudgets
			undefined, // maxRetryDelayMs
			undefined, // sourceContext
			turnPrefixSourceContext,
		);

		const requestContext = completeSimpleMock.mock.calls[0][1] as Context;
		expect(requestContext.messages.slice(0, -1)).toEqual(turnPrefixSourceContext.messages);
		const instruction = JSON.stringify(requestContext.messages.at(-1));
		expect(instruction).toContain("Summarize only the final, incomplete turn");
		expect(instruction).toContain("last user-role request before this instruction");
		expect(instruction).toContain("Do not summarize earlier turns");
		expect(instruction).not.toContain("Earlier request");
		expect(turnPrefixSourceContext.messages).toHaveLength(4);
	});

	it("rejects tool calls from a source-context turn-prefix summary", async () => {
		completeSimpleMock.mockResolvedValueOnce(mockToolCallResponse);
		const turnPrefixSourceContext: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [{ role: "user", content: "Inspect README.md", timestamp: 1 }],
			tools: [],
		};
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: [],
			turnPrefixMessages: messages,
			isSplitTurn: true,
			tokensBefore: 100,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 2000, keepRecentTokens: 20 },
		};

		await expect(
			compact(
				preparation,
				createModel(false),
				"test-key",
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined,
				undefined, // sessionId
				undefined, // onPayload
				undefined, // onResponse
				undefined, // transport
				undefined, // thinkingBudgets
				undefined, // maxRetryDelayMs
				undefined, // sourceContext
				turnPrefixSourceContext,
			),
		).rejects.toThrow("Turn prefix summarization attempted to call a tool");
	});

	it("does not set reasoning when thinking is off", async () => {
		await generateSummary(
			messages,
			createModel(true),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			"off",
		);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			apiKey: "test-key",
		});
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("reasoning");
	});

	it("does not set reasoning for non-reasoning models", async () => {
		await generateSummary(
			messages,
			createModel(false),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			undefined,
			"medium",
		);

		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
		expect(completeSimpleMock.mock.calls[0][2]).toMatchObject({
			apiKey: "test-key",
		});
		expect(completeSimpleMock.mock.calls[0][2]).not.toHaveProperty("reasoning");
	});

	it("clamps compaction summary maxTokens to the model output cap", async () => {
		const preparation: CompactionPreparation = {
			firstKeptEntryId: "entry-keep",
			messagesToSummarize: messages,
			sourceMessages: messages,
			turnPrefixMessages: messages,
			turnPrefixSourceMessages: messages,
			isSplitTurn: true,
			tokensBefore: 600000,
			fileOps: { read: new Set(), written: new Set(), edited: new Set() },
			settings: { enabled: true, reserveTokens: 500000, keepRecentTokens: 20000 },
		};

		const result = await compact(preparation, createModel(false, 128000), "test-key");

		expect(result.usage).toEqual({
			...mockSummaryResponse.usage,
			input: 20,
			output: 20,
			totalTokens: 40,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		});
		expect(completeSimpleMock.mock.calls.map((call) => call[2]?.maxTokens)).toEqual([128000, 128000]);
	});
});
