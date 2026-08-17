import { type Context, createAssistantMessageEventStream, type SimpleStreamOptions } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { assistantMsg, userMsg } from "../../utilities.ts";
import { createHarness, type Harness } from "../harness.ts";

describe("issue #6324 branch summary ambient auth", () => {
	const harnesses: Harness[] = [];
	const originalPiExperimental = process.env.PI_EXPERIMENTAL;

	beforeEach(() => {
		delete process.env.PI_EXPERIMENTAL;
	});

	afterEach(() => {
		if (originalPiExperimental === undefined) {
			delete process.env.PI_EXPERIMENTAL;
		} else {
			process.env.PI_EXPERIMENTAL = originalPiExperimental;
		}
		while (harnesses.length > 0) {
			harnesses.pop()?.cleanup();
		}
	});

	it("summarizes tree branches when request auth has no API key", async () => {
		const harness = await createHarness({ withConfiguredAuth: false });
		harnesses.push(harness);

		let streamCallCount = 0;
		harness.session.agent.streamFunction = (model, _context, options) => {
			streamCallCount++;
			expect(options?.apiKey).toBeUndefined();

			const stream = createAssistantMessageEventStream();
			stream.push({
				type: "done",
				reason: "stop",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "branch summary text" }],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: {
						input: 1,
						output: 1,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 2,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0.25 },
					},
					stopReason: "stop",
					timestamp: Date.now(),
				},
			});
			return stream;
		};

		const targetId = harness.sessionManager.appendMessage(userMsg("first branch"));
		harness.sessionManager.appendMessage(assistantMsg("first reply"));
		harness.sessionManager.appendMessage(userMsg("abandoned branch work"));
		harness.sessionManager.appendMessage(assistantMsg("abandoned reply"));

		const result = await harness.session.navigateTree(targetId, { summarize: true });

		expect(result.cancelled).toBe(false);
		expect(streamCallCount).toBe(1);
		expect(result.summaryEntry?.type).toBe("branch_summary");
		expect(result.summaryEntry?.summary).toContain("branch summary text");
		expect(result.summaryEntry?.usage?.cost.total).toBe(0.25);
	});

	it("uses the active provider prefix for experimental branch summarization", async () => {
		process.env.PI_EXPERIMENTAL = "1";
		const harness = await createHarness({ withConfiguredAuth: false });
		harnesses.push(harness);

		const targetId = harness.sessionManager.appendMessage(userMsg("shared request"));
		harness.sessionManager.appendMessage(assistantMsg("shared reply"));
		harness.sessionManager.appendMessage(userMsg("abandoned request"));
		harness.sessionManager.appendMessage(assistantMsg("abandoned reply"));
		harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;

		let transformContextCalls = 0;
		harness.session.agent.transformContext = async (messages) => {
			transformContextCalls++;
			return messages;
		};
		const originalConvertToLlm = harness.session.agent.convertToLlm;
		harness.session.agent.sessionId = "branch-routing-session";
		harness.session.agent.transport = "websocket";
		harness.session.agent.thinkingBudgets = { low: 1234 };
		harness.session.agent.maxRetryDelayMs = 4321;

		let requestContext: Context | undefined;
		let requestOptions: SimpleStreamOptions | undefined;
		harness.session.agent.streamFunction = (model, context, options) => {
			requestContext = context;
			requestOptions = options;
			const stream = createAssistantMessageEventStream();
			stream.push({
				type: "done",
				reason: "stop",
				message: {
					...assistantMsg("cache-friendly branch summary"),
					api: model.api,
					provider: model.provider,
					model: model.id,
				},
			});
			return stream;
		};

		const activeMessages = harness.sessionManager.buildSessionContext().messages;
		const result = await harness.session.navigateTree(targetId, { summarize: true });

		expect(result.cancelled).toBe(false);
		expect(result.summaryEntry?.summary).toContain("cache-friendly branch summary");
		const expectedPrefix = await originalConvertToLlm(activeMessages);
		expect(requestContext?.messages.slice(0, -1)).toEqual(expectedPrefix);
		expect(transformContextCalls).toBe(1);
		expect(requestContext?.systemPrompt).toBe(harness.session.agent.state.systemPrompt);
		expect(requestContext?.tools).toEqual(harness.session.agent.state.tools);
		const instruction = JSON.stringify(requestContext?.messages.at(-1));
		expect(instruction).toContain("final 3 conversation messages");
		expect(instruction).toContain("Earlier messages are shared context");
		expect(requestOptions).toMatchObject({
			cacheRetention: "none",
			sessionId: "branch-routing-session",
			transport: "websocket",
			thinkingBudgets: { low: 1234 },
			maxRetryDelayMs: 4321,
		});
	});
});
