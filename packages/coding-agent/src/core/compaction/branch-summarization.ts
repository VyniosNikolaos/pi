/**
 * Branch summarization for tree navigation.
 *
 * When navigating to a different point in the session tree, this generates
 * a summary of the branch being left so context isn't lost.
 */

import type { AgentMessage, StreamFn } from "@earendil-works/pi-agent-core";
import type { RetryCallbacks, RetryPolicy } from "@earendil-works/pi-ai";
import { contentText } from "@earendil-works/pi-ai";
import type { Context, Model, SimpleStreamOptions, Usage } from "@earendil-works/pi-ai/compat";
import {
	convertToLlm,
	createBranchSummaryMessage,
	createCompactionSummaryMessage,
	createCustomMessage,
} from "../messages.ts";
import type { ReadonlySessionManager, SessionEntry } from "../session-manager.ts";
import { completeSummarization, estimateTokens } from "./compaction.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
} from "./utils.ts";

// ============================================================================
// Types
// ============================================================================

export interface BranchSummaryResult {
	summary?: string;
	usage?: Usage;
	readFiles?: string[];
	modifiedFiles?: string[];
	aborted?: boolean;
	error?: string;
}

/** Details stored in BranchSummaryEntry.details for file tracking */
export interface BranchSummaryDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

export type { FileOperations } from "./utils.ts";

export interface BranchPreparation {
	/** Messages extracted for summarization, in chronological order */
	messages: AgentMessage[];
	/** File operations extracted from tool calls */
	fileOps: FileOperations;
	/** Total estimated tokens in messages */
	totalTokens: number;
}

export interface CollectEntriesResult {
	/** Entries to summarize, in chronological order */
	entries: SessionEntry[];
	/** Common ancestor between old and new position, if any */
	commonAncestorId: string | null;
}

export interface GenerateBranchSummaryOptions {
	/** Model to use for summarization */
	model: Model<any>;
	/** API key for the model */
	apiKey?: string;
	/** Request headers for the model */
	headers?: Record<string, string>;
	/** Provider-scoped environment values for the model */
	env?: Record<string, string>;
	/** Abort signal for cancellation */
	signal: AbortSignal;
	/** Optional custom instructions for summarization */
	customInstructions?: string;
	/** If true, customInstructions replaces the default prompt instead of being appended */
	replaceInstructions?: boolean;
	/** Tokens reserved for prompt + LLM response (default 16384) */
	reserveTokens?: number;
	/** Optional session stream function. Used to preserve SDK request behavior without mutating agent state. */
	streamFn?: StreamFn;
	/** Retry policy for transient summarization errors. Reuses coding-agent's `settings.retry`. */
	retry?: RetryPolicy;
	/** Optional callbacks for retry reporting (e.g. TUI retry indicators). */
	callbacks?: RetryCallbacks;
	/** Active provider context whose final messages are the branch being abandoned. */
	sourceContext?: Context;
	/** Number of final messages in sourceContext that belong to the abandoned branch. */
	sourceBranchMessageCount?: number;
	/** Provider request hooks and transport settings from the active agent path. */
	onPayload?: SimpleStreamOptions["onPayload"];
	onResponse?: SimpleStreamOptions["onResponse"];
	transport?: SimpleStreamOptions["transport"];
	thinkingBudgets?: SimpleStreamOptions["thinkingBudgets"];
	maxRetryDelayMs?: number;
}

// ============================================================================
// Entry Collection
// ============================================================================

/**
 * Collect entries that should be summarized when navigating from one position to another.
 *
 * Walks from oldLeafId back to the common ancestor with targetId, collecting entries
 * along the way. Does NOT stop at compaction boundaries - those are included and their
 * summaries become context.
 *
 * @param session - Session manager (read-only access)
 * @param oldLeafId - Current position (where we're navigating from)
 * @param targetId - Target position (where we're navigating to)
 * @returns Entries to summarize and the common ancestor
 */
export function collectEntriesForBranchSummary(
	session: ReadonlySessionManager,
	oldLeafId: string | null,
	targetId: string,
): CollectEntriesResult {
	// If no old position, nothing to summarize
	if (!oldLeafId) {
		return { entries: [], commonAncestorId: null };
	}

	// Find common ancestor (deepest node that's on both paths)
	const oldPath = new Set(session.getBranch(oldLeafId).map((e) => e.id));
	const targetPath = session.getBranch(targetId);

	// targetPath is root-first, so iterate backwards to find deepest common ancestor
	let commonAncestorId: string | null = null;
	for (let i = targetPath.length - 1; i >= 0; i--) {
		if (oldPath.has(targetPath[i].id)) {
			commonAncestorId = targetPath[i].id;
			break;
		}
	}

	// Collect entries from old leaf back to common ancestor
	const entries: SessionEntry[] = [];
	let current: string | null = oldLeafId;

	while (current && current !== commonAncestorId) {
		const entry = session.getEntry(current);
		if (!entry) break;
		entries.push(entry);
		current = entry.parentId;
	}

	// Reverse to get chronological order
	entries.reverse();

	return { entries, commonAncestorId };
}

// ============================================================================
// Entry to Message Conversion
// ============================================================================

/**
 * Extract AgentMessage from a session entry.
 * Similar to getMessageFromEntry in compaction.ts but also handles compaction entries.
 */
function getMessageFromEntry(entry: SessionEntry): AgentMessage | undefined {
	switch (entry.type) {
		case "message":
			// Skip tool results - context is in assistant's tool call
			if (entry.message.role === "toolResult") return undefined;
			return entry.message;

		case "custom_message":
			return createCustomMessage(entry.customType, entry.content, entry.display, entry.details, entry.timestamp);

		case "branch_summary":
			return createBranchSummaryMessage(entry.summary, entry.fromId, entry.timestamp);

		case "compaction":
			return createCompactionSummaryMessage(entry.summary, entry.tokensBefore, entry.timestamp);

		// These don't contribute to conversation content
		case "thinking_level_change":
		case "model_change":
		case "custom":
		case "label":
		case "session_info":
			return undefined;
	}
}

/**
 * Prepare entries for summarization with token budget.
 *
 * Walks entries from NEWEST to OLDEST, adding messages until we hit the token budget.
 * This ensures we keep the most recent context when the branch is too long.
 *
 * Also collects file operations from:
 * - Tool calls in assistant messages
 * - Existing branch_summary entries' details (for cumulative tracking)
 *
 * @param entries - Entries in chronological order
 * @param tokenBudget - Maximum tokens to include (0 = no limit)
 */
export function prepareBranchEntries(entries: SessionEntry[], tokenBudget: number = 0): BranchPreparation {
	const messages: AgentMessage[] = [];
	const fileOps = createFileOps();
	let totalTokens = 0;

	// First pass: collect file ops from ALL entries (even if they don't fit in token budget)
	// This ensures we capture cumulative file tracking from nested branch summaries
	// Only extract from pi-generated summaries (fromHook !== true), not extension-generated ones
	for (const entry of entries) {
		if (entry.type === "branch_summary" && !entry.fromHook && entry.details) {
			const details = entry.details as BranchSummaryDetails;
			if (Array.isArray(details.readFiles)) {
				for (const f of details.readFiles) fileOps.read.add(f);
			}
			if (Array.isArray(details.modifiedFiles)) {
				// Modified files go into both edited and written for proper deduplication
				for (const f of details.modifiedFiles) {
					fileOps.edited.add(f);
				}
			}
		}
	}

	// Second pass: walk from newest to oldest, adding messages until token budget
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		const message = getMessageFromEntry(entry);
		if (!message) continue;

		// Extract file ops from assistant messages (tool calls)
		extractFileOpsFromMessage(message, fileOps);

		const tokens = estimateTokens(message);

		// Check budget before adding
		if (tokenBudget > 0 && totalTokens + tokens > tokenBudget) {
			// If this is a summary entry, try to fit it anyway as it's important context
			if (entry.type === "compaction" || entry.type === "branch_summary") {
				if (totalTokens < tokenBudget * 0.9) {
					messages.unshift(message);
					totalTokens += tokens;
				}
			}
			// Stop - we've hit the budget
			break;
		}

		messages.unshift(message);
		totalTokens += tokens;
	}

	return { messages, fileOps, totalTokens };
}

// ============================================================================
// Summary Generation
// ============================================================================

const BRANCH_SUMMARY_PREAMBLE = `The user explored a different conversation branch before returning here.
Summary of that exploration:

`;

const BRANCH_SUMMARY_PROMPT = `Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

/**
 * Build the instruction appended to a cache-friendly active branch context.
 *
 * The source context contains both shared history and the abandoned branch, so the
 * provider-visible suffix length is always included to constrain the summary to the branch.
 * `replaceInstructions` replaces the default summary format, not these boundary and tool-use
 * constraints; removing them could summarize shared history or continue the agent tool loop.
 *
 * @param branchMessageCount Number of final provider-visible messages in the abandoned branch
 * @param customInstructions Optional user-supplied focus or replacement summary instructions
 * @param replaceInstructions Whether custom instructions replace the default summary format
 */
function buildSourceBranchInstructions(
	branchMessageCount: number,
	customInstructions?: string,
	replaceInstructions?: boolean,
) {
	const boundaryInstructions = `The source conversation above is the active conversation branch being abandoned.
Summarize only the final ${branchMessageCount} conversation ${branchMessageCount === 1 ? "message" : "messages"} immediately before this instruction. Those messages are the abandoned branch. Earlier messages are shared context and must not be summarized except where needed to understand the abandoned branch.

Do not call tools. Return only the summary.`;
	if (replaceInstructions && customInstructions) {
		return `${boundaryInstructions}\n\n${customInstructions}`;
	}
	if (customInstructions) {
		return `${boundaryInstructions}\n\n${BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: ${customInstructions}`;
	}
	return `${boundaryInstructions}\n\n${BRANCH_SUMMARY_PROMPT}`;
}

/**
 * Generate a summary of abandoned branch entries.
 *
 * @param entries - Session entries to summarize (chronological order)
 * @param options - Generation options
 * @param sessionId - Optional active routing session used by cache-aware providers
 */
export async function generateBranchSummary(
	entries: SessionEntry[],
	options: GenerateBranchSummaryOptions,
	sessionId?: string,
): Promise<BranchSummaryResult> {
	const {
		model,
		apiKey,
		headers,
		env,
		signal,
		customInstructions,
		replaceInstructions,
		reserveTokens = 16384,
		streamFn,
		retry,
		callbacks,
		sourceContext,
		sourceBranchMessageCount,
		onPayload,
		onResponse,
		transport,
		thinkingBudgets,
		maxRetryDelayMs,
	} = options;

	// Prepare abandoned branch entries within the model's available input budget.
	// File operations are collected from all entries even when older messages are omitted
	// from the standalone summarization prompt by the token budget.
	const contextWindow = model.contextWindow || 128000;
	const tokenBudget = contextWindow - reserveTokens;
	const { messages, fileOps } = prepareBranchEntries(entries, tokenBudget);

	if (messages.length === 0) {
		return { summary: "No content to summarize" };
	}

	// Select the cache-friendly request shape only when the caller supplied a valid
	// provider-visible abandoned suffix. Otherwise preserve standalone summarization.
	const useSourceContext =
		sourceContext !== undefined &&
		sourceBranchMessageCount !== undefined &&
		sourceBranchMessageCount > 0 &&
		sourceBranchMessageCount <= sourceContext.messages.length;

	// Build only an appended branch instruction for a source-context request. The
	// standalone path instead serializes selected branch messages into an isolated prompt.
	let promptText: string;
	if (useSourceContext) {
		promptText = buildSourceBranchInstructions(sourceBranchMessageCount, customInstructions, replaceInstructions);
	} else {
		// Transform to LLM-compatible messages, then serialize to text. Serialization
		// prevents the model from treating the standalone request as a conversation to continue.
		const llmMessages = convertToLlm(messages);
		const conversationText = serializeConversation(llmMessages);
		let instructions: string;
		if (replaceInstructions && customInstructions) {
			instructions = customInstructions;
		} else if (customInstructions) {
			instructions = `${BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: ${customInstructions}`;
		} else {
			instructions = BRANCH_SUMMARY_PROMPT;
		}
		promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${instructions}`;
	}

	// Preserve the complete source prefix when available and append the instruction as a
	// new message. Standalone requests retain the dedicated summarization system prompt.
	const instructionMessage = {
		role: "user" as const,
		content: [{ type: "text" as const, text: promptText }],
		timestamp: Date.now(),
	};
	const context: Context = useSourceContext
		? { ...sourceContext, messages: [...sourceContext.messages, instructionMessage] }
		: { systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: [instructionMessage] };

	// Call the LLM without running through agent state or lifecycle events. The session
	// stream function and active request options preserve SDK/provider behavior, while
	// completeSummarization applies the shared transient-failure retry policy.
	const requestOptions: SimpleStreamOptions = {
		apiKey,
		headers,
		env,
		signal,
		maxTokens: 2048,
		cacheRetention: useSourceContext ? "short" : undefined,
		sessionId,
		onPayload,
		onResponse,
		transport,
		thinkingBudgets,
		maxRetryDelayMs,
	};
	const response = await completeSummarization(model, context, requestOptions, streamFn, retry, callbacks);

	// Convert provider termination states into the branch-summary result contract. Tool
	// calls are rejected because tools are present only to preserve a source-context prefix.
	if (response.stopReason === "aborted") {
		return { aborted: true };
	}
	if (response.stopReason === "error") {
		return { error: response.errorMessage || "Summarization failed" };
	}
	if (response.content.some((block) => block.type === "toolCall")) {
		return { error: "Branch summarization attempted to call a tool" };
	}

	// Mark the generated text as abandoned-branch context, then append cumulative file
	// tracking so later branch summaries and compactions can preserve those operations.
	let summary = contentText(response.content);
	summary = BRANCH_SUMMARY_PREAMBLE + summary;
	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	return {
		summary: summary || "No summary generated",
		usage: response.usage,
		readFiles,
		modifiedFiles,
	};
}
