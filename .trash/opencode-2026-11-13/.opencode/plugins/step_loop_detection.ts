/**
 * Step Loop Detection Plugin
 *
 * Detects when the agent is stuck in a loop of empty steps without making progress.
 * This catches issues where the agent thinks it's making tool calls, but OpenCode
 * isn't recognizing them (e.g., malformed XML function calls, incorrect syntax).
 *
 * Unlike pre_tool_use.ts which detects repeated tool invocations, this plugin
 * detects repeated step_finish events with minimal output, indicating the agent
 * is stuck in a non-productive loop.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { logToFile } from "../utils";

interface StepInfo {
  timestamp: number;
  outputTokens: number;
  reason: string;
}

// Track step history per session
const sessionStepHistory = new Map<string, StepInfo[]>();

// Configuration
const MAX_EMPTY_STEPS = 20; // Trigger after 20 consecutive low-output steps
const MIN_OUTPUT_TOKENS = 15; // Steps with fewer tokens are considered "empty"
const WINDOW_SIZE = 25; // Look at last 25 steps

/**
 * Detect if the agent is stuck in an empty step loop
 */
function detectEmptyStepLoop(
  sessionId: string,
  currentStep: StepInfo
): { isLoop: boolean; message?: string } {
  // Get or create step history for this session
  if (!sessionStepHistory.has(sessionId)) {
    sessionStepHistory.set(sessionId, []);
  }

  const history = sessionStepHistory.get(sessionId)!;

  // Add current step
  history.push(currentStep);

  // Keep only recent history (avoid unbounded memory growth)
  if (history.length > WINDOW_SIZE) {
    history.shift();
  }

  // Need enough steps to detect a pattern
  if (history.length < MAX_EMPTY_STEPS) {
    return { isLoop: false };
  }

  // Count consecutive low-output steps
  const recentSteps = history.slice(-MAX_EMPTY_STEPS);
  const emptySteps = recentSteps.filter(
    (step) => step.outputTokens < MIN_OUTPUT_TOKENS
  );

  // Check if most recent steps are empty
  if (emptySteps.length >= MAX_EMPTY_STEPS * 0.9) {
    // 90% threshold
    const totalTokens = recentSteps.reduce(
      (sum, step) => sum + step.outputTokens,
      0
    );
    const avgTokens = totalTokens / recentSteps.length;

    const message =
      `LOOP DETECTED: Agent stuck in empty step loop\n` +
      `- ${emptySteps.length}/${MAX_EMPTY_STEPS} recent steps have minimal output\n` +
      `- Average output: ${avgTokens.toFixed(1)} tokens (threshold: ${MIN_OUTPUT_TOKENS})\n` +
      `- Step reason: ${currentStep.reason}\n` +
      `\n` +
      `This usually indicates:\n` +
      `1. Agent is outputting malformed tool calls (e.g., XML syntax)\n` +
      `2. Tool calls aren't being recognized by OpenCode\n` +
      `3. Agent instructions are causing confusion\n` +
      `\n` +
      `Recommendation: Check the command/prompt for ambiguous instructions\n` +
      `that might cause the agent to output text instead of using tools.`;

    return { isLoop: true, message };
  }

  return { isLoop: false };
}

/**
 * Clear session history when session stops
 */
function clearSessionHistory(sessionId: string): void {
  sessionStepHistory.delete(sessionId);
}

const StepLoopDetectionPlugin: Plugin = async ({ project }) => {
  return {
    // Monitor step completion events
    "message.created": async (event) => {
      try {
        // Extract session ID
        const sessionId =
          typeof event.session?.id === "string"
            ? event.session.id
            : typeof event.session === "string"
              ? event.session
              : "unknown";

        // Check if this is a step_finish event
        const part = event.message?.parts?.[0];
        if (part?.type === "step-finish") {
          const stepInfo: StepInfo = {
            timestamp: Date.now(),
            outputTokens: part.tokens?.output || 0,
            reason: part.reason || "unknown",
          };

          // Detect loop
          const result = detectEmptyStepLoop(sessionId, stepInfo);

          if (result.isLoop) {
            // Log the detection
            await logToFile(sessionId, "loop_detection.json", {
              event: "empty_step_loop_detected",
              session_id: sessionId,
              timestamp: new Date().toISOString(),
              details: result.message,
              step_info: stepInfo,
            });

            // Output warning to console (visible in logs)
            console.error("\n" + "=".repeat(80));
            console.error(result.message);
            console.error("=".repeat(80) + "\n");

            // TODO: Optionally terminate the session here
            // For now, just log and warn
          }
        }
      } catch (error) {
        // Fail silently - monitoring shouldn't break the workflow
        console.error("Step loop detection plugin error:", error);
      }
    },

    // Clean up when session stops
    "session.deleted": async (event) => {
      try {
        const sessionId =
          typeof event.session?.id === "string"
            ? event.session.id
            : typeof event.session === "string"
              ? event.session
              : "unknown";

        clearSessionHistory(sessionId);
      } catch (error) {
        console.error("Step loop detection cleanup error:", error);
      }
    },
  };
};

export default StepLoopDetectionPlugin;
