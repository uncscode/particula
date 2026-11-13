/**
 * User Prompt Submit Plugin
 *
 * Logs user prompts and optionally validates them.
 * Equivalent to Claude Code's user_prompt_submit.py hook.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { logToFile } from "../utils";

/**
 * Validate the user prompt for security or policy violations
 */
function validatePrompt(prompt: string): { isValid: boolean; reason?: string } {
  // Example validation rules (customize as needed)
  const blockedPatterns: Array<[string, string]> = [
    // Add any patterns you want to block
    // Example: ['rm -rf /', 'Dangerous command detected'],
  ];

  const promptLower = prompt.toLowerCase();

  for (const [pattern, reason] of blockedPatterns) {
    if (promptLower.includes(pattern.toLowerCase())) {
      return { isValid: false, reason };
    }
  }

  return { isValid: true };
}

const UserPromptSubmitPlugin: Plugin = async ({ project }) => {
  // Check if validation should be enabled (from environment or config)
  const enableValidation = process.env.OPENCODE_VALIDATE_PROMPTS === "true";
  const logOnly = process.env.OPENCODE_LOG_PROMPTS_ONLY === "true";

  return {
    "message.updated": async (event) => {
      try {
        // Only process user messages
        if (event.message?.role !== "user") {
          return;
        }

        const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                          typeof event.session === 'string' ? event.session :
                          "unknown";
        const prompt = event.message?.content || "";

        // Log the user prompt
        const logData = {
          event: "message.updated",
          session_id: sessionId,
          prompt: prompt,
          timestamp: new Date().toISOString(),
          ...event,
        };

        await logToFile(sessionId, "user_prompt_submit.json", logData);

        // Validate prompt if requested and not in log-only mode
        if (enableValidation && !logOnly && prompt) {
          const { isValid, reason } = validatePrompt(prompt);
          if (!isValid) {
            throw new Error(`Prompt blocked: ${reason}`);
          }
        }

        // Add context information if needed
        // Example: You could modify the prompt here or add additional context
      } catch (error) {
        if (error instanceof Error && error.message.startsWith("Prompt blocked:")) {
          // Re-throw blocking errors
          throw error;
        }
        // Log other errors but don't block
        console.error("User prompt submit plugin error:", error);
      }
    },

    "tui.prompt.append": async (event) => {
      try {
        // Also log when prompts are appended in TUI
        const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                          typeof event.session === 'string' ? event.session :
                          "unknown";

        const logData = {
          event: "tui.prompt.append",
          session_id: sessionId,
          timestamp: new Date().toISOString(),
          ...event,
        };

        await logToFile(sessionId, "user_prompt_submit.json", logData);
      } catch (error) {
        console.error("User prompt submit plugin error:", error);
      }
    },
  };
};

export default UserPromptSubmitPlugin;
