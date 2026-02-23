/**
 * Pre Tool Use Plugin
 *
 * Security checks and logging before tool execution.
 * Blocks dangerous commands, .env file access, and infinite loops.
 * Equivalent to Claude Code's pre_tool_use.py hook.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { 
  logToFile, 
  isDangerousRmCommand, 
  isEnvFileAccess,
  loadToolUseLog,
  detectInfiniteLoop
} from "../utils";

const PreToolUsePlugin: Plugin = async ({ project, directory, worktree }) => {
  return {
    "tool.execute.before": async (input: any, output: any) => {
      try {
        const toolName = input.tool || "";
        const toolInput = output.args || {};
        const sessionId = "default"; // Use session ID if available from context

        // Load existing log data to check for infinite loops
        const logData = await loadToolUseLog(sessionId);

        // Check for infinite loop (same tool/input repeated 10+ times)
        if (detectInfiniteLoop(logData, toolName, toolInput)) {
          const errorMsg = 
            "BLOCKED: Infinite loop detected - same tool called 10+ times with identical input\n" +
            `Tool: ${toolName}\n` +
            `Input: ${JSON.stringify(toolInput, null, 2)}\n` +
            "This usually indicates the tool is failing silently. Please check tool output and try a different approach.";
          
          throw new Error(errorMsg);
        }

        // Check for .env file access (blocks access to sensitive environment files)
        // Allow template files (.env.example, .env.sample) — they contain no secrets
        if (isEnvFileAccess(toolName, toolInput)) {
          throw new Error(
            "BLOCKED: Access to .env files containing sensitive data is prohibited. " +
            "Use .env.example or .env.sample for template files instead."
          );
        }

        // Check for dangerous rm -rf commands
        if (toolName === "bash") {
          const command = toolInput.command || "";

          if (isDangerousRmCommand(command)) {
            throw new Error(
              "BLOCKED: Dangerous rm command detected and prevented"
            );
          }
        }

        // Log the tool use (for infinite loop detection and debugging)
        const logEntry = {
          event: "tool.execute.before",
          tool_name: toolName,
          tool_input: toolInput,
          timestamp: new Date().toISOString(),
        };

        await logToFile(sessionId, "pre_tool_use.json", logEntry);

        // Allow the tool to execute
      } catch (error) {
        if (error instanceof Error && error.message.startsWith("BLOCKED:")) {
          // Re-throw blocking errors
          throw error;
        }
        // Log other errors but don't block execution
        console.error("Pre tool use plugin error:", error);
      }
    },
  };
};

export default PreToolUsePlugin;
