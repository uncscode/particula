/**
 * Post Tool Use Plugin
 *
 * Logs tool execution results to session directory.
 * Equivalent to Claude Code's post_tool_use.py hook.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { logToFile } from "../utils";

const PostToolUsePlugin: Plugin = async ({ project }) => {
  return {
    "tool.execute.after": async (event) => {
      try {
        const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                          typeof event.session === 'string' ? event.session :
                          "unknown";

        const logData = {
          event: "tool.execute.after",
          session_id: sessionId,
          tool_name: event.tool?.name,
          tool_input: event.tool?.input,
          tool_output: event.result,
          timestamp: new Date().toISOString(),
          ...event,
        };

        await logToFile(sessionId, "post_tool_use.json", logData);
      } catch (error) {
        // Fail silently - logging shouldn't break the workflow
        console.error("Post tool use plugin error:", error);
      }
    },
  };
};

export default PostToolUsePlugin;
