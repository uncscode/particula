/**
 * Pre Compact Plugin
 *
 * Logs session compaction events to session directory.
 * Equivalent to Claude Code's pre_compact.py hook.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { logToFile } from "../utils";

const PreCompactPlugin: Plugin = async ({ project }) => {
  return {
    "session.compacted": async (event) => {
      try {
        const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                          typeof event.session === 'string' ? event.session :
                          "unknown";

        const logData = {
          event: "session.compacted",
          session_id: sessionId,
          timestamp: new Date().toISOString(),
          ...event,
        };

        await logToFile(sessionId, "pre_compact.json", logData);
      } catch (error) {
        // Fail silently - logging shouldn't break the workflow
        console.error("Pre compact plugin error:", error);
      }
    },
  };
};

export default PreCompactPlugin;
