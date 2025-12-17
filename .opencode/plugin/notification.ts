/**
 * Notification Plugin
 *
 * Logs notification events to session directory.
 * Equivalent to Claude Code's notification.py hook.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { logToFile } from "../utils";

const NotificationPlugin: Plugin = async ({ project, directory, worktree }) => {
  return {
    "session.idle": async (event: any) => {
      try {
        const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                          typeof event.session === 'string' ? event.session :
                          "unknown";

        const logData = {
          event: "session.idle",
          session_id: sessionId,
          timestamp: new Date().toISOString(),
          ...event,
        };

        await logToFile(sessionId, "notification.json", logData);
      } catch (error) {
        // Fail silently - logging shouldn't break the workflow
        console.error("Notification plugin error:", error);
      }
    },
  };
};

export default NotificationPlugin;
