/**
 * Session Stop Plugin
 *
 * Logs session stop events and optionally copies transcript to chat.json.
 * Combines functionality of Claude Code's stop.py and subagent_stop.py hooks.
 */

import type { Plugin } from "@opencode-ai/plugin";
import { logToFile } from "../utils";
import { readFile, writeFile, exists } from "node:fs/promises";
import { join } from "node:path";

/**
 * Copy transcript from .jsonl to chat.json
 */
async function copyTranscriptToChat(
  sessionId: string,
  transcriptPath: string
): Promise<void> {
  try {
    if (!(await exists(transcriptPath))) {
      return;
    }

    // Read .jsonl file and convert to JSON array
    const content = await readFile(transcriptPath, "utf-8");
    const lines = content.split("\n").filter((line) => line.trim());

    const chatData: any[] = [];
    for (const line of lines) {
      try {
        chatData.push(JSON.parse(line));
      } catch {
        // Skip invalid lines
      }
    }

    // Write to session-specific chat.json
    const logDir = join("agents", "_default", sessionId);
    const chatFile = join(logDir, "chat.json");
    await writeFile(chatFile, JSON.stringify(chatData, null, 2));
  } catch (error) {
    // Fail silently
    console.error("Failed to copy transcript:", error);
  }
}

const SessionStopPlugin: Plugin = async ({ project }) => {
  // Check if --chat flag should be enabled (from environment or config)
  const enableChatCopy = process.env.OPENCODE_COPY_CHAT === "true";

  return {
    "session.deleted": async (event) => {
      try {
        const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                          typeof event.session === 'string' ? event.session :
                          "unknown";
        const isSubagent = event.session?.type === "subagent";

        const logData = {
          event: "session.deleted",
          session_id: sessionId,
          is_subagent: isSubagent,
          timestamp: new Date().toISOString(),
          ...event,
        };

        // Log to appropriate file based on type
        const logFile = isSubagent ? "subagent_stop.json" : "stop.json";
        await logToFile(sessionId, logFile, logData);

        // Handle transcript copying if enabled
        if (enableChatCopy && event.session?.transcriptPath) {
          await copyTranscriptToChat(sessionId, event.session.transcriptPath);
        }
      } catch (error) {
        // Fail silently - logging shouldn't break the workflow
        console.error("Session stop plugin error:", error);
      }
    },

    "session.updated": async (event) => {
      try {
        // Also log on session updates that indicate stopping
        if (event.session?.status === "completed" || event.session?.status === "stopped") {
          const sessionId = typeof event.session?.id === 'string' ? event.session.id :
                            typeof event.session === 'string' ? event.session :
                            "unknown";
          const isSubagent = event.session?.type === "subagent";

          const logData = {
            event: "session.updated",
            status: event.session.status,
            session_id: sessionId,
            is_subagent: isSubagent,
            timestamp: new Date().toISOString(),
            ...event,
          };

          const logFile = isSubagent ? "subagent_stop.json" : "stop.json";
          await logToFile(sessionId, logFile, logData);
        }
      } catch (error) {
        console.error("Session stop plugin error:", error);
      }
    },
  };
};

export default SessionStopPlugin;
