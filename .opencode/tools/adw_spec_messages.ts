import { tool } from "@opencode-ai/plugin";

import {
  normalizeOptionalString,
  runAdwSpecCommand,
  validateAndNormalizeAdwId,
} from "./adw_spec_shared";

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key === "VIRTUAL_ENV" || value === undefined) continue;
    env[key] = value;
  }
  return env;
}

export default tool({
  description: `Read/write ADW workflow messages via focused wrapper.

COMMANDS:
- messages-write
- messages-read

No contract change versus legacy adw_spec messages behavior.`,

  args: {
    command: tool.schema.enum(["messages-write", "messages-read"]),
    adw_id: tool.schema.string(),
    agent: tool.schema.string().optional(),
    message: tool.schema.string().optional(),
    last: tool.schema.number().optional(),
    raw: tool.schema.boolean().optional(),
  },

  async execute(args) {
    const { command, adw_id, agent, message, last, raw } = args;
    const normalized = validateAndNormalizeAdwId(command, adw_id);
    if (!normalized.ok) {
      return normalized.error;
    }

    const maxLast = 50;
    if (last !== undefined) {
      if (!Number.isInteger(last)) {
        return `ERROR: 'last' must be an integer (0-${maxLast}).`;
      }
      if (last < 0 || last > maxLast) {
        return `ERROR: 'last' must be between 0 and ${maxLast}.`;
      }
    }

    const cmdParts = ["uv", "run", "adw", "spec", "messages"];

    if (command === "messages-write") {
      const normalizedAgent = normalizeOptionalString(agent);
      const normalizedMessage = normalizeOptionalString(message);
      if (!normalizedAgent) {
        return `ERROR: 'agent' parameter is required for messages-write command.

Usage:
  { command: "messages-write", adw_id: "${adw_id}", agent: "planner", message: "Done." }`;
      }
      if (!normalizedMessage) {
        return `ERROR: 'message' parameter is required for messages-write command.

Usage:
  { command: "messages-write", adw_id: "${adw_id}", agent: "planner", message: "Done." }`;
      }

      cmdParts.push(
        "write",
        "--adw-id",
        normalized.adwId,
        "--agent",
        normalizedAgent,
        "--message",
        normalizedMessage,
      );
    } else {
      cmdParts.push("read", "--adw-id", normalized.adwId);
      if (last !== undefined && last > 0) {
        cmdParts.push("--last", String(last));
      }
      if (raw === true) {
        cmdParts.push("--raw");
      }
    }

    const result = runAdwSpecCommand(command, cmdParts, sanitizedEnv());
    if (!result.ok) {
      return result.error;
    }
    return `ADW Spec Command: ${command}\n\n${result.stdout}`;
  },
});
