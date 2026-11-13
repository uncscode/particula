import { tool } from "@opencode-ai/plugin";

import {
  normalizeOptionalString,
  parseSpecOptions,
  runAdwSpecCommand,
  validateAndNormalizeAdwId,
} from "./adw_spec_shared";

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
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
    options: tool.schema.string().optional(),
  },

  async execute(args) {
    const { command, adw_id, agent, message, options } = args;
    const normalized = validateAndNormalizeAdwId(command, adw_id);
    if (!normalized.ok) {
      return normalized.error;
    }

    const parsedOptions = parseSpecOptions(command, options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const cmdParts = ["uv", "run", "--active", "adw", "spec", "messages"];

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
      if (parsedOptions.options.last !== undefined && parsedOptions.options.last > 0) {
        cmdParts.push("--last", String(parsedOptions.options.last));
      }
      if (parsedOptions.options.raw) {
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
