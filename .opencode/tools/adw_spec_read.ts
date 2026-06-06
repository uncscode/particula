import { tool } from "@opencode-ai/plugin";

import {
  normalizeOptionalString,
  runAdwSpecCommand,
  validateAndNormalizeAdwId,
} from "./adw_spec_shared";

// --- Inlined from lib/env_utils ---

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key === "VIRTUAL_ENV" || value === undefined) continue;
    env[key] = value;
  }
  return env;
}

export default tool({
  description: `Read ADW specification content via focused read/list wrapper.

COMMANDS:
- list: list adw_state fields
- read: read a field value (default spec_content)

No contract change versus legacy adw_spec behavior for read semantics.`,

  args: {
    command: tool.schema.enum(["list", "read"]),
    adw_id: tool.schema.string(),
    field: tool.schema.string().optional(),
    json: tool.schema.boolean().optional(),
    raw: tool.schema.boolean().optional(),
  },

  async execute(args) {
    const { command, adw_id, field, json, raw } = args;
    const normalized = validateAndNormalizeAdwId(command, adw_id);
    if (!normalized.ok) {
      return normalized.error;
    }

    const cmdParts = ["uv", "run", "adw", "spec", command, "--adw-id", normalized.adwId];
    if (command === "list") {
      if (json === true) {
        cmdParts.push("--json");
      }
    } else {
      const normalizedField = normalizeOptionalString(field);
      if (normalizedField) {
        cmdParts.push("--field", normalizedField);
      }
      if (raw === true) {
        cmdParts.push("--raw");
      }
    }

    const result = runAdwSpecCommand(command, cmdParts, sanitizedEnv());
    if (!result.ok) {
      return result.error;
    }

    if (command === "read") {
      return result.stdout;
    }
    return `ADW Spec Command: ${command}\n\n${result.stdout}`;
  },
});
