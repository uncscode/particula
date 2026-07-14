import { tool } from "@opencode-ai/plugin";

import {
  normalizeOptionalString,
  parseSpecOptions,
  runAdwSpecCommand,
  validateAndNormalizeAdwId,
} from "./adw_spec_shared";

// --- Inlined from lib/env_utils ---

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    env[key] = value;
  }
  return env;
}

export default tool({
  description: `Read ADW specification content via focused read/list wrapper.

COMMANDS:
- list: list adw_state fields
- read: read a raw field value (default spec_content)

Focused reads use raw output by default; broad list displays retain CLI redaction.`,

  args: {
    command: tool.schema.enum(["list", "read"]),
    adw_id: tool.schema.string(),
    field: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },

  async execute(args) {
    const { command, adw_id, field, options } = args;
    const normalized = validateAndNormalizeAdwId(command, adw_id);
    if (!normalized.ok) {
      return normalized.error;
    }
    const parsedOptions = parseSpecOptions(command, options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const cmdParts = ["uv", "run", "--active", "adw", "spec", command, "--adw-id", normalized.adwId];
    if (command === "list") {
      if (parsedOptions.options.json) {
        cmdParts.push("--json");
      }
    } else {
      const normalizedField = normalizeOptionalString(field);
      if (normalizedField) {
        cmdParts.push("--field", normalizedField);
      }
      // Focused field reads are machine-to-machine state handoffs. Keep broad
      // list/status displays redacted, but return the selected field verbatim.
      cmdParts.push("--raw");
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
