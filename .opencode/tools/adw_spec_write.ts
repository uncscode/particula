import { tool } from "@opencode-ai/plugin";

import {
  normalizeOptionalString,
  runAdwSpecCommand,
  validateAndNormalizeAdwId,
  validateCanonicalInRepoPath,
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
  description: `Write/delete ADW specification content via focused wrapper.

COMMANDS:
- write
- delete

No contract change versus legacy adw_spec write/delete behavior.`,

  args: {
    command: tool.schema.enum(["write", "delete"]),
    adw_id: tool.schema.string(),
    field: tool.schema.string().optional(),
    content: tool.schema.string().optional(),
    file: tool.schema.string().optional(),
    append: tool.schema.boolean().optional(),
    confirm: tool.schema.boolean().optional(),
  },

  async execute(args) {
    const { command, adw_id, field, content, file, append, confirm } = args;
    const normalized = validateAndNormalizeAdwId(command, adw_id);
    if (!normalized.ok) {
      return normalized.error;
    }

    const cmdParts = ["uv", "run", "adw", "spec", command, "--adw-id", normalized.adwId];

    if (command === "write") {
      const hasContent = typeof content === "string";
      const normalizedFile = normalizeOptionalString(file);
      const normalizedField = normalizeOptionalString(field);
      const hasFile = normalizedFile !== undefined;
      if (!hasContent && !hasFile) {
        return `ERROR: 'write' command requires either 'content' or 'file' parameter (non-empty).

Usage: 
  { command: "write", adw_id: "${adw_id}", content: "New spec content" }
  { command: "write", adw_id: "${adw_id}", file: "path/to/file.md" }`;
      }

      if (normalizedField) {
        cmdParts.push("--field", normalizedField);
      }
      if (hasContent) {
        cmdParts.push("--content", content!);
      }
      if (hasFile) {
        const validatedPath = validateCanonicalInRepoPath(normalizedFile!);
        if (!validatedPath.ok) {
          return validatedPath.error;
        }
        cmdParts.push("--file", validatedPath.canonicalPath);
      }
      if (append === true) {
        cmdParts.push("--append");
      }
    } else {
      const normalizedField = normalizeOptionalString(field);
      if (!normalizedField) {
        return `ERROR: 'delete' command requires 'field' parameter.

Usage: { command: "delete", adw_id: "${adw_id}", field: "field_name" }

Note: Protected fields (adw_id, issue_number) cannot be deleted.`;
      }

      cmdParts.push("--field", normalizedField);
      if (confirm === true) {
        cmdParts.push("--confirm");
      }
    }

    const result = runAdwSpecCommand(command, cmdParts, sanitizedEnv());
    if (!result.ok) {
      return result.error;
    }
    return `ADW Spec Command: ${command}\n\n${result.stdout}`;
  },
});
