/**
 * ADW Spec Management Tool for OpenCode Integration
 * 
 * Provides commands for reading and writing to adw_state.json spec fields.
 * This tool enables agents to manage specification content directly.
 * 
 * See https://opencode.ai/docs/custom-tools/ for OpenCode tool development patterns.
 */

import { tool } from "@opencode-ai/plugin";

import {
  adwIdValidationMessage,
  normalizeAdwId,
  normalizeOptionalString,
  runAdwSpecCommand,
  validateCanonicalInRepoPath,
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

// --- Tool implementation ---

export default tool({
  description: `Manage ADW specification content in adw_state.json files. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

  Read spec:     { command: "read", adw_id: "abc12345" }
  Read field:    { command: "read", adw_id: "abc12345", field: "plan_file" }
  Write spec:    { command: "write", adw_id: "abc12345", content: "New spec" }
  Append:        { command: "write", adw_id: "abc12345", content: "\\n\\nNotes", append: true }
  Write file:    { command: "write", adw_id: "abc12345", file: "plan.md" }
  List fields:   { command: "list", adw_id: "abc12345", json: true }
  Write message: { command: "messages-write", adw_id: "abc12345", agent: "planner", message: "Done." }
  Read messages: { command: "messages-read", adw_id: "abc12345", last: 3, raw: true }

RULES:
- adw_id is required for all commands (8-character hex string).
- read/write default to spec_content field; use field param for others.
- write requires content OR file parameter.
- messages-read: omit 'last' or pass 0 to read all messages.

See .opencode/tools/adw_spec.md for full parameter reference and field list.`,

  args: {
    command: tool.schema
      .enum(["list", "read", "write", "delete", "messages-write", "messages-read"])
      .describe(`Spec command to execute: list, read, write, delete, messages-write, messages-read.`),

    adw_id: tool.schema
      .string()
      .describe(`ADW workflow ID (8-character hex string).

REQUIRED FOR: All commands

This identifies which workflow state file to operate on.
State files are located at: agents/{adw_id}/adw_state.json

EXAMPLE: adw_id: "abc12345"
GET ACTIVE IDS: Use the main ADW tool with command: "status"`),

    field: tool.schema
      .string()
      .optional()
      .describe(`Field name to read, write, or delete.

DEFAULT FOR 'read'/'write': spec_content
REQUIRED FOR 'delete': Yes (must specify field to delete)

COMMON FIELDS:
• spec_content - Main implementation specification/plan
• branch_name - Git branch name for the workflow
• workflow_type - Type of workflow (complete, patch, document, generate)
• model_tier - AI model tier (light, base, heavy)
• current_workflow - Current workflow name
• current_step - Current step in workflow
• pr_url - Pull request URL (if created)
• pr_number - Pull request number (if created)

Use 'adw spec list --adw-id <id>' to see all available fields for a workflow.

EXAMPLE: field: "spec_content"`),

    content: tool.schema
      .string()
      .optional()
      .describe(`Content to write to the field.

REQUIRED FOR: 'write' command (unless using 'file' parameter)

Can contain any string content including:
• Implementation specifications
• Planning notes
• Status updates
• Multi-line markdown content

For multi-line content, use newline characters (\\n).

EXAMPLE: content: "# Implementation Plan\\n\\n## Step 1\\n..."`),

    file: tool.schema
      .string()
      .optional()
      .describe(`Path to file containing content to write.

ALTERNATIVE TO: 'content' parameter for 'write' command

Use this when content is stored in a file rather than passed directly.
The entire file content will be read and written to the specified field.

EXAMPLE: file: "path/to/plan.md"`),

    append: tool.schema
      .boolean()
      .optional()
      .describe(`Append to existing content instead of replacing.

APPLIES TO: 'write' command only

When true, new content is appended to the end of existing field content.
Only works with string fields (spec_content, plan_file, etc.).

EXAMPLE: append: true`),

    json: tool.schema
      .boolean()
      .optional()
      .describe(`Output as JSON instead of formatted table.

APPLIES TO: 'list' command only

When true, outputs the complete adw_state.json as JSON instead of a
formatted table. Useful for programmatic processing.

EXAMPLE: json: true`),

    raw: tool.schema
      .boolean()
      .optional()
      .describe(`Output raw content without formatting.

APPLIES TO: 'read' command only

When true, outputs only the field value without any formatting or labels.
Useful for piping to other tools or saving to files.

EXAMPLE: raw: true`),

    confirm: tool.schema
      .boolean()
      .optional()
      .describe(`Skip confirmation prompt for delete command.

APPLIES TO: 'delete' command only

When true, deletes the field immediately without asking for confirmation.
Use with caution as deleted fields cannot be recovered easily.

EXAMPLE: confirm: true`),

    agent: tool.schema
      .string()
      .optional()
      .describe(`Agent name for messages-write command.

REQUIRED FOR: messages-write

EXAMPLE: agent: "planner"`),

    message: tool.schema
      .string()
      .optional()
      .describe(`Message text for messages-write command.

REQUIRED FOR: messages-write

EXAMPLE: message: "Done."`),

    last: tool.schema
      .number()
      .optional()
      .describe(`Return the last N messages for messages-read command.

OPTIONAL FOR: messages-read

Omit this field or pass 0 to read all persisted messages. Use a positive
integer to limit the output to that many recent messages.

EXAMPLE: last: 3`),
  },

  async execute(args) {
    const {
      command,
      adw_id,
      field,
      content,
      file,
      append,
      json,
      raw,
      confirm,
      agent,
      message,
      last,
    } = args;

    // Command validation — delegate to canonical adw_id.ts normalizer
    if (typeof adw_id !== "string" || adw_id.trim().length === 0) {
      return `ERROR: 'adw_id' parameter is required for all spec commands.

Usage: Use the ADW tool with command "status" to see active workflow IDs.
Example: { command: "${command}", adw_id: "abc12345" }`;
    }

    const normalizedId = normalizeAdwId(adw_id);
    if (!normalizedId) {
      return `ERROR: ${adwIdValidationMessage()}`;
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

    // Build CLI command
    const cmdParts = ["uv", "run", "adw", "spec"];

    // Add command-specific parameters
    switch (command) {
      case "list":
        cmdParts.push("list", "--adw-id", normalizedId);
        if (json) {
          cmdParts.push("--json");
        }
        break;

      case "read":
        cmdParts.push("read", "--adw-id", normalizedId);
        {
          const normalizedField = normalizeOptionalString(field);
          if (normalizedField) {
            cmdParts.push("--field", normalizedField);
          }
        }
        // Preserve CLI formatting by default so help panels and rich output remain intact.
        // Callers that need raw content should pass the explicit --raw flag.
        if (raw) {
          cmdParts.push("--raw");
        }
        break;

      case "write": {
        const hasContent = typeof content === "string";
        const normalizedFile = normalizeOptionalString(file);
        const hasFile = normalizedFile !== undefined;
        if (!hasContent && !hasFile) {
          return `ERROR: 'write' command requires either 'content' or 'file' parameter (non-empty).

Usage: 
  { command: "write", adw_id: "${adw_id}", content: "New spec content" }
  { command: "write", adw_id: "${adw_id}", file: "path/to/file.md" }`;
        }
        cmdParts.push("write", "--adw-id", normalizedId);
        {
          const normalizedField = normalizeOptionalString(field);
          if (normalizedField) {
            cmdParts.push("--field", normalizedField);
          }
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
        if (append) {
          cmdParts.push("--append");
        }
        break;
      }

      case "delete":
        if (!normalizeOptionalString(field)) {
          return `ERROR: 'delete' command requires 'field' parameter.

Usage: { command: "delete", adw_id: "${adw_id}", field: "field_name" }

Note: Protected fields (adw_id, issue_number) cannot be deleted.`;
        }
        cmdParts.push("delete", "--adw-id", normalizedId, "--field", normalizeOptionalString(field)!);
        if (confirm) {
          cmdParts.push("--confirm");
        }
        break;

      case "messages-write":
        if (!normalizeOptionalString(agent)) {
          return `ERROR: 'agent' parameter is required for messages-write command.

Usage:
  { command: "messages-write", adw_id: "${adw_id}", agent: "planner", message: "Done." }`;
        }
        if (!normalizeOptionalString(message)) {
          return `ERROR: 'message' parameter is required for messages-write command.

Usage:
  { command: "messages-write", adw_id: "${adw_id}", agent: "planner", message: "Done." }`;
        }
        cmdParts.push(
          "messages",
          "write",
          "--adw-id",
          normalizedId,
          "--agent",
          normalizeOptionalString(agent)!,
          "--message",
          normalizeOptionalString(message)!,
        );
        break;

      case "messages-read":
        cmdParts.push("messages", "read", "--adw-id", normalizedId);
        if (last !== undefined && last > 0) {
          cmdParts.push("--last", String(last));
        }
        if (raw) {
          cmdParts.push("--raw");
        }
        break;
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
