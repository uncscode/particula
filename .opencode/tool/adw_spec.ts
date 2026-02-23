/**
 * ADW Spec Management Tool for OpenCode Integration
 * 
 * Provides commands for reading and writing to adw_state.json spec fields.
 * This tool enables agents to manage specification content directly.
 * 
 * See https://opencode.ai/docs/custom-tools/ for OpenCode tool development patterns.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: `Manage ADW specification content in adw_state.json files.

The spec tool provides commands to read and write specification content
stored in the adw_state.json file for each workflow. This includes:
- spec_content: Main implementation specification/plan
- plan_file: Path to the plan markdown file  
- Other metadata fields

AVAILABLE COMMANDS:
• list: List all fields in adw_state.json
  Usage: { command: "list", adw_id: "abc12345" }
  With JSON: { command: "list", adw_id: "abc12345", json: true }

• read: Read a field from adw_state.json (defaults to spec_content)
  Usage: { command: "read", adw_id: "abc12345" }
  Specific field: { command: "read", adw_id: "abc12345", field: "plan_file" }
  Raw output: { command: "read", adw_id: "abc12345", raw: true }

• write: Write content to a field (defaults to spec_content)
  Usage: { command: "write", adw_id: "abc12345", content: "New spec" }
  From file: { command: "write", adw_id: "abc12345", file: "plan.md" }
  Append: { command: "write", adw_id: "abc12345", content: "\\n\\nNotes", append: true }
  Other field: { command: "write", adw_id: "abc12345", field: "plan_file", content: "path/to/plan.md" }

• delete: Delete a field from adw_state.json (field parameter required)
  Usage: { command: "delete", adw_id: "abc12345", field: "custom_field" }
  Skip confirm: { command: "delete", adw_id: "abc12345", field: "custom_field", confirm: true }

COMMON USE CASES:
• Read current spec: { command: "read", adw_id: "abc12345" }
• Update spec: { command: "write", adw_id: "abc12345", content: "Updated plan..." }
• List all fields: { command: "list", adw_id: "abc12345", json: true }
• Append to spec: { command: "write", adw_id: "abc12345", content: "\\n\\nNotes", append: true }`,

  args: {
    command: tool.schema
      .enum(["list", "read", "write", "delete"])
      .describe(`Spec management command to execute.

COMMAND DESCRIPTIONS:
• list - List all fields in adw_state.json with types and previews
• read - Read a specific field from adw_state.json (defaults to spec_content)
• write - Write content to a field (defaults to spec_content)
• delete - Delete a field from adw_state.json (protected fields like adw_id, issue_number cannot be deleted)

REQUIRED PARAMETERS BY COMMAND:
• list: adw_id, optional: json
• read: adw_id, optional: field (default: spec_content), raw
• write: adw_id, (content OR file), optional: field (default: spec_content), append
• delete: adw_id, field (required), optional: confirm`),

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
  },

  async execute(args) {
    const { command, adw_id, field, content, file, append, json, raw, confirm } = args;

    // Command validation
    if (!adw_id) {
      return `ERROR: 'adw_id' parameter is required for all spec commands.

Usage: Use the ADW tool with command "status" to see active workflow IDs.
Example: { command: "${command}", adw_id: "abc12345" }`;
    }

    // Build CLI command
    const cmdParts = ["uv", "run", "adw", "spec", command, "--adw-id", adw_id];

    // Add command-specific parameters
    switch (command) {
      case "list":
        if (json) {
          cmdParts.push("--json");
        }
        break;

      case "read":
        if (field) {
          cmdParts.push("--field", field);
        }
        // Preserve CLI formatting by default so help panels and rich output remain intact.
        // Callers that need raw content should pass the explicit --raw flag.
        if (raw) {
          cmdParts.push("--raw");
        }
        break;

      case "write":
        if (!content && !file) {
          return `ERROR: 'write' command requires either 'content' or 'file' parameter.

Usage: 
  { command: "write", adw_id: "${adw_id}", content: "New spec content" }
  { command: "write", adw_id: "${adw_id}", file: "path/to/file.md" }`;
        }

        if (field) {
          cmdParts.push("--field", field);
        }
        if (content) {
          cmdParts.push("--content", content);
        }
        if (file) {
          cmdParts.push("--file", file);
        }
        if (append) {
          cmdParts.push("--append");
        }
        break;

      case "delete":
        if (!field) {
          return `ERROR: 'delete' command requires 'field' parameter.

Usage: { command: "delete", adw_id: "${adw_id}", field: "field_name" }

Note: Protected fields (adw_id, issue_number) cannot be deleted.`;
        }
        cmdParts.push("--field", field);
        if (confirm) {
          cmdParts.push("--confirm");
        }
        break;
    }

    try {
      // Execute the ADW spec CLI command using Bun's process API
      const result = Bun.spawnSync({
        cmd: cmdParts,
        stdout: "pipe",
        stderr: "pipe",
      });
      const decoder = new TextDecoder();
      const stdout = result.stdout ? decoder.decode(result.stdout) : "";
      const stderr = result.stderr ? decoder.decode(result.stderr) : "";

      if (result.exitCode !== 0) {
        const errorOutput = stderr || stdout || `Exit code ${result.exitCode}`;
        return `ADW Spec Command Failed (exit ${result.exitCode}):\n${errorOutput}`;
      }

      return `ADW Spec Command: ${command}\n\n${stdout}`;
    } catch (error: any) {
      const errorOutput = error?.stdout ? error.stdout.toString() : "";
      const errorMsg = error?.stderr ? error.stderr.toString() : error?.message;
      const fallback = errorMsg || errorOutput || "Unknown execution error";
      return `ADW Spec Execution Error:\n${fallback}`;
    }
  },
});
