/**
 * ADW Issue Batch Spec Tool for OpenCode Integration.
 *
 * Provides structured access to `adw spec batch` CLI commands with validation
 * and consistent error messaging.
 */

import { tool } from "@opencode-ai/plugin";

const ISSUE_SECTIONS = [
  "description",
  "context",
  "scope",
  "acceptance_criteria",
  "technical_notes",
  "testing_strategy",
  "edge_cases",
  "example_usage",
  "references",
];

const COMMANDS = [
  "batch-init",
  "batch-read",
  "batch-write",
  "batch-log",
  "batch-summary",
] as const;

const STATUS_VALUES = ["PASS", "REVISED"] as const;
const MAX_TOTAL = 50;

const USAGE_EXAMPLE = `Example usage:
  adw_issues_spec({ command: "batch-init", total: "5", source: "path/to/doc.md" })
  adw_issues_spec({ command: "batch-read", adw_id: "abc12345", section: "scope" })
  adw_issues_spec({
    command: "batch-write",
    adw_id: "abc12345",
    issue: "1",
    section: "testing_strategy",
    content: "## Tests\n- add coverage"
  })
  adw_issues_spec({
    command: "batch-write",
    adw_id: "abc12345",
    issue: "1",
    content: '{"metadata": {"title": "Add feature X", "phase": "P1"}}'
  })`;

const COMMAND_DESCRIPTIONS = `AVAILABLE COMMANDS:
• batch-init: Initialize batch content
  Usage: { command: "batch-init", total: "5", source: "path/to/doc.md", adw_id?: "..." }

• batch-read: Read batch content
  Usage: { command: "batch-read", adw_id: "abc12345", issue?: "1", section?: "scope", raw?: true }

• batch-write: Write batch content
  Usage: { command: "batch-write", adw_id: "abc12345", issue: "1", content: "...", section?: "scope" }
  Metadata merge: { command: "batch-write", adw_id: "abc12345", issue: "1", content: '{"metadata": {"title": "...", "phase": "P1"}}' }

• batch-log: Append or read review log
  Usage (write): { command: "batch-log", adw_id: "abc12345", issue: "1", reviewer: "testing", status: "PASS" }
  Usage (read): { command: "batch-log", adw_id: "abc12345", issue: "1", read: true }

• batch-summary: Summary table
  Usage: { command: "batch-summary", adw_id: "abc12345" }`;

type BatchCommand = (typeof COMMANDS)[number];

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE_EXAMPLE}`;
}

function normalizePositiveInt(value: string | number): string | null {
  const normalized = String(value ?? "").trim();
  if (!/^[1-9]\d*$/.test(normalized)) {
    return null;
  }
  return normalized;
}

function normalizeIssue(value: string | number): string | null {
  return normalizePositiveInt(value);
}

function normalizeTotal(value: string | number): string | null {
  const normalized = normalizePositiveInt(value);
  if (!normalized) {
    return null;
  }
  if (Number(normalized) > MAX_TOTAL) {
    return null;
  }
  return normalized;
}

function isCommand(command: string): command is BatchCommand {
  return COMMANDS.includes(command as BatchCommand);
}

function buildCliError(output: string): string {
  return `ERROR: adw spec batch command failed.\n${output}\n\n${USAGE_EXAMPLE}`;
}

export default tool({
  description: `Manage ADW issue batch specification content via \`adw spec batch\`.

This tool wraps the batch issue commands with validation to provide agents
safe, typed access to the issue batch content stored in \`adw_state.json\`.

${COMMAND_DESCRIPTIONS}

NOTES:
• \`content\` for batch-write accepts multiline markdown and is passed as a single argument.
• batch-write without --section accepts JSON with \`metadata\` only, \`sections\` only, or both:
  - Both: full replace of the issue entry (backward-compatible).
  - metadata only: merges into existing metadata (use this to set title/phase).
  - sections only: merges into existing sections.
• Errors return with an ERROR: prefix and include a usage example.

${USAGE_EXAMPLE}`,

  args: {
    command: tool.schema
      .enum([...COMMANDS])
      .describe(`Batch command to execute.

${COMMAND_DESCRIPTIONS}

REQUIRED PARAMETERS BY COMMAND:
• batch-init: total, source; optional: adw_id
• batch-read: adw_id; optional: issue, section, raw
• batch-write: adw_id, issue, content; optional: section
• batch-log: adw_id, issue; optional: reviewer, status, note, read
• batch-summary: adw_id`),

    adw_id: tool.schema
      .string()
      .optional()
      .describe(`ADW workflow ID (required for all commands except batch-init).

EXAMPLE: adw_id: "abc12345"`),

    issue: tool.schema
      .string()
      .optional()
      .describe(`Issue number as a positive integer string.

REQUIRED FOR: batch-write, batch-log
OPTIONAL FOR: batch-read

EXAMPLE: issue: "1"`),

    section: tool.schema
      .enum([...ISSUE_SECTIONS])
      .optional()
      .describe(`Issue section to target.

VALID SECTIONS: ${ISSUE_SECTIONS.join(", ")}
OPTIONAL FOR: batch-read, batch-write`),

    content: tool.schema
      .string()
      .optional()
      .describe(`Content payload for batch-write.

REQUIRED FOR: batch-write
Supports multi-line markdown content.`),

    total: tool.schema
      .string()
      .optional()
      .describe(`Total issue count for batch-init (1-${MAX_TOTAL}).

REQUIRED FOR: batch-init
EXAMPLE: total: "5"`),

    source: tool.schema
      .string()
      .optional()
      .describe(`Source document for batch-init.

REQUIRED FOR: batch-init
EXAMPLE: source: "adw-docs/dev-plans/features/F27.md"`),

    reviewer: tool.schema
      .string()
      .optional()
      .describe(`Reviewer name for batch-log write operations.

REQUIRED FOR: batch-log (when read is false)`),

    status: tool.schema
      .enum([...STATUS_VALUES])
      .optional()
      .describe(`Review status for batch-log write operations.

VALID VALUES: PASS, REVISED
REQUIRED FOR: batch-log (when read is false)`),

    note: tool.schema
      .string()
      .optional()
      .describe(`Optional review note for batch-log write operations.`),

    raw: tool.schema
      .boolean()
      .optional()
      .describe(`Return raw JSON or section output.

APPLIES TO: batch-read only`),

    read: tool.schema
      .boolean()
      .optional()
      .describe(`Read review log instead of writing.

APPLIES TO: batch-log only`),
  },

  async execute(args) {
    const {
      command,
      adw_id,
      issue,
      section,
      content,
      total,
      source,
      reviewer,
      status,
      note,
      raw,
      read,
    } = args as Record<string, any>;

    if (!isCommand(command)) {
      return buildError(
        `Invalid command "${command}". Valid commands: ${COMMANDS.join(", ")}.`,
      );
    }

    if (command !== "batch-init" && !adw_id) {
      return buildError("'adw_id' is required for all commands except batch-init.");
    }

    if (section && !ISSUE_SECTIONS.includes(section)) {
      return buildError(
        `Invalid section "${section}". Valid sections: ${ISSUE_SECTIONS.join(", ")}.`,
      );
    }

    if (status && !STATUS_VALUES.includes(status)) {
      return buildError(`Invalid status "${status}". Valid values: PASS, REVISED.`);
    }

    const cmdParts: (string | number)[] = ["uv", "run", "adw", "spec", "batch"];

    switch (command) {
      case "batch-init": {
        if (!total || !source) {
          return buildError("batch-init requires 'total' and 'source'.");
        }
        const normalizedTotal = normalizeTotal(total);
        if (!normalizedTotal) {
          return buildError(`Invalid total "${total}". Must be between 1 and ${MAX_TOTAL}.`);
        }
        cmdParts.push("init", "--total", normalizedTotal, "--source", source);
        if (adw_id) {
          cmdParts.push("--adw-id", adw_id);
        }
        break;
      }

      case "batch-read": {
        cmdParts.push("read", "--adw-id", adw_id as string);
        if (issue) {
          const normalizedIssue = normalizeIssue(issue);
          if (!normalizedIssue) {
            return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
          }
          cmdParts.push("--issue", normalizedIssue);
        }
        if (section) {
          cmdParts.push("--section", section);
        }
        if (raw) {
          cmdParts.push("--raw");
        }
        break;
      }

      case "batch-write": {
        if (!issue) {
          return buildError("batch-write requires 'issue'.");
        }
        if (!content) {
          return buildError("batch-write requires 'content'.");
        }
        const normalizedIssue = normalizeIssue(issue);
        if (!normalizedIssue) {
          return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
        }
        cmdParts.push(
          "write",
          "--adw-id",
          adw_id as string,
          "--issue",
          normalizedIssue,
          "--content",
          content,
        );
        if (section) {
          cmdParts.push("--section", section);
        }
        break;
      }

      case "batch-log": {
        if (!issue) {
          return buildError("batch-log requires 'issue'.");
        }
        const normalizedIssue = normalizeIssue(issue);
        if (!normalizedIssue) {
          return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
        }
        cmdParts.push("log", "--adw-id", adw_id as string, "--issue", normalizedIssue);
        if (read) {
          cmdParts.push("--read");
          break;
        }
        if (!reviewer) {
          return buildError("batch-log requires 'reviewer' when read is false.");
        }
        if (!status) {
          return buildError("batch-log requires 'status' when read is false.");
        }
        cmdParts.push("--reviewer", reviewer, "--status", status);
        if (note) {
          cmdParts.push("--note", note);
        }
        break;
      }

      case "batch-summary": {
        cmdParts.push("summary", "--adw-id", adw_id as string);
        break;
      }
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      if (result.includes("ERROR:") || result.includes("Error:")) {
        return buildCliError(result);
      }
      return result || "adw spec batch completed but returned no output.";
    } catch (error: any) {
      const errorOutput = error?.stdout ? error.stdout.toString() : "";
      const errorMsg = error?.stderr ? error.stderr.toString() : error?.message;

      if (errorOutput && (errorOutput.includes("ERROR") || errorOutput.includes("Error"))) {
        return buildCliError(errorOutput);
      }

      if (errorMsg || errorOutput) {
        return `ERROR: Failed to run adw spec batch command.\n${errorMsg || ""}${
          errorOutput ? `\n\nOutput:\n${errorOutput}` : ""
        }\n\n${USAGE_EXAMPLE}`;
      }

      return buildCliError(String(error));
    }
  },
});
