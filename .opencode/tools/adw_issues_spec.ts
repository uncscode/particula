/**
 * ADW Issue Batch Spec Tool for OpenCode Integration.
 *
 * Provides structured access to `adw spec batch` CLI commands with validation
 * and consistent error messaging.
 */

import { tool } from "@opencode-ai/plugin";

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;

function normalizeAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

function adwIdValidationMessage(): string {
  return "'adw_id' must be an 8-character hex string (e.g., \"abc12345\").";
}

const COMMANDS = [
  "batch-init",
  "batch-read",
  "batch-write",
  "batch-log",
  "batch-summary",
] as const;

const STATUS_VALUES = ["PASS", "REVISED"] as const;
const MAX_TOTAL = 50;
const MAX_DIAGNOSTIC_SNIPPET = 400;

/**
 * Strip inert optional values before command routing.
 *
 * OpenCode callers sometimes send optional fields with empty string, null,
 * false, or 0 defaults. Treat those the same as omitted so metadata batch
 * writes can reach the CLI without an accidental --section argument.
 */
function isInertOptionalValue(value: unknown): boolean {
  if (value === undefined || value === null) return true;
  if (typeof value === "string" && value.trim() === "") return true;
  if (value === false) return true;
  if (value === 0) return true;
  return false;
}

function isCliFailureOutput(output: string): boolean {
  const normalizedLines = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (normalizedLines.length === 0) return false;

  return normalizedLines.some(
    (line) => line.startsWith("ERROR:") || line.startsWith("Error:"),
  );
}

function normalizeSectionToken(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (trimmed.length === 0) return null;
  // Preserve runtime-owned section-name contract while rejecting control/newline payloads.
  if (/[\x00-\x1F\x7F]/.test(trimmed)) {
    return null;
  }
  return trimmed;
}

function stripDefaultArgs(
  raw: Record<string, any>,
  optionalKeys: Set<string>,
): Record<string, any> {
  const cleaned: Record<string, any> = { command: raw.command };
  for (const [key, value] of Object.entries(raw)) {
    if (key === "command") continue;
    if (optionalKeys.has(key) && isInertOptionalValue(value)) continue;
    cleaned[key] = value;
  }
  return cleaned;
}

function isProvidedValue(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  return true;
}

function sanitizeDiagnostic(value: unknown): string {
  return String(value ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();
}

function boundDiagnostic(value: string): string {
  if (value.length <= MAX_DIAGNOSTIC_SNIPPET) {
    return value;
  }
  return `${value.slice(0, MAX_DIAGNOSTIC_SNIPPET)}... [truncated]`;
}

function buildExecutionFailure(command: BatchCommand, error: any): string {
  const stderr = boundDiagnostic(sanitizeDiagnostic(error?.stderr?.toString?.() ?? error?.stderr));
  const stdout = boundDiagnostic(sanitizeDiagnostic(error?.stdout?.toString?.() ?? error?.stdout));
  const message = boundDiagnostic(sanitizeDiagnostic(error?.message));
  const details: string[] = [];
  if (stderr) {
    details.push(`stderr:\n${stderr}`);
  }
  if (stdout) {
    details.push(`stdout:\n${stdout}`);
  }
  if (!stderr && !stdout && message) {
    details.push(`message:\n${message}`);
  }
  const detail = details.join("\n\n") || "No stderr/stdout/message available.";

  return `ERROR: Failed to execute 'adw spec batch ${command}'.\n${detail}\n\n${USAGE_EXAMPLE}`;
}

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
      .string()
      .optional()
      .describe(`Issue section to target.

OPTIONAL FOR: batch-read, batch-write. Omit for batch-write metadata JSON merges.
Section validation is owned by the runtime CLI to support template-defined
dynamic section contracts.`),

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
    const rawArgs = args as Record<string, any>;
    const rawCommand = rawArgs.command;
    if (!isCommand(rawCommand)) {
      return buildError(
        `Invalid command "${rawCommand}". Valid commands: ${COMMANDS.join(", ")}.`,
      );
    }

    const requiredByCommand: Record<BatchCommand, string[]> = {
      "batch-init": ["total", "source"],
      "batch-read": ["adw_id"],
      "batch-write": ["adw_id", "issue", "content"],
      "batch-log": ["adw_id", "issue"],
      "batch-summary": ["adw_id"],
    };

    for (const key of requiredByCommand[rawCommand]) {
      if (!isProvidedValue(rawArgs[key])) {
        if (rawCommand === "batch-init") {
          return buildError("batch-init requires 'total' and 'source'.");
        }
        if (rawCommand === "batch-log" && key === "issue") {
          return buildError("batch-log requires 'issue'.");
        }
        if (rawCommand === "batch-write" && key === "issue") {
          return buildError("batch-write requires 'issue'.");
        }
        if (rawCommand === "batch-write" && key === "content") {
          return buildError("batch-write requires 'content'.");
        }
        return buildError("'adw_id' is required for all commands except batch-init.");
      }
    }

    if (rawCommand === "batch-read" && Object.hasOwn(rawArgs, "issue")) {
      const rawIssue = rawArgs.issue;
      const issueProvided =
        rawIssue === 0 || (typeof rawIssue === "string" ? rawIssue.trim().length > 0 : !!rawIssue);
      if (issueProvided && !normalizeIssue(rawIssue)) {
        return buildError(`Invalid issue "${rawIssue}". Issue must be a positive integer.`);
      }
    }

    if (rawCommand === "batch-log" && rawArgs.read !== true) {
      if (!isProvidedValue(rawArgs.reviewer)) {
        return buildError("batch-log requires 'reviewer' when read is false.");
      }
      if (!isProvidedValue(rawArgs.status)) {
        return buildError("batch-log requires 'status' when read is false.");
      }
    }

    const optionalKeys = new Set([
      "adw_id",
      "issue",
      "section",
      "content",
      "total",
      "source",
      "reviewer",
      "status",
      "note",
      "raw",
      "read",
    ]);
    for (const key of requiredByCommand[rawCommand]) {
      optionalKeys.delete(key);
    }
    if (rawCommand === "batch-log" && rawArgs.read !== true) {
      optionalKeys.delete("reviewer");
      optionalKeys.delete("status");
    }

    const normalizedArgs = stripDefaultArgs(rawArgs, optionalKeys);
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
    } = normalizedArgs;

    if (command !== "batch-init" && !adw_id) {
      return buildError("'adw_id' is required for all commands except batch-init.");
    }

    // Normalize adw_id via canonical validator when provided.
    const normalizedAdwId = adw_id ? normalizeAdwId(adw_id) : null;
    if (adw_id && !normalizedAdwId) {
      return buildError(adwIdValidationMessage());
    }

    if (status && !STATUS_VALUES.includes(status)) {
      return buildError(`Invalid status "${status}". Valid values: PASS, REVISED.`);
    }

    const cmdParts: (string | number)[] = ["uv", "run", "adw", "spec", "batch"];

    switch (command) {
      case "batch-init": {
        const normalizedTotal = normalizeTotal(total);
        if (!normalizedTotal) {
          return buildError(`Invalid total "${total}". Must be between 1 and ${MAX_TOTAL}.`);
        }
        cmdParts.push("init", "--total", normalizedTotal, "--source", source);
        if (normalizedAdwId) {
          cmdParts.push("--adw-id", normalizedAdwId);
        }
        break;
      }

      case "batch-read": {
        cmdParts.push("read", "--adw-id", normalizedAdwId!);
        if (issue) {
          const normalizedIssue = normalizeIssue(issue);
          if (!normalizedIssue) {
            return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
          }
          cmdParts.push("--issue", normalizedIssue);
        }
        if (section) {
          const normalizedSection = normalizeSectionToken(section);
          if (!normalizedSection) {
            return buildError(
              `Invalid section token "${section}". Section must not contain control/newline characters.`,
            );
          }
          cmdParts.push("--section", normalizedSection);
        }
        if (raw) {
          cmdParts.push("--raw");
        }
        break;
      }

      case "batch-write": {
        const normalizedIssue = normalizeIssue(issue);
        if (!normalizedIssue) {
          return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
        }
        cmdParts.push(
          "write",
          "--adw-id",
          normalizedAdwId!,
          "--issue",
          normalizedIssue,
          "--content",
          content,
        );
        if (section) {
          const normalizedSection = normalizeSectionToken(section);
          if (!normalizedSection) {
            return buildError(
              `Invalid section token "${section}". Section must not contain control/newline characters.`,
            );
          }
          cmdParts.push("--section", normalizedSection);
        }
        break;
      }

      case "batch-log": {
        const normalizedIssue = normalizeIssue(issue);
        if (!normalizedIssue) {
          return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
        }
        cmdParts.push("log", "--adw-id", normalizedAdwId!, "--issue", normalizedIssue);
        if (read) {
          cmdParts.push("--read");
          break;
        }
        cmdParts.push("--reviewer", reviewer, "--status", status);
        if (note) {
          cmdParts.push("--note", note);
        }
        break;
      }

      case "batch-summary": {
        cmdParts.push("summary", "--adw-id", normalizedAdwId!);
        break;
      }
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      if (isCliFailureOutput(result)) {
        return buildCliError(result);
      }
      return result || "adw spec batch completed but returned no output.";
    } catch (error: any) {
      return buildExecutionFailure(command, error);
    }
  },
});
