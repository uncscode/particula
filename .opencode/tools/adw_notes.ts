/**
 * ADW Notes Tool for OpenCode Integration.
 *
 * Wraps `adw notes` CLI commands for agent-facing workflow note operations.
 */

import { tool } from "@opencode-ai/plugin";

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

const COMMANDS = ["write", "write-from-state", "show"] as const;

const USAGE = `Usage:
  { command: "write", ref: "HEAD", fields?: [["key", "value"], ...] }
  { command: "write-from-state", ref: "HEAD", adw_id: "abc12345", fields?: [["key", "value"], ...] }
  { command: "show", ref: "HEAD" }

Notes:
  - fields may be provided as an ordered array of [key, value] pairs (duplicates preserved),
    an array of { key, value } objects, a plain object, or a JSON string containing one of those forms.
  - write-from-state requires adw_id and forwards only --adw-id to CLI state resolution.`;

const SHOW_PARSE_SNIPPET_LIMIT = 160;
const ERROR_OUTPUT_SNIPPET_LIMIT = 400;
const SPAWN_TIMEOUT_MS = 30_000;

type NoteCommand = (typeof COMMANDS)[number];

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE}`;
}

function normalizeCommand(value: unknown): NoteCommand | null {
  if (typeof value !== "string") {
    return null;
  }
  return (COMMANDS as readonly string[]).includes(value) ? (value as NoteCommand) : null;
}

function normalizeRef(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function parseFieldEntries(fields: unknown): [string, string][] | null {
  if (fields === undefined || fields === null) {
    return [];
  }

  let parsed: unknown = fields;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) {
      return [];
    }
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      return null;
    }
  }

  const normalized: [string, string][] = [];

  if (Array.isArray(parsed)) {
    for (const entry of parsed) {
      if (
        Array.isArray(entry) &&
        entry.length === 2 &&
        typeof entry[0] === "string" &&
        typeof entry[1] === "string"
      ) {
        normalized.push([entry[0], entry[1]]);
        continue;
      }

      if (
        entry &&
        typeof entry === "object" &&
        typeof (entry as Record<string, unknown>).key === "string" &&
        typeof (entry as Record<string, unknown>).value === "string"
      ) {
        normalized.push([
          (entry as Record<string, string>).key,
          (entry as Record<string, string>).value,
        ]);
        continue;
      }

      return null;
    }

    return normalized;
  }

  if (parsed && typeof parsed === "object") {
    for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof value !== "string") {
        return null;
      }
      normalized.push([key, value]);
    }
    return normalized;
  }

  return null;
}

function sanitizeSnippet(value: string, maxLen = ERROR_OUTPUT_SNIPPET_LIMIT): string {
  const collapsed = value
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (collapsed.length === 0) {
    return "";
  }
  return collapsed.length > maxLen ? `${collapsed.slice(0, maxLen)}...(truncated)` : collapsed;
}

function buildFieldArgs(fields: [string, string][]): string[] {
  return fields.flatMap(([key, value]) => ["--field", key, value]);
}

export default tool({
  description: `Write and show ADW workflow context notes via \`adw notes\`.

AVAILABLE COMMANDS:
• write: Write explicit note fields to a git ref
• write-from-state: Write note fields from ADW workflow state for an adw_id
• show: Read and parse note JSON from a git ref

Examples:
  adw_notes({ command: "write", ref: "HEAD", fields: [["plan_summary", "done"]] })
  adw_notes({ command: "write-from-state", ref: "HEAD", adw_id: "abc12345" })
  adw_notes({ command: "show", ref: "HEAD" })`,

  args: {
    command: tool.schema
      .enum([...COMMANDS])
      .describe("Notes command to execute: write, write-from-state, or show."),
    ref: tool.schema.string().describe("Git ref to target (e.g., HEAD, branch name, commit SHA)."),
    adw_id: tool.schema
      .string()
      .optional()
      .describe("ADW workflow ID required only for write-from-state."),
    fields: tool.schema
      .any()
      .optional()
      .describe(
        "Optional fields payload for write/write-from-state. Accepts ordered [key,value] entries, {key,value} objects, plain object, or JSON string of those forms.",
      ),
  },

  async execute(args) {
    const command = normalizeCommand(args.command);
    if (!command) {
      return buildError(`Invalid command \"${String(args.command)}\". Valid commands: ${COMMANDS.join(
        ", ",
      )}.`);
    }

    const ref = normalizeRef(args.ref);
    if (!ref) {
      return buildError(`'ref' is required for '${command}'.`);
    }

    const adwId = normalizeAdwId(args.adw_id);
    if (command === "write-from-state" && !adwId) {
      return buildError(`'adw_id' is required for '${command}'.`);
    }

    let fields: [string, string][] = [];
    if (command !== "show") {
      const parsedFields = parseFieldEntries((args as Record<string, unknown>).fields);
      if (!parsedFields) {
        return buildError(
          "'fields' must be an ordered array of [key, value], array of { key, value }, plain object, or JSON string.",
        );
      }
      fields = parsedFields;
    }

    const fieldArgs = buildFieldArgs(fields);
    const cmdParts = ["uv", "run", "adw", "notes"];
    if (command === "write") {
      cmdParts.push("write", "--ref", ref, ...fieldArgs);
    } else if (command === "write-from-state") {
      cmdParts.push("write-from-state", "--ref", ref, "--adw-id", adwId as string, ...fieldArgs);
    } else {
      cmdParts.push("show", "--ref", ref);
    }

    try {
      const result = Bun.spawnSync({
        cmd: cmdParts,
        stdout: "pipe",
        stderr: "pipe",
        timeout: SPAWN_TIMEOUT_MS,
        env: sanitizedEnv(),
      });

      const decoder = new TextDecoder();
      const stdout = result.stdout ? decoder.decode(result.stdout) : "";
      const stderr = result.stderr ? decoder.decode(result.stderr) : "";

      if (result.exitCode !== 0) {
        const prioritized = sanitizeSnippet(stderr) || sanitizeSnippet(stdout);
        const fallback = prioritized || `Exit code ${result.exitCode}`;
        return `ERROR: adw notes ${command} failed.\n${fallback}`;
      }

      if (command === "show") {
        try {
          const parsed = JSON.parse(stdout);
          return JSON.stringify(parsed, null, 2);
        } catch {
          const snippet = sanitizeSnippet(stdout, SHOW_PARSE_SNIPPET_LIMIT);
          const safeSnippet = snippet.length > 0 ? snippet : "<empty stdout>";
          return `ERROR: Failed to parse JSON output from adw notes show. Snippet: ${safeSnippet}`;
        }
      }

      return `ADW Notes Command: ${command}\n\n${stdout}`;
    } catch (error: any) {
      const message = sanitizeSnippet(
        error?.message ? String(error.message) : "Unknown execution error",
      );
      return `ERROR: Failed to execute adw notes ${command}. ${message}`;
    }
  },
});
