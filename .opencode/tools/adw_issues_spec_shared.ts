import { sanitizeSnippet } from "./adw_spec_shared";

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;

export const COMMANDS = [
  "batch-init",
  "batch-read",
  "batch-write",
  "batch-log",
  "batch-summary",
] as const;
export type BatchCommand = (typeof COMMANDS)[number];

export const STATUS_VALUES = ["PASS", "REVISED"] as const;
export const MAX_TOTAL = 50;

export type ParsedBatchOptions = {
  raw?: true;
  read?: true;
};

const BATCH_OPTION_RULES: Record<BatchCommand, readonly string[]> = {
  "batch-init": [],
  "batch-read": ["raw"],
  "batch-write": [],
  "batch-log": ["read"],
  "batch-summary": [],
};

const BOOLEAN_OPTION_TOKENS = new Set(["raw", "read"]);

export const USAGE_EXAMPLE = `Example usage:
  adw_issues_spec({ command: "batch-init", total: "5", source: "path/to/doc.md" })
  adw_issues_spec({ command: "batch-read", adw_id: "abc12345", section: "scope", options: "raw" })
  adw_issues_spec({
    command: "batch-write",
    adw_id: "abc12345",
    issue: "1",
    section: "testing_strategy",
    content: "## Tests\n- add coverage"
  })
  adw_issues_spec({
    command: "batch-log",
    adw_id: "abc12345",
    issue: "1",
    options: "read"
  })`;

export const COMMAND_DESCRIPTIONS = `AVAILABLE COMMANDS:
• batch-init: Initialize batch content
  Usage: { command: "batch-init", total: "5", source: "path/to/doc.md", adw_id?: "..." }

• batch-read: Read batch content
  Usage: { command: "batch-read", adw_id: "abc12345", issue?: "1", section?: "scope", options?: "raw" }

• batch-write: Write batch content
  Usage: { command: "batch-write", adw_id: "abc12345", issue: "1", content: "...", section?: "scope" }
  Metadata merge: { command: "batch-write", adw_id: "abc12345", issue: "1", content: '{"metadata": {"title": "...", "phase": "P1"}}' }

• batch-log: Append or read review log
  Usage (write): { command: "batch-log", adw_id: "abc12345", issue: "1", reviewer: "testing", status: "PASS" }
  Usage (read): { command: "batch-log", adw_id: "abc12345", issue: "1", options: "read" }

• batch-summary: Summary table
  Usage: { command: "batch-summary", adw_id: "abc12345" }`;

export function isCommand(command: string): command is BatchCommand {
  return COMMANDS.includes(command as BatchCommand);
}

export function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE_EXAMPLE}`;
}

export function buildCliError(output: string): string {
  return `ERROR: adw spec batch command failed.\n${output}\n\n${USAGE_EXAMPLE}`;
}

function renderOptionToken(token: string): string {
  const sanitized = sanitizeSnippet(token, 120).text;
  return JSON.stringify(sanitized || "<empty>");
}

export function isCliFailureOutput(output: string): boolean {
  const normalizedLines = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (normalizedLines.length === 0) {
    return false;
  }

  return normalizedLines.some(
    (line) => line.startsWith("ERROR:") || line.startsWith("Error:"),
  );
}

export function normalizeAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

export function adwIdValidationMessage(): string {
  return "'adw_id' must be an 8-character hex string (e.g., \"abc12345\").";
}

export function normalizeAndValidateAdwId(
  adwId: unknown,
): { ok: true; value: string | null } | { ok: false; error: string } {
  if (!adwId) return { ok: true, value: null };
  if (typeof adwId === "string" && adwId.trim() === "") {
    return { ok: true, value: null };
  }
  const normalized = normalizeAdwId(String(adwId));
  if (!normalized) return { ok: false, error: buildError(adwIdValidationMessage()) };
  return { ok: true, value: normalized };
}

export function isProvidedValue(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
}

export function isInertOptionalValue(value: unknown): boolean {
  if (value === undefined || value === null) return true;
  if (typeof value === "string" && value.trim() === "") return true;
  if (value === false) return true;
  if (value === 0) return true;
  return false;
}

export function stripDefaultArgs(
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

export function normalizePositiveInt(value: string | number): string | null {
  const normalized = String(value ?? "").trim();
  if (!/^[1-9]\d*$/.test(normalized)) return null;
  return normalized;
}

export function normalizeIssue(value: string | number): string | null {
  return normalizePositiveInt(value);
}

export function normalizeTotal(value: string | number): string | null {
  const normalized = normalizePositiveInt(value);
  if (!normalized) return null;
  if (Number(normalized) > MAX_TOTAL) return null;
  return normalized;
}

export function containsControlCharacters(value: string): boolean {
  return /[\x00-\x1F\x7F]/.test(value);
}

export function normalizeSectionToken(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (containsControlCharacters(trimmed)) return null;
  return trimmed;
}

export function normalizeSafeRelativeSourcePath(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (containsControlCharacters(trimmed)) return null;
  if (trimmed.startsWith("/") || trimmed.startsWith("~")) return null;
  const normalized = trimmed.replace(/\\/g, "/");
  if (/^[A-Za-z]:($|\/)/.test(normalized)) return null;
  const segments = normalized.split("/").filter((segment) => segment.length > 0);
  if (segments.length === 0) return null;
  if (segments.some((segment) => segment === "." || segment === "..")) return null;
  return normalized;
}

const MAX_REVIEWER_LENGTH = 120;
const MAX_NOTE_LENGTH = 2000;

export function normalizeReviewer(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length > MAX_REVIEWER_LENGTH) return null;
  if (containsControlCharacters(trimmed)) return null;
  return trimmed;
}

export function normalizeReviewNote(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length > MAX_NOTE_LENGTH) return null;
  if (containsControlCharacters(trimmed)) return null;
  return trimmed;
}

export function parseBatchOptions(
  command: BatchCommand,
  options: unknown,
): { ok: true; options: ParsedBatchOptions } | { ok: false; error: string } {
  if (options === undefined || options === null) {
    return { ok: true, options: {} };
  }
  if (typeof options !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  if (containsControlCharacters(options)) {
    return {
      ok: false,
      error: "ERROR: 'options' must not contain control characters.",
    };
  }

  const normalizedOptions = options.trim();
  if (!normalizedOptions) {
    return { ok: true, options: {} };
  }

  const allowedTokens = BATCH_OPTION_RULES[command];
  const parsed: ParsedBatchOptions = {};

  for (const token of normalizedOptions.split(/\s+/)) {
    if (!token) continue;
    const renderedToken = renderOptionToken(token);
    const separatorCount = token.split("=").length - 1;
    if (separatorCount > 1) {
      return {
        ok: false,
        error: `ERROR: Invalid options token ${renderedToken} for '${command}': tokens must contain at most one '=' separator.`,
      };
    }
    if (separatorCount === 1) {
      const [name] = token.split("=", 1);
      if (!allowedTokens.includes(name)) {
        return {
          ok: false,
          error: `ERROR: Invalid options token ${renderedToken} for '${command}': token is not allowed for this command.`,
        };
      }
      return {
        ok: false,
        error: `ERROR: Invalid options token ${renderedToken} for '${command}': token does not accept a value.`,
      };
    }
    if (!allowedTokens.includes(token)) {
      return {
        ok: false,
        error: `ERROR: Invalid options token ${renderedToken} for '${command}': token is not allowed for this command.`,
      };
    }
    if (!BOOLEAN_OPTION_TOKENS.has(token)) {
      return {
        ok: false,
        error: `ERROR: Invalid options token ${renderedToken} for '${command}': token requires a non-empty '=value' suffix.`,
      };
    }
    if (token in parsed) {
      return {
        ok: false,
        error: `ERROR: Invalid options token ${renderedToken} for '${command}': duplicate token.`,
      };
    }
    if (token === "raw") parsed.raw = true;
    if (token === "read") parsed.read = true;
  }

  return { ok: true, options: parsed };
}

export function buildExecutionFailure(command: BatchCommand, error: any): string {
  const stderr = sanitizeSnippet(error?.stderr?.toString?.() ?? error?.stderr ?? "", 400).text;
  const stdout = sanitizeSnippet(error?.stdout?.toString?.() ?? error?.stdout ?? "", 400).text;
  const message = sanitizeSnippet(error?.message ?? "", 400).text;
  const detail = stderr
    ? `stderr:\n${stderr}`
    : stdout
      ? `stdout:\n${stdout}`
      : message
        ? `message:\n${message}`
        : "No stderr/stdout/message available.";
  return `ERROR: Failed to execute 'adw spec batch ${command}'.\n${detail}\n\n${USAGE_EXAMPLE}`;
}

export async function runBatchCommandText(
  command: BatchCommand,
  cmdParts: (string | number)[],
): Promise<string> {
  try {
    const result = await Bun.$`${cmdParts}`.text();
    const output = result || "adw spec batch completed but returned no output.";
    return isCliFailureOutput(output) ? buildCliError(output) : output;
  } catch (error: any) {
    return buildExecutionFailure(command, error);
  }
}

export function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    env[key] = value;
  }
  return env;
}

export function runBatchCommandSpawn(
  command: BatchCommand,
  cmdParts: (string | number)[],
): string {
  try {
    const result = Bun.spawnSync({
      cmd: cmdParts.map(String),
      stdout: "pipe",
      stderr: "pipe",
      env: sanitizedEnv(),
    });
    const decoder = new TextDecoder();
    const stdout = result.stdout ? decoder.decode(result.stdout) : "";
    const stderr = result.stderr ? decoder.decode(result.stderr) : "";

    if (result.exitCode !== 0) {
      return buildExecutionFailure(command, {
        stdout,
        stderr,
        message: `Exit code ${result.exitCode}`,
      });
    }

    const output = stdout || "adw spec batch completed but returned no output.";
    return isCliFailureOutput(output) ? buildCliError(output) : output;
  } catch (error: any) {
    return buildExecutionFailure(command, error);
  }
}
