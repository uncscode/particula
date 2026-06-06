import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/adw_id ---

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

// --- Inlined from lib/adw_issues_spec_shared ---

const COMMANDS = ["batch-init","batch-read","batch-write","batch-log","batch-summary"] as const;
type BatchCommand = (typeof COMMANDS)[number];

const STATUS_VALUES = ["PASS", "REVISED"] as const;
const MAX_DIAGNOSTIC_SNIPPET = 400;
const MAX_REVIEWER_LENGTH = 120;
const MAX_NOTE_LENGTH = 2000;

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

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE_EXAMPLE}`;
}

function isProvidedValue(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
}

function normalizePositiveInt(value: string | number): string | null {
  const normalized = String(value ?? "").trim();
  if (!/^[1-9]\d*$/.test(normalized)) return null;
  return normalized;
}

function normalizeIssue(value: string | number): string | null {
  return normalizePositiveInt(value);
}

function containsControlCharacters(value: string): boolean {
  return /[\x00-\x1F\x7F]/.test(value);
}

function normalizeReviewer(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length > MAX_REVIEWER_LENGTH) return null;
  if (containsControlCharacters(trimmed)) return null;
  return trimmed;
}

function normalizeReviewNote(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length > MAX_NOTE_LENGTH) return null;
  if (containsControlCharacters(trimmed)) return null;
  return trimmed;
}

function sanitizeDiagnostic(value: unknown): string {
  return String(value ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
}

function boundDiagnostic(value: string): string {
  if (value.length <= MAX_DIAGNOSTIC_SNIPPET) return value;
  return `${value.slice(0, MAX_DIAGNOSTIC_SNIPPET)}... [truncated]`;
}

function buildExecutionFailure(command: BatchCommand, error: any): string {
  const stderr = boundDiagnostic(sanitizeDiagnostic(error?.stderr?.toString?.() ?? error?.stderr));
  const stdout = boundDiagnostic(sanitizeDiagnostic(error?.stdout?.toString?.() ?? error?.stdout));
  const message = boundDiagnostic(sanitizeDiagnostic(error?.message));
  const details: string[] = [];
  if (stderr) details.push(`stderr:\n${stderr}`);
  if (stdout) details.push(`stdout:\n${stdout}`);
  if (!stderr && !stdout && message) details.push(`message:\n${message}`);
  const detail = details.join("\n\n") || "No stderr/stdout/message available.";
  return `ERROR: Failed to execute 'adw spec batch ${command}'.\n${detail}\n\n${USAGE_EXAMPLE}`;
}

function normalizeAndValidateAdwId(adw_id: unknown): { ok: true; value: string | null } | { ok: false; error: string } {
  if (!adw_id) return { ok: true, value: null };
  const normalized = normalizeAdwId(String(adw_id));
  if (!normalized) return { ok: false, error: buildError(adwIdValidationMessage()) };
  return { ok: true, value: normalized };
}

async function runBatchCommand(command: BatchCommand, cmdParts: (string | number)[]): Promise<string> {
  try {
    const result = await Bun.$`${cmdParts}`.text();
    return result || "adw spec batch completed but returned no output.";
  } catch (error: any) {
    return buildExecutionFailure(command, error);
  }
}

// --- Tool implementation ---

async function executeBatchLog(args: Record<string, any>): Promise<string> {
  const { adw_id, issue, reviewer, status, note, read } = args;
  if (!isProvidedValue(adw_id)) return buildError("'adw_id' is required for all commands except batch-init.");
  if (!isProvidedValue(issue)) return buildError("batch-log requires 'issue'.");

  const normalizedAdw = normalizeAndValidateAdwId(adw_id);
  if (!normalizedAdw.ok || !normalizedAdw.value) return normalizedAdw.ok ? buildError("'adw_id' is required for all commands except batch-init.") : normalizedAdw.error;
  const normalizedIssue = normalizeIssue(issue);
  if (!normalizedIssue) return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);

  const cmdParts: (string | number)[] = [
    "uv",
    "run",
    "adw",
    "spec",
    "batch",
    "log",
    "--adw-id",
    normalizedAdw.value,
    "--issue",
    normalizedIssue,
  ];
  if (read === true) {
    cmdParts.push("--read");
    return runBatchCommand("batch-log", cmdParts);
  }

  if (!isProvidedValue(reviewer)) return buildError("batch-log requires 'reviewer' when read is false.");
  if (!isProvidedValue(status)) return buildError("batch-log requires 'status' when read is false.");
  if (!STATUS_VALUES.includes(status)) {
    return buildError(`Invalid status "${status}". Valid values: PASS, REVISED.`);
  }
  const normalizedReviewer = normalizeReviewer(reviewer);
  if (!normalizedReviewer) {
    return buildError("Invalid reviewer. Use a non-empty reviewer without control characters.");
  }
  cmdParts.push("--reviewer", normalizedReviewer, "--status", status);

  const normalizedNote = normalizeReviewNote(note);
  if (typeof note === "string" && note.trim().length > 0 && !normalizedNote) {
    return buildError("Invalid note. Use a non-empty note without control characters.");
  }
  if (normalizedNote) cmdParts.push("--note", normalizedNote);
  return runBatchCommand("batch-log", cmdParts);
}

export default tool({
  description: "Read/append review log via adw spec batch log.",
  args: {
    adw_id: tool.schema.string(),
    issue: tool.schema.string(),
    reviewer: tool.schema.string().optional(),
    status: tool.schema.enum([...STATUS_VALUES]).optional(),
    note: tool.schema.string().optional(),
    read: tool.schema.boolean().optional(),
  },
  async execute(args) {
    return executeBatchLog(args as Record<string, any>);
  },
});
