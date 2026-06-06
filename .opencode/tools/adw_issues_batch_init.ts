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

const MAX_TOTAL = 50;
const MAX_DIAGNOSTIC_SNIPPET = 400;

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

function normalizePositiveInt(value: string | number): string | null {
  const normalized = String(value ?? "").trim();
  if (!/^[1-9]\d*$/.test(normalized)) return null;
  return normalized;
}

function normalizeTotal(value: string | number): string | null {
  const normalized = normalizePositiveInt(value);
  if (!normalized) return null;
  if (Number(normalized) > MAX_TOTAL) return null;
  return normalized;
}

function containsControlCharacters(value: string): boolean {
  return /[\x00-\x1F\x7F]/.test(value);
}

function normalizeSafeRelativeSourcePath(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (containsControlCharacters(trimmed)) return null;
  if (trimmed.startsWith("/") || trimmed.startsWith("~")) return null;
  const normalized = trimmed.replace(/\\/g, "/");
  const segments = normalized.split("/").filter((segment) => segment.length > 0);
  if (segments.length === 0) return null;
  if (segments.some((segment) => segment === "." || segment === "..")) return null;
  return normalized;
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
      return buildExecutionFailure(command, { stdout, stderr, message: `Exit code ${result.exitCode}` });
    }

    return stdout || "adw spec batch completed but returned no output.";
  } catch (error: any) {
    return buildExecutionFailure(command, error);
  }
}

// --- Tool implementation ---

async function executeBatchInit(args: Record<string, any>): Promise<string> {
  const { total, source, adw_id } = args;
  if (!(typeof total === "string" ? total.trim() : total)) {
    return buildError("batch-init requires 'total' and 'source'.");
  }
  if (!(typeof source === "string" && source.trim().length > 0)) {
    return buildError("batch-init requires 'total' and 'source'.");
  }
  const normalizedSource = normalizeSafeRelativeSourcePath(source);
  if (!normalizedSource) {
    return buildError("Invalid source path. Use a safe, non-empty relative path without control characters or traversal segments.");
  }
  const normalizedTotal = normalizeTotal(total);
  if (!normalizedTotal) {
    return buildError(`Invalid total "${total}". Must be between 1 and ${MAX_TOTAL}.`);
  }
  const normalized = normalizeAndValidateAdwId(adw_id);
  if (!normalized.ok) return normalized.error;
  const cmdParts: (string | number)[] = [
    "uv",
    "run",
    "adw",
    "spec",
    "batch",
    "init",
    "--total",
    normalizedTotal,
    "--source",
    normalizedSource,
  ];
  if (normalized.value) cmdParts.push("--adw-id", normalized.value);
  return runBatchCommand("batch-init", cmdParts);
}

export default tool({
  description: "Initialize batch content via adw spec batch init.",
  args: {
    total: tool.schema.string(),
    source: tool.schema.string(),
    adw_id: tool.schema.string().optional(),
  },
  async execute(args) {
    return executeBatchInit(args as Record<string, any>);
  },
});
