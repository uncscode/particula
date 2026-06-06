import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/adw_notes_shared (+ transitive deps from lib/adw_id) ---

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;

function normalizeAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

const ERROR_OUTPUT_SNIPPET_LIMIT = 400;
const SPAWN_TIMEOUT_MS = 30_000;

function sanitizeSnippet(value: string, maxLen = ERROR_OUTPUT_SNIPPET_LIMIT): string {
  const collapsed = value
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (collapsed.length === 0) return "";
  return collapsed.length > maxLen ? `${collapsed.slice(0, maxLen)}...(truncated)` : collapsed;
}

function normalizeRef(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeCanonicalAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  return normalizeAdwId(value);
}

function parseFieldEntries(fields: unknown): [string, string][] | null {
  if (fields === undefined || fields === null) return [];
  let parsed: unknown = fields;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) return [];
    try { parsed = JSON.parse(trimmed); } catch { return null; }
  }
  const normalized: [string, string][] = [];
  if (Array.isArray(parsed)) {
    for (const entry of parsed) {
      if (entry === null || entry === undefined) continue;
      if (Array.isArray(entry) && entry.length === 2) {
        const rawKey = entry[0]; const rawValue = entry[1];
        if (typeof rawKey !== "string" || typeof rawValue !== "string") return null;
        const key = rawKey.trim();
        if (key.length === 0) continue;
        normalized.push([key, rawValue]);
        continue;
      }
      if (entry && typeof entry === "object") {
        const rawKey = (entry as Record<string, unknown>).key;
        const rawValue = (entry as Record<string, unknown>).value;
        if (rawKey === null || rawKey === undefined || rawValue === null || rawValue === undefined) continue;
        if (typeof rawKey !== "string" || typeof rawValue !== "string") return null;
        const key = rawKey.trim();
        if (key.length === 0) continue;
        normalized.push([key, rawValue]);
        continue;
      }
      return null;
    }
    return normalized;
  }
  if (parsed && typeof parsed === "object") {
    for (const [rawKey, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (value === null || value === undefined) continue;
      if (typeof value !== "string") return null;
      const key = rawKey.trim();
      if (key.length === 0) continue;
      normalized.push([key, value]);
    }
    return normalized;
  }
  return null;
}

function buildFieldArgs(fields: [string, string][]): string[] {
  return fields.flatMap(([key, value]) => ["--field", key, value]);
}

function selectDiagnostic(...candidates: unknown[]): string {
  for (const candidate of candidates) {
    const snippet = sanitizeSnippet(candidate ? String(candidate) : "");
    if (snippet) return snippet;
  }
  return "";
}

function runNotesCommand(command: "write" | "write-from-state" | "show", cmdParts: string[]): {
  ok: true; stdout: string;
} | {
  ok: false; error: string;
} {
  try {
    const result = Bun.spawnSync({
      cmd: cmdParts,
      stdout: "pipe",
      stderr: "pipe",
      timeout: SPAWN_TIMEOUT_MS,
    });
    const decoder = new TextDecoder();
    const stdout = result.stdout ? decoder.decode(result.stdout) : "";
    const stderr = result.stderr ? decoder.decode(result.stderr) : "";
    if (result.exitCode !== 0) {
      const diagnostic = selectDiagnostic(stderr, stdout, (result as any)?.message);
      const fallback = diagnostic || `Exit code ${result.exitCode}`;
      return { ok: false, error: `ERROR: adw notes ${command} failed.\n${fallback}` };
    }
    return { ok: true, stdout };
  } catch (error: any) {
    const diagnostic =
      selectDiagnostic(error?.stderr?.toString?.() ?? error?.stderr, error?.stdout?.toString?.() ?? error?.stdout, error?.message) ||
      "Unknown execution error";
    return { ok: false, error: `ERROR: Failed to execute adw notes ${command}. ${diagnostic}` };
  }
}

// --- End inlined helpers ---

const COMMANDS = ["write", "write-from-state"] as const;
type WriteCommand = (typeof COMMANDS)[number];

const USAGE = `Usage:
  { command: "write", ref: "HEAD", fields?: [["key", "value"], ...] }
  { command: "write-from-state", ref: "HEAD", adw_id: "abc12345", fields?: [["key", "value"], ...] }

Notes:
  - fields may be provided as an ordered array of [key, value] pairs (duplicates preserved),
    an array of { key, value } objects, a plain object, or a JSON string containing one of those forms.
  - write-from-state requires adw_id and forwards only --adw-id to CLI state resolution.`;

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE}`;
}

function normalizeCommand(value: unknown): WriteCommand | null {
  if (typeof value !== "string") {
    return null;
  }
  return (COMMANDS as readonly string[]).includes(value) ? (value as WriteCommand) : null;
}

export default tool({
  description: `Write ADW workflow context notes via \`adw notes\`.

AVAILABLE COMMANDS:
• write: Write explicit note fields to a git ref
• write-from-state: Write note fields from ADW workflow state for an adw_id

Examples:
  adw_notes_write({ command: "write", ref: "HEAD", fields: [["plan_summary", "done"]] })
  adw_notes_write({ command: "write-from-state", ref: "HEAD", adw_id: "abc12345" })`,

  args: {
    command: tool.schema.enum([...COMMANDS]).describe("Write command to execute: write or write-from-state."),
    ref: tool.schema.string().describe("Git ref to target (e.g., HEAD, branch name, commit SHA)."),
    adw_id: tool.schema.string().optional().describe("ADW workflow ID required only for write-from-state."),
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
      return buildError(`Invalid command \"${String(args.command)}\". Valid commands: ${COMMANDS.join(", ")}.`);
    }

    const ref = normalizeRef(args.ref);
    if (!ref) {
      return buildError(`'ref' is required for '${command}'.`);
    }

    const normalizedAdwId = normalizeCanonicalAdwId(args.adw_id);
    if (command === "write-from-state" && !normalizedAdwId) {
      return buildError(`'adw_id' is required for '${command}'.`);
    }

    const parsedFields = parseFieldEntries((args as Record<string, unknown>).fields);
    if (!parsedFields) {
      return buildError(
        "'fields' must be an ordered array of [key, value], array of { key, value }, plain object, or JSON string.",
      );
    }

    const fieldArgs = buildFieldArgs(parsedFields);
    const cmdParts = ["uv", "run", "adw", "notes"];
    if (command === "write") {
      cmdParts.push("write", "--ref", ref, ...fieldArgs);
    } else {
      cmdParts.push("write-from-state", "--ref", ref, "--adw-id", normalizedAdwId as string, ...fieldArgs);
    }

    const result = runNotesCommand(command, cmdParts);
    if (!result.ok) {
      return result.error;
    }

    return `ADW Notes Command: ${command}\n\n${result.stdout}`;
  },
});
