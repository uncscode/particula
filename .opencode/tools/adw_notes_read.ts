import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/adw_notes_shared (+ transitive deps) ---

const SHOW_PARSE_SNIPPET_LIMIT = 160;
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

const COMMANDS = ["show"] as const;
type ReadCommand = (typeof COMMANDS)[number];

const USAGE = `Usage:
  { command: "show", ref: "HEAD" }`;

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE}`;
}

function normalizeCommand(value: unknown): ReadCommand | null {
  if (typeof value !== "string") {
    return null;
  }
  return (COMMANDS as readonly string[]).includes(value) ? (value as ReadCommand) : null;
}

export default tool({
  description: `Read ADW workflow context notes via \`adw notes show\`.

AVAILABLE COMMANDS:
• show: Read and parse note JSON from a git ref

Examples:
  adw_notes_read({ command: "show", ref: "HEAD" })`,

  args: {
    command: tool.schema.enum([...COMMANDS]).describe("Read command to execute: show."),
    ref: tool.schema.string().describe("Git ref to target (e.g., HEAD, branch name, commit SHA)."),
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

    const result = runNotesCommand(command, ["uv", "run", "adw", "notes", "show", "--ref", ref]);
    if (!result.ok) {
      return result.error;
    }

    try {
      const parsed = JSON.parse(result.stdout);
      return JSON.stringify(parsed, null, 2);
    } catch {
      const snippet = sanitizeSnippet(result.stdout, SHOW_PARSE_SNIPPET_LIMIT);
      const safeSnippet = snippet.length > 0 ? snippet : "<empty stdout>";
      return `ERROR: Failed to parse JSON output from adw notes show. Snippet: ${safeSnippet}`;
    }
  },
});
