import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/diagnostics.ts ---
const CONTROL_CHARS_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WHITESPACE_COLLAPSE_PATTERN = /\s+/g;

const REDACTION_PATTERNS: RegExp[] = [
  /\bgh[pousr]_[A-Za-z0-9]{10,}\b/g,
  /\b(?:token|api[_-]?key|secret|password)\s*[:=]\s*([^\s"']+)/gi,
  /\bAuthorization\s*:\s*Bearer\s+([^\s"']+)/gi,
];

const REDACTION_MARKER = "[REDACTED]";

function redactSensitiveFragments(value: string): string {
  let redacted = value;
  redacted = redacted.replace(REDACTION_PATTERNS[0], REDACTION_MARKER);
  redacted = redacted.replace(REDACTION_PATTERNS[1], (match) => {
    const splitIndex = match.indexOf(":") >= 0 ? match.indexOf(":") : match.indexOf("=");
    if (splitIndex < 0) return REDACTION_MARKER;
    return `${match.slice(0, splitIndex + 1)} ${REDACTION_MARKER}`;
  });
  redacted = redacted.replace(REDACTION_PATTERNS[2], `Authorization: Bearer ${REDACTION_MARKER}`);
  return redacted;
}

function decodeUnknown(value: unknown): string {
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  return String(value);
}

function sanitizeDiagnosticText(value: string, maxChars = 4000): string {
  if (!value) return "";
  const noControlChars = value.replace(CONTROL_CHARS_PATTERN, " ");
  const redacted = redactSensitiveFragments(noControlChars);
  const normalized = redacted.replace(WHITESPACE_COLLAPSE_PATTERN, " ").trim();
  if (!normalized) return "";
  if (normalized.length <= maxChars) return normalized;
  return `${normalized.slice(0, maxChars)} ...(truncated)`;
}
// --- End inlined diagnostics ---

const ERROR_SNIPPET_LIMIT = 2000;

type PlatformPrWriteCommand = "create-pr";

function truncate(value: string | undefined): string {
  if (!value) return "";
  return value.length > ERROR_SNIPPET_LIMIT
    ? `${value.slice(0, ERROR_SNIPPET_LIMIT)}...<truncated>`
    : value;
}

function sanitizeAndTruncate(value: unknown): string {
  const sanitized = sanitizeDiagnosticText(decodeUnknown(value), ERROR_SNIPPET_LIMIT);
  return truncate(sanitized);
}

function isCanonicalAdwId(value: string): boolean {
  return /^[a-f0-9]{8}$/.test(value);
}

function buildMissingArgMessage(message: string): string {
  return `ERROR: ${message}`;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const token = String(value).trim();
  return token.length > 0 ? token : undefined;
}

async function runCommand(cmdParts: (string | number)[]): Promise<string> {
  return Bun.$`${cmdParts}`.text();
}

export default tool({
  description:
    "Execute adw platform create-pr with marker compatibility and deterministic validation.",
  args: {
    command: tool.schema.enum(["create-pr"]).describe("Command to execute."),
    title: tool.schema.string().optional(),
    body: tool.schema.string().optional(),
    head: tool.schema.string().optional(),
    base: tool.schema.string().optional(),
    adw_id: tool.schema.string().optional(),
    draft: tool.schema.boolean().optional(),
    prefer_scope: tool.schema.enum(["fork", "upstream"]).optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const command = args.command as PlatformPrWriteCommand;
    const title = normalizeOptionalString(args.title);
    const body = normalizeOptionalString(args.body);
    const head = normalizeOptionalString(args.head);
    const base = normalizeOptionalString(args.base);
    const adwId = normalizeOptionalString(args.adw_id);
    const preferScope = normalizeOptionalString(args.prefer_scope);

    if (command !== "create-pr") {
      return buildMissingArgMessage(`Unsupported command: ${String(command)}`);
    }

    const cmdParts: (string | number)[] = ["uv", "run", "adw", "platform", command];

    if (args.help) {
      cmdParts.push("--help");
      try {
        return await runCommand(cmdParts);
      } catch (error: any) {
        const stdout = sanitizeAndTruncate(error.stdout);
        const stderr = sanitizeAndTruncate(error.stderr);
        return `ERROR: Failed to fetch help for '${command}'.${
          stdout ? `\nSTDOUT:\n${stdout}` : ""
        }${stderr ? `\nSTDERR:\n${stderr}` : ""}`;
      }
    }

    if (!title) {
      return buildMissingArgMessage("'title' is required for command 'create-pr'");
    }
    if (!head) {
      return buildMissingArgMessage("'head' is required for command 'create-pr'");
    }
    if (adwId && !isCanonicalAdwId(adwId)) {
      return buildMissingArgMessage("'adw_id' must be an 8-character lowercase hex string");
    }

    if (args.prefer_scope !== undefined && preferScope === undefined) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (preferScope && !["fork", "upstream"].includes(preferScope)) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }

    cmdParts.push("--title", title, "--head", head);
    if (base) {
      cmdParts.push("--base", base);
    }
    if (adwId) {
      cmdParts.push("--adw-id", adwId);
    }
    if (body) {
      cmdParts.push("--body", body);
    }
    if (args.draft === true) {
      cmdParts.push("--draft");
    }
    if (preferScope) {
      cmdParts.push("--prefer-scope", preferScope);
    }

    try {
      const result = await runCommand(cmdParts);
      if (result.includes("PLATFORM_PR_CREATED") || result.includes("PLATFORM_PR_FAILED")) {
        return result;
      }

      return `PLATFORM_PR_CREATED\n\n${result}\n\n---\nSTATUS: SUCCESS`;
    } catch (error: any) {
      const stdout = sanitizeAndTruncate(error.stdout);
      const stderr = sanitizeAndTruncate(error.stderr);
      const parts: string[] = [
        "PLATFORM_PR_FAILED",
        "",
        `ERROR: Failed to create pull request via 'adw platform ${command}'`,
      ];
      if (stderr) {
        parts.push(`STDERR:\n${stderr}`);
      }
      if (stdout) {
        parts.push(`STDOUT:\n${stdout}`);
      }
      parts.push("", "---", "STATUS: FAILED");
      return parts.join("\n");
    }
  },
});
