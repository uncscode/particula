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

type PlatformPrReadCommand = "pr-comments" | "pr-diff";
type OutputFormat = "text" | "json";

function getStructuredJsonPayload(value: unknown): string | undefined {
  const decoded = decodeUnknown(value).trim();
  if (!decoded) return undefined;
  try {
    JSON.parse(decoded);
    return decoded;
  } catch {
    return undefined;
  }
}

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

function buildMissingArgMessage(message: string): string {
  return `ERROR: ${message}`;
}

function normalizeIssueNumberToken(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const token = String(value).trim();
  return token.length > 0 ? token : undefined;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const token = String(value).trim();
  return token.length > 0 ? token : undefined;
}

function isSafeIssueNumberToken(token: string): boolean {
  return /^[0-9]+$/.test(token);
}

function isStrictPositiveIssueNumberToken(token: string): boolean {
  return isSafeIssueNumberToken(token) && !/^0+$/.test(token);
}

async function runCommand(cmdParts: (string | number)[]): Promise<string> {
  return Bun.$`${cmdParts}`.text();
}

export default tool({
  description: "Execute read-only adw platform PR read commands with strict validation.",
  args: {
    command: tool.schema.enum(["pr-comments", "pr-diff"]).describe("Command to execute."),
    issue_number: tool.schema.string().optional(),
    output_format: tool.schema.enum(["text", "json"]).optional(),
    actionable_only: tool.schema.boolean().optional(),
    prefer_scope: tool.schema.enum(["fork", "upstream"]).optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const command = args.command as PlatformPrReadCommand;
    const issueNumberToken = normalizeIssueNumberToken(args.issue_number);
    const outputFormat: OutputFormat = (args.output_format as OutputFormat) || "text";
    const preferScope = normalizeOptionalString(args.prefer_scope);

    const cmdParts: (string | number)[] = ["uv", "run", "adw", "platform", command];

    if (args.help) {
      cmdParts.push("--help");
      try {
        return await runCommand(cmdParts);
      } catch (error: any) {
        const stderr = sanitizeAndTruncate(error.stderr);
        const stdout = sanitizeAndTruncate(error.stdout);
        return `ERROR: Failed to fetch help for '${command}'.${
          stderr ? `\nSTDERR:\n${stderr}` : ""
        }${stdout ? `\nSTDOUT:\n${stdout}` : ""}`;
      }
    }

    if (!issueNumberToken) {
      return buildMissingArgMessage(`'issue_number' is required for command '${command}'`);
    }
    if (!isStrictPositiveIssueNumberToken(issueNumberToken)) {
      return buildMissingArgMessage(
        "'issue_number' must be a positive integer token (digits only; leading zeros allowed)",
      );
    }
    if (args.prefer_scope !== undefined && preferScope === undefined) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (preferScope && !["fork", "upstream"].includes(preferScope)) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (command === "pr-diff" && args.actionable_only !== undefined) {
      return buildMissingArgMessage("'actionable_only' is only supported for command 'pr-comments'");
    }

    cmdParts.push(issueNumberToken);
    if (args.output_format) {
      cmdParts.push("--format", outputFormat);
    }
    if (command === "pr-comments" && args.actionable_only === true) {
      cmdParts.push("--actionable-only");
    }
    if (preferScope) {
      cmdParts.push("--prefer-scope", preferScope);
    }

    try {
      return await runCommand(cmdParts);
    } catch (error: any) {
      if (command === "pr-diff" && outputFormat === "json") {
        const structuredJson = getStructuredJsonPayload(error.stdout);
        if (structuredJson) {
          return structuredJson;
        }
      }
      const stdout = sanitizeAndTruncate(error.stdout);
      const stderr = sanitizeAndTruncate(error.stderr);
      const parts: string[] = [`ERROR: Failed to execute 'adw platform ${command}'`];
      if (stderr) {
        parts.push(`STDERR:\n${stderr}`);
      }
      if (stdout) {
        parts.push(`STDOUT:\n${stdout}`);
      }
      return parts.join("\n\n");
    }
  },
});
