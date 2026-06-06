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

function isStrictPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && Number.isInteger(value) && value > 0;
}

function isSafeInlinePath(pathToken: string): boolean {
  if (!pathToken || pathToken.startsWith("/") || pathToken.startsWith("\\")) {
    return false;
  }
  if (pathToken.includes("\\")) {
    return false;
  }
  const segments = pathToken.split("/");
  if (segments.some((segment) => segment.length === 0 || segment === "." || segment === "..")) {
    return false;
  }
  return true;
}

function isShaLikeToken(value: string): boolean {
  return /^[0-9a-fA-F]{7,64}$/.test(value);
}

async function runCommand(cmdParts: (string | number)[]): Promise<string> {
  return Bun.$`${cmdParts}`.text();
}

export default tool({
  description: "Execute adw platform pr-review with strict validation.",
  args: {
    command: tool.schema.enum(["pr-review"]).describe("Command to execute."),
    issue_number: tool.schema.string().optional(),
    body: tool.schema.string().optional(),
    path: tool.schema.string().optional(),
    line: tool.schema.number().optional(),
    position: tool.schema.number().optional(),
    commit_sha: tool.schema.string().optional(),
    prefer_scope: tool.schema.enum(["fork", "upstream"]).optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const command = "pr-review";
    const issueNumberToken = normalizeIssueNumberToken(args.issue_number);
    const body = normalizeOptionalString(args.body);
    const path = normalizeOptionalString(args.path);
    const commitSha = normalizeOptionalString(args.commit_sha);
    const preferScope = normalizeOptionalString(args.prefer_scope);
    const cmdParts: (string | number)[] = ["uv", "run", "adw", "platform", command];

    if (args.help) {
      cmdParts.push("--help");
      try {
        return await runCommand(cmdParts);
      } catch (error: any) {
        const stdout = sanitizeAndTruncate(error.stdout);
        const stderr = sanitizeAndTruncate(error.stderr);
        const parts: string[] = [`ERROR: Failed to fetch help for '${command}'.`];
        if (stderr) {
          parts.push(`STDERR:\n${stderr}`);
        }
        if (stdout) {
          parts.push(`STDOUT:\n${stdout}`);
        }
        return parts.join("\n\n");
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
    if (!body) {
      return buildMissingArgMessage("'body' is required for command 'pr-review'");
    }
    if (args.prefer_scope !== undefined && preferScope === undefined) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (preferScope && !["fork", "upstream"].includes(preferScope)) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }

    const line = args.line;
    const position = args.position;
    if (line !== undefined && path === undefined) {
      return buildMissingArgMessage("'--line' requires '--path' for command 'pr-review'");
    }
    if (position !== undefined && path === undefined) {
      return buildMissingArgMessage("'--position' requires '--path' for command 'pr-review'");
    }
    if (path && line === undefined && position === undefined) {
      return buildMissingArgMessage("'--path' requires '--line' or '--position' for command 'pr-review'");
    }
    if (line !== undefined && !isStrictPositiveInteger(line)) {
      return buildMissingArgMessage("'line' must be a positive integer for command 'pr-review'");
    }
    if (position !== undefined && !isStrictPositiveInteger(position)) {
      return buildMissingArgMessage("'position' must be a positive integer for command 'pr-review'");
    }
    if (path && !isSafeInlinePath(path)) {
      return buildMissingArgMessage(
        "'path' must be a safe repository-relative path without traversal for command 'pr-review'",
      );
    }
    if (commitSha && !isShaLikeToken(commitSha)) {
      return buildMissingArgMessage("'commit_sha' must be a SHA-like hex token (7-64 chars)");
    }

    cmdParts.push(issueNumberToken, "--body", body);
    if (path) {
      cmdParts.push("--path", path);
    }
    if (line !== undefined) {
      cmdParts.push("--line", line);
    }
    if (commitSha) {
      cmdParts.push("--commit-sha", commitSha);
    }
    if (position !== undefined) {
      cmdParts.push("--position", position);
    }
    if (preferScope) {
      cmdParts.push("--prefer-scope", preferScope);
    }

    try {
      return await runCommand(cmdParts);
    } catch (error: any) {
      const stdout = sanitizeAndTruncate(error.stdout);
      const stderr = sanitizeAndTruncate(error.stderr);
      const parts: string[] = ["ERROR: Failed to execute 'adw platform pr-review'"];
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
