import fs from "node:fs";
import path from "node:path";

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;
const ERROR_SNIPPET_LIMIT = 1200;
const CONTROL_CHAR_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WINDOWS_ABSOLUTE_PATH_PATTERN = /[A-Za-z]:\\(?:[^\\\r\n"']+\\)*[^\\\r\n"']+/g;
const QUOTED_UNIX_ABSOLUTE_PATH_PATTERN = /(["'])(\/[^\r\n"']+)\1/g;
const UNIX_COLON_PATH_PATTERN = /(^|[\s(\[])(\/(?:[^\s:\r\n)\]"']|:(?!\s))+?)(?=:\s|:\d|$)/gm;
const UNIX_BARE_PATH_PATTERN = /(^|[\s(\[])(\/[^\s)\]"']+)/gm;
const REDACTED_SECRET = "<redacted-secret>";
const SECRET_ASSIGNMENT_PATTERNS = [
  /\b(authorization\s*:\s*bearer\s+)([^\s]+)/gi,
  /\b((?:token|secret|password|passwd|api(?:_|-)?key|access(?:_|-)?token|refresh(?:_|-)?token)\s*[:=]\s*)("?)([^\s",']+)("?)/gi,
  /\b(gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]+)\b/g,
];

function resolveRepoRoot(start: string): string {
  let current = fs.realpathSync(start);
  while (true) {
    if (fs.existsSync(path.join(current, ".git"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return fs.realpathSync(start);
    }
    current = parent;
  }
}

type DiagnosticSources = {
  stderr?: string;
  stdout?: string;
  message?: string;
};

type SanitizedDiagnostic = {
  text: string;
  hasVisibleContent: boolean;
};

type ParsedSpecOptions = {
  raw?: true;
  json?: true;
  append?: true;
  confirm?: true;
  last?: number;
};

const SPEC_OPTION_RULES = {
  list: ["json"],
  read: ["raw"],
  write: ["append"],
  delete: ["confirm"],
  "messages-write": [],
  "messages-read": ["last", "raw"],
} as const;

const SPEC_BOOLEAN_OPTION_TOKENS = new Set(["raw", "json", "append", "confirm"]);
const SPEC_MAX_LAST = 50;

export type AdwSpecCommandResult =
  | { ok: true; stdout: string }
  | { ok: false; error: string };

type ParsedSpecOptionsResult =
  | { ok: true; options: ParsedSpecOptions }
  | { ok: false; error: string };

function redactAbsolutePaths(raw: string): string {
  return raw
    .replace(WINDOWS_ABSOLUTE_PATH_PATTERN, "<path>")
    .replace(QUOTED_UNIX_ABSOLUTE_PATH_PATTERN, "$1<path>$1")
    .replace(UNIX_COLON_PATH_PATTERN, (_, prefix) => `${prefix}<path>`)
    .replace(UNIX_BARE_PATH_PATTERN, (_, prefix) => `${prefix}<path>`);
}

function redactSecrets(raw: string): string {
  return SECRET_ASSIGNMENT_PATTERNS.reduce((output, pattern) => {
    if (pattern.global && pattern.source.includes("authorization")) {
      return output.replace(pattern, `$1${REDACTED_SECRET}`);
    }
    if (pattern.global && pattern.source.includes("token|secret|password")) {
      return output.replace(pattern, `$1$2${REDACTED_SECRET}$4`);
    }
    return output.replace(pattern, REDACTED_SECRET);
  }, raw);
}

export function sanitizeSnippet(
  value: string,
  limit: number = ERROR_SNIPPET_LIMIT,
): SanitizedDiagnostic {
  if (!value) {
    return { text: "", hasVisibleContent: false };
  }
  const normalized = value.replace(/\r\n?/g, "\n").replace(CONTROL_CHAR_PATTERN, "");
  const redacted = redactSecrets(redactAbsolutePaths(normalized)).replace(
    /<path>(?:\s+<path>)+/g,
    "<path>",
  );
  if (!redacted) {
    return { text: "", hasVisibleContent: false };
  }
  const hasVisibleContent = Boolean(redacted.trim());
  let output = redacted.trim();
  if (output.length > limit) {
    output = `${output.slice(0, limit).trimEnd()}...`;
  }
  return { text: output, hasVisibleContent };
}

export function normalizeOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
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

export function validateAndNormalizeAdwId(
  command: string,
  adwId: unknown,
): { ok: true; adwId: string } | { ok: false; error: string } {
  if (typeof adwId !== "string" || adwId.trim().length === 0) {
    return {
      ok: false,
      error: `ERROR: 'adw_id' parameter is required for all spec commands.\n\nUsage: Use the ADW tool with command "status" to see active workflow IDs.\nExample: { command: "${command}", adw_id: "abc12345" }`,
    };
  }
  const normalized = normalizeAdwId(adwId);
  if (!normalized) {
    return { ok: false, error: `ERROR: ${adwIdValidationMessage()}` };
  }
  return { ok: true, adwId: normalized };
}

export function validateCanonicalInRepoPath(
  inputPath: string,
): { ok: true; canonicalPath: string } | { ok: false; error: string } {
  const trimmed = inputPath.trim();
  if (trimmed.length === 0) {
    return { ok: false, error: "ERROR: '--file' path must be non-empty." };
  }
  const candidate = path.resolve(process.cwd(), trimmed);
  const repoRoot = resolveRepoRoot(process.cwd());
  let canonicalPath: string;
  try {
    canonicalPath = fs.realpathSync(candidate);
  } catch {
    return { ok: false, error: "ERROR: '--file' path does not exist." };
  }
  const rel = path.relative(repoRoot, canonicalPath);
  if (rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel))) {
    return { ok: true, canonicalPath };
  }
  return { ok: false, error: "ERROR: '--file' path resolves outside repository root." };
}

export function parseSpecOptions(command: string, options: unknown): ParsedSpecOptionsResult {
  if (options === undefined || options === null) {
    return { ok: true, options: {} };
  }
  if (typeof options !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalizedOptions = options.trim();
  if (!normalizedOptions) {
    return { ok: true, options: {} };
  }

  const allowedTokens = SPEC_OPTION_RULES[command as keyof typeof SPEC_OPTION_RULES];
  if (!allowedTokens) {
    return { ok: false, error: `ERROR: Unsupported spec command '${command}' for options parsing.` };
  }

  const parsed: ParsedSpecOptions = {};
  for (const token of normalizedOptions.split(/\s+/)) {
    if (!token) continue;
    const separatorCount = token.split("=").length - 1;
    if (separatorCount > 1) {
      return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': tokens must contain at most one '=' separator.` };
    }

    if (separatorCount === 0) {
      if (!allowedTokens.includes(token as never)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': token is not allowed for this command.` };
      }
      if (!SPEC_BOOLEAN_OPTION_TOKENS.has(token)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': token requires a non-empty '=value' suffix.` };
      }
      if (token in parsed) {
        return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': duplicate token.` };
      }
      if (token === "raw") parsed.raw = true;
      if (token === "json") parsed.json = true;
      if (token === "append") parsed.append = true;
      if (token === "confirm") parsed.confirm = true;
      continue;
    }

    const [name, rawValue] = token.split("=", 2);
    if (!allowedTokens.includes(name as never)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': token is not allowed for this command.` };
    }
    if (SPEC_BOOLEAN_OPTION_TOKENS.has(name)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': token does not accept a value.` };
    }
    if (!rawValue) {
      return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': token requires a non-empty '=value' suffix.` };
    }
    if (name === "last") {
      if (!/^\d+$/.test(rawValue)) {
        return { ok: false, error: `ERROR: 'last' must be an integer (0-${SPEC_MAX_LAST}).` };
      }
      const last = Number(rawValue);
      if (last < 0 || last > SPEC_MAX_LAST) {
        return { ok: false, error: `ERROR: 'last' must be between 0 and ${SPEC_MAX_LAST}.` };
      }
      if (parsed.last !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': duplicate token.` };
      }
      parsed.last = last;
      continue;
    }

    return { ok: false, error: `ERROR: Invalid options token '${token}' for '${command}': unsupported token.` };
  }

  return { ok: true, options: parsed };
}

export function selectDiagnostic(sources: DiagnosticSources, fallback: string): string {
  const safeStderr = sanitizeSnippet(sources.stderr ?? "");
  const safeStdout = sanitizeSnippet(sources.stdout ?? "");
  const safeMessage = sanitizeSnippet(sources.message ?? "");
  if (safeStderr.hasVisibleContent) return safeStderr.text;
  if (safeStdout.hasVisibleContent) return safeStdout.text;
  if (safeMessage.hasVisibleContent) return safeMessage.text;
  return fallback;
}

export function runAdwSpecCommand(
  command: string,
  cmd: string[],
  env: Record<string, string>,
): AdwSpecCommandResult {
  try {
    const result = Bun.spawnSync({ cmd, stdout: "pipe", stderr: "pipe", env });
    const decoder = new TextDecoder();
    const stdout = result.stdout ? decoder.decode(result.stdout) : "";
    const stderr = result.stderr ? decoder.decode(result.stderr) : "";
    if (result.exitCode !== 0) {
      const errorOutput = selectDiagnostic({ stderr, stdout }, `Exit code ${result.exitCode}`);
      return {
        ok: false,
        error: `ERROR: adw spec ${command} failed (exit ${result.exitCode})\n${errorOutput}`,
      };
    }
    return { ok: true, stdout };
  } catch (error: unknown) {
    const errorWithStreams = error as { stdout?: Uint8Array | string; stderr?: Uint8Array | string };
    const errorWithMessage = error as { message?: string };
    const stderr =
      errorWithStreams?.stderr
        ? typeof errorWithStreams.stderr === "string"
          ? errorWithStreams.stderr
          : Buffer.from(errorWithStreams.stderr).toString()
        : "";
    const stdout =
      errorWithStreams?.stdout
        ? typeof errorWithStreams.stdout === "string"
          ? errorWithStreams.stdout
          : Buffer.from(errorWithStreams.stdout).toString()
        : "";
    const fallback = selectDiagnostic(
      { stderr, stdout, message: errorWithMessage?.message || "" },
      "Unknown execution error",
    );
    return { ok: false, error: `ERROR: Failed to execute adw spec ${command}. ${fallback}` };
  }
}
