/**
 * Function signature for wrapper-local deterministic error envelopes.
 */
export type BuildError = (message: string) => string;

/**
 * Required-argument metadata for a specific plans command.
 */
export type RequiredArgSpec = {
  field: string;
  message: string;
};

/**
 * Map of command names to their required-argument checks.
 */
export type RequiredArgMap = Record<string, RequiredArgSpec[]>;

/**
 * Sanitized command-failure text and visibility state.
 */
export type SanitizedOutput = {
  text: string;
  hasVisibleContent: boolean;
};

type CommandFailureSources = {
  stderr?: string;
  stdout?: string;
  message?: string;
};

const OUTPUT_CHAR_LIMIT = 4_000;
const CONTROL_CHAR_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WINDOWS_ABSOLUTE_PATH_PATTERN = /[A-Za-z]:\\(?:[^\\\r\n"']+\\)*[^\\\r\n"']+/g;
const QUOTED_UNIX_ABSOLUTE_PATH_PATTERN = /(["'])(\/[^\r\n"']+)\1/g;
const UNIX_COLON_PATH_PATTERN = /(^|[\s(\[])(\/(?:[^:\r\n]|:(?!\s))+?)(?=:\s|:\d|$)/gm;
const UNIX_BARE_PATH_PATTERN = /(^|[\s(\[])(\/(?:[^\s)\]"']|\s+(?![-\w]+:))+)/gm;
const REDACTED_SECRET = "<redacted-secret>";
const SECRET_ASSIGNMENT_PATTERNS = [
  /\b(authorization\s*:\s*bearer\s+)([^\s]+)/gi,
  /\b((?:token|secret|password|passwd|api(?:_|-)?key|access(?:_|-)?token|refresh(?:_|-)?token)\s*[:=]\s*)("?)([^\s",']+)("?)/gi,
  /\b(gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]+)\b/g,
];

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

export function redactPathLikeText(raw: string): string {
  const sanitized = sanitizeCommandFailureOutput(raw);
  return sanitized.hasVisibleContent ? sanitized.text : "<path>";
}

/**
 * Sanitize spawned-command diagnostics for the adw_plans wrapper family.
 *
 * Removes control characters, redacts absolute path-like substrings, and
 * preserves the existing bounded truncation contract.
 *
 * Args:
 *   raw: Raw diagnostic text from stderr/stdout/message.
 *
 * Returns:
 *   A normalized text payload plus a visibility flag.
 */
export function sanitizeCommandFailureOutput(raw: string): SanitizedOutput {
  if (!raw) {
    return { text: "", hasVisibleContent: false };
  }

  const normalized = raw.replace(CONTROL_CHAR_PATTERN, "");
  const redacted = redactSecrets(redactAbsolutePaths(normalized)).replace(
    /<path>(?:\s+<path>)+/g,
    "<path>",
  );
  if (!redacted) {
    return { text: "", hasVisibleContent: false };
  }

  const hasVisibleContent = Boolean(redacted.trim());
  let output = redacted;
  if (output.length > OUTPUT_CHAR_LIMIT) {
    const originalLength = output.length;
    output = output.slice(0, OUTPUT_CHAR_LIMIT);
    output += `\n...[output truncated to ${OUTPUT_CHAR_LIMIT} characters; original length ${originalLength}]`;
  }

  return { text: output, hasVisibleContent };
}

export function sanitizeSuccessOutput(raw: string): SanitizedOutput {
  if (!raw) {
    return { text: "", hasVisibleContent: false };
  }

  const normalized = raw.replace(CONTROL_CHAR_PATTERN, "");
  if (!normalized) {
    return { text: "", hasVisibleContent: false };
  }

  const hasVisibleContent = Boolean(normalized.trim());
  let output = normalized;
  if (output.length > OUTPUT_CHAR_LIMIT) {
    const originalLength = output.length;
    output = output.slice(0, OUTPUT_CHAR_LIMIT);
    output += `\n...[output truncated to ${OUTPUT_CHAR_LIMIT} characters; original length ${originalLength}]`;
  }

  return { text: output, hasVisibleContent };
}

/**
 * Select the canonical spawned-command diagnostic using fixed precedence.
 *
 * Args:
 *   sources: Candidate stderr/stdout/message strings.
 *   fallback: Fallback text when every source is empty.
 *
 * Returns:
 *   The selected sanitized diagnostic text.
 */
export function selectCommandFailureDiagnostic(
  sources: CommandFailureSources,
  fallback: string,
): string {
  const safeStderr = sanitizeCommandFailureOutput(sources.stderr ?? "");
  const safeStdout = sanitizeCommandFailureOutput(sources.stdout ?? "");
  const safeMessage = sanitizeCommandFailureOutput(sources.message ?? "");
  if (safeStderr.hasVisibleContent) {
    return safeStderr.text;
  }
  if (safeStdout.hasVisibleContent) {
    return safeStdout.text;
  }
  if (safeMessage.hasVisibleContent) {
    return safeMessage.text;
  }
  return fallback;
}

/**
 * Derive a bounded next-safe-action hint for recognized failure classes.
 *
 * Args:
 *   diagnostic: Selected sanitized diagnostic text.
 *
 * Returns:
 *   A hint line when the failure class is recognized; otherwise undefined.
 */
export function deriveCommandFailureHint(diagnostic: string): string | undefined {
  if (!diagnostic) {
    return undefined;
  }

  if (
    /(?:ENOENT|python3(?:\s*:)?\s+not found|uv(?:\s*:)?\s+not found|can't open file|No such file or directory|Cannot find module)/i.test(
      diagnostic,
    )
  ) {
    return "hint: verify the required runtime/tooling is installed and the backend script exists in this repository.";
  }

  if (/(?:\bcwd path does not exist\b|\bcwd path is not a directory\b|\bcwd path resolves outside repository root\b|\b--cwd\b)/i.test(diagnostic)) {
    return "hint: verify --cwd points to an existing in-repository repository/worktree root.";
  }

  return undefined;
}

/**
 * Build the shared adw_plans spawned-command failure envelope.
 *
 * Args:
 *   command: Plans subcommand name.
 *   reason: Stable failure reason segment.
 *   sources: Candidate stderr/stdout/message strings.
 *   fallback: Fallback text when diagnostics are empty.
 *
 * Returns:
 *   Deterministic ERROR envelope with optional recognized-action hint.
 */
export function buildCommandFailureError(
  command: string,
  reason: string,
  sources: CommandFailureSources,
  fallback: string,
): string {
  const diagnostic = selectCommandFailureDiagnostic(sources, fallback);
  const hint = deriveCommandFailureHint(diagnostic);
  const suffix = hint ? `\n${hint}` : "";
  return `ERROR: adw plans ${command} failed (${reason}).\n${diagnostic}${suffix}`;
}

export function stripDefaultArgs(raw: Record<string, any>): Record<string, any> {
  const cleaned: Record<string, any> = { command: raw.command };
  for (const [key, value] of Object.entries(raw)) {
    if (key === "command") continue;
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.trim() === "") continue;
    if (value === false) continue;
    cleaned[key] = value;
  }
  return cleaned;
}

export function validateRequiredArgs(
  raw: Record<string, any>,
  requirements: RequiredArgMap,
  buildError: BuildError,
): string | undefined {
  const command = String(raw.command ?? "");
  const specs = requirements[command] ?? [];
  for (const spec of specs) {
    const value = raw[spec.field];
    if (typeof value !== "string" || value.trim() === "") {
      return buildError(spec.message);
    }
  }
  return undefined;
}
