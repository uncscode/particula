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
    if (splitIndex < 0) {
      return REDACTION_MARKER;
    }
    return `${match.slice(0, splitIndex + 1)} ${REDACTION_MARKER}`;
  });
  redacted = redacted.replace(REDACTION_PATTERNS[2], `Authorization: Bearer ${REDACTION_MARKER}`);
  return redacted;
}

function decodeUnknown(value: unknown): string {
  if (value instanceof Uint8Array) {
    return new TextDecoder().decode(value);
  }
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function sanitizeDiagnosticText(value: string, maxChars = 4000): string {
  if (!value) {
    return "";
  }
  const noControlChars = value.replace(CONTROL_CHARS_PATTERN, " ");
  const redacted = redactSensitiveFragments(noControlChars);
  const normalized = redacted.replace(WHITESPACE_COLLAPSE_PATTERN, " ").trim();
  if (!normalized) {
    return "";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)} ...(truncated)`;
}

// --- Inlined from lib/wrapper_contract.ts (selectDiagnostic + buildErrorEnvelope) ---

const DEFAULT_ERROR_SNIPPET_LIMIT = 500;
const DEFAULT_TRUNCATION_MARKER = "... [truncated]";
const DEFAULT_EMPTY_DIAGNOSTIC_FALLBACK = "No diagnostic details available.";
const MAX_ERROR_SNIPPET_LIMIT = 10_000;

type DiagnosticSource = "stderr" | "stdout" | "message" | "fallback";

type DiagnosticSelectionResult = {
  type: DiagnosticSource;
  message: string;
};

type SelectDiagnosticOptions = {
  limit?: number;
  truncationMarker?: string;
  emptyFallback?: string;
};

type BuildErrorEnvelopeOptions = {
  prefix: string;
  contextLines?: string[];
  diagnostic?: string;
  hint?: string;
  usage?: string;
};

const clipDiagnostic = (
  value: unknown,
  limit = DEFAULT_ERROR_SNIPPET_LIMIT,
  truncationMarker = DEFAULT_TRUNCATION_MARKER,
): string => {
  const normalizedLimit =
    Number.isFinite(limit) && limit > 0
      ? Math.min(Math.floor(limit), MAX_ERROR_SNIPPET_LIMIT)
      : DEFAULT_ERROR_SNIPPET_LIMIT;
  const normalizedMarker = sanitizeDiagnosticText(truncationMarker) || DEFAULT_TRUNCATION_MARKER;

  const sanitizeBudget = Math.max(normalizedLimit * 4, normalizedLimit);
  const sanitized = sanitizeDiagnosticText(decodeUnknown(value), sanitizeBudget);
  if (!sanitized) {
    return "";
  }
  if (sanitized.length <= normalizedLimit) {
    return sanitized;
  }
  return `${sanitized.slice(0, normalizedLimit)}${normalizedMarker}`;
};

const selectDiagnostic = (
  stderr: unknown,
  stdout: unknown,
  message: unknown,
  options: SelectDiagnosticOptions = {},
): DiagnosticSelectionResult => {
  const {
    limit = DEFAULT_ERROR_SNIPPET_LIMIT,
    truncationMarker = DEFAULT_TRUNCATION_MARKER,
    emptyFallback = DEFAULT_EMPTY_DIAGNOSTIC_FALLBACK,
  } = options;

  const stderrText = clipDiagnostic(stderr, limit, truncationMarker);
  if (stderrText) {
    return { type: "stderr", message: stderrText };
  }

  const stdoutText = clipDiagnostic(stdout, limit, truncationMarker);
  if (stdoutText) {
    return { type: "stdout", message: stdoutText };
  }

  const messageText = clipDiagnostic(message, limit, truncationMarker);
  if (messageText) {
    return { type: "message", message: messageText };
  }

  const fallbackText = clipDiagnostic(emptyFallback, limit, truncationMarker);
  if (fallbackText) {
    return {
      type: "fallback",
      message: fallbackText,
    };
  }

  return {
    type: "fallback",
    message:
      clipDiagnostic(DEFAULT_EMPTY_DIAGNOSTIC_FALLBACK, limit, truncationMarker) ||
      DEFAULT_EMPTY_DIAGNOSTIC_FALLBACK,
  };
};

const buildErrorEnvelope = (options: BuildErrorEnvelopeOptions): string => {
  const sanitizedPrefix = sanitizeDiagnosticText(options.prefix);
  const normalizedPrefix = sanitizedPrefix.startsWith("ERROR:")
    ? sanitizedPrefix
    : `ERROR: ${sanitizedPrefix || "Unknown error"}`;

  const lines: string[] = [normalizedPrefix];

  for (const contextLine of options.contextLines ?? []) {
    const normalized = sanitizeDiagnosticText(String(contextLine));
    if (normalized) {
      lines.push(normalized);
    }
  }

  const diagnostic = sanitizeDiagnosticText(options.diagnostic ?? "");
  if (diagnostic) {
    lines.push(`diagnostic: ${diagnostic}`);
  }

  const hint = sanitizeDiagnosticText(options.hint ?? "");
  if (hint) {
    lines.push(`hint: ${hint}`);
  }

  const usage = sanitizeDiagnosticText(options.usage ?? "");
  if (usage) {
    lines.push(`usage: ${usage}`);
  }

  return lines.join("\n");
};

// --- Inlined from lib/refactor_astgrep_shared.ts ---

const ALLOWED_LANGUAGES = [
  "python",
  "typescript",
  "javascript",
  "cpp",
  "c",
  "go",
  "java",
  "rust",
  "csharp",
  "kotlin",
  "swift",
  "ruby",
  "php",
] as const;

const MISSING_BINARY_HINT =
  "Install ast-grep-cli (pip install ast-grep-cli) and ensure ast-grep is on PATH.";

type RefactorAstgrepNormalizedArgs = {
  pattern: string;
  rewrite: string;
  lang: (typeof ALLOWED_LANGUAGES)[number];
  path: string;
};

type RefactorAstgrepExecutionMode = "preview" | "apply";

type RefactorAstgrepErrorClassification =
  | "missing_binary"
  | "parse_input"
  | "execution";

const normalizeAndValidateArgs = (
  args: Record<string, unknown>,
): RefactorAstgrepNormalizedArgs | string => {
  const pattern = typeof args.pattern === "string" ? args.pattern.trim() : "";
  if (!pattern) {
    return "ERROR: pattern is required. Provide the AST pattern to match.";
  }

  const rewrite = typeof args.rewrite === "string" ? args.rewrite.trim() : "";
  if (!rewrite) {
    return "ERROR: rewrite is required. Provide the replacement pattern.";
  }

  const lang = typeof args.lang === "string" ? args.lang.trim() : "";
  if (!lang) {
    return `ERROR: lang is required. Choose one of: ${ALLOWED_LANGUAGES.join(", ")}.`;
  }

  if (!ALLOWED_LANGUAGES.includes(lang as (typeof ALLOWED_LANGUAGES)[number])) {
    return `ERROR: lang must be one of ${ALLOWED_LANGUAGES.join(", ")} (received ${lang}).`;
  }

  const pathInput = typeof args.path === "string" ? args.path.trim() : "";
  const pathValue = pathInput || ".";

  return {
    pattern,
    rewrite,
    lang: lang as (typeof ALLOWED_LANGUAGES)[number],
    path: pathValue,
  };
};

const buildBaseCommand = (args: RefactorAstgrepNormalizedArgs): string[] => {
  return ["ast-grep", "run", "-p", args.pattern, "-r", args.rewrite, "-l", args.lang, "--", args.path];
};

const classifyExecutionFailure = (combinedLower: string): RefactorAstgrepErrorClassification => {
  if (combinedLower.includes("enoent") || combinedLower.includes("not found")) {
    return "missing_binary";
  }

  const parseSignals = [
    "parse error",
    "failed to parse",
    "cannot parse",
    "invalid pattern",
    "invalid rewrite",
    "pattern parse",
    "rewrite parse",
  ];
  if (parseSignals.some((signal) => combinedLower.includes(signal))) {
    return "parse_input";
  }

  return "execution";
};

const formatExecutionError = (
  error: unknown,
  prefix: string,
  mode: RefactorAstgrepExecutionMode,
): string => {
  const stderr = (error as any)?.stderr?.toString?.() ?? "";
  const stdout = (error as any)?.stdout?.toString?.() ?? "";
  const message = (error as any)?.message ?? "";
  const diagnostic = selectDiagnostic(stderr, stdout, message).message;
  const combinedLower = `${stderr} ${stdout} ${message}`.toLowerCase();
  const classification = classifyExecutionFailure(combinedLower);

  const hint =
    classification === "missing_binary"
      ? MISSING_BINARY_HINT
      : classification === "parse_input"
        ? "Fix the ast-grep pattern/rewrite input and retry; this is an input parse failure, not a tooling install issue."
        : mode === "apply"
          ? "Apply mode may have partially modified files. Inspect `git diff`, restore affected paths, then retry."
          : undefined;

  return buildErrorEnvelope({
    prefix,
    contextLines: [`classification: ${classification}`],
    diagnostic,
    hint,
  });
};

// --- Tool definition ---

const DESCRIPTION = `Preview AST-aware refactors using ast-grep without mutating files.

Use this wrapper for read-only preview mode. It never appends --update-all.
For mutating rewrites, use refactor_astgrep_apply.
`;

export default tool({
  description: DESCRIPTION,
  args: {
    pattern: tool.schema.string(),
    rewrite: tool.schema.string(),
    lang: tool.schema.enum(ALLOWED_LANGUAGES),
    path: tool.schema.string().optional(),
  },
  async execute(args) {
    const normalized = normalizeAndValidateArgs(args as Record<string, unknown>);
    if (typeof normalized === "string") {
      return normalized;
    }

    const cmd = buildBaseCommand(normalized);
    try {
      const result = await Bun.$`${cmd}`.text();
      if (result.trim()) {
        return result;
      }
      return `No matches found for pattern: ${normalized.pattern}`;
    } catch (error) {
      return formatExecutionError(error, "Failed to execute 'ast-grep preview'.", "preview");
    }
  },
});
