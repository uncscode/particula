/**
 * Atomic git commit wrapper for OpenCode integration.
 *
 * This tool is intentionally scoped to `uv run adw git commit` only.
 * It preserves deterministic commit validation/error-envelope behavior.
 */

import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/wrapper_contract.ts ---

type SparseNormalizationOptions = {
  optionalKeys: string[];
  falseOverrideKeys?: string[];
  noisyDefaultInertThreshold?: number;
  noisyDefaultMeaningfulThreshold?: number;
};

type NormalizedSparseResult = {
  normalized: Record<string, unknown>;
  isNoisyDefaultCall: boolean;
};

const isInertDefaultValue = (value: unknown): boolean =>
  value === undefined ||
  value === null ||
  value === false ||
  value === 0 ||
  (typeof value === "string" && value.trim() === "") ||
  (Array.isArray(value) && value.length === 0);

const isMeaningfulValue = (value: unknown): boolean => !isInertDefaultValue(value);

const normalizeOptionalString = (value: unknown): string | undefined => {
  if (value === undefined || value === null) {
    return undefined;
  }
  const text = String(value).trim();
  return text || undefined;
};

const normalizeOptionalBoolean = (value: unknown): boolean | unknown | undefined => {
  if (value === undefined || value === null || value === false) {
    return undefined;
  }
  if (value === true) {
    return true;
  }
  return value;
};

const normalizeOptionalNumber = (
  value: unknown,
  isNoisyDefaultCall: boolean,
): number | unknown | undefined => {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (value === 0 && isNoisyDefaultCall) {
    return undefined;
  }
  return value;
};

const normalizeFalseOverrideBoolean = (
  value: unknown,
  isNoisyDefaultCall: boolean,
): boolean | unknown | undefined => {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (value === false && isNoisyDefaultCall) {
    return undefined;
  }
  if (value === true || value === false) {
    return value;
  }
  return value;
};

const normalizeSparseOptions = (
  rawArgs: Record<string, unknown>,
  options: SparseNormalizationOptions,
): NormalizedSparseResult => {
  const {
    optionalKeys,
    falseOverrideKeys = ["force", "abort_on_conflict"],
    noisyDefaultInertThreshold = 8,
    noisyDefaultMeaningfulThreshold = 6,
  } = options;

  const inertDefaultCount = optionalKeys.filter(
    (key) => key in rawArgs && isInertDefaultValue(rawArgs[key]),
  ).length;
  const meaningfulValueCount = optionalKeys.filter(
    (key) => key in rawArgs && isMeaningfulValue(rawArgs[key]),
  ).length;

  const isNoisyDefaultCall =
    inertDefaultCount >= noisyDefaultInertThreshold &&
    meaningfulValueCount <= noisyDefaultMeaningfulThreshold;

  const normalized: Record<string, unknown> = { ...rawArgs };
  for (const key of optionalKeys) {
    if (!(key in rawArgs)) {
      continue;
    }
    const value = rawArgs[key];
    if (Array.isArray(value)) {
      normalized[key] = value.length === 0 ? undefined : value;
      continue;
    }
    if (falseOverrideKeys.includes(key)) {
      normalized[key] = normalizeFalseOverrideBoolean(value, isNoisyDefaultCall);
      continue;
    }
    if (typeof value === "string" || value === undefined || value === null) {
      normalized[key] = normalizeOptionalString(value);
      continue;
    }
    if (typeof value === "boolean") {
      normalized[key] = normalizeOptionalBoolean(value);
      continue;
    }
    if (typeof value === "number") {
      normalized[key] = normalizeOptionalNumber(value, isNoisyDefaultCall);
      continue;
    }
    normalized[key] = value;
  }

  return { normalized, isNoisyDefaultCall };
};

// --- Inlined from lib/git_shared.ts ---

const DEFAULT_TRUNCATION_MARKER = "... [truncated]";

const normalizeSnippet = (value: unknown): string => {
  const raw = typeof value === "string" ? value : String(value ?? "");
  return raw.replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").trim();
};

const clipDiagnostic = (
  value: unknown,
  limit = 500,
  truncationMarker = "... [truncated]",
): string => {
  const text = normalizeSnippet(value);
  if (!text) return "";
  return text.length > limit ? `${text.slice(0, limit)}${truncationMarker}` : text;
};

const selectDiagnostic = (
  stderr: unknown,
  stdout: unknown,
  fallback: unknown,
  limit = 500,
  truncationMarker = "... [truncated]",
): { type: "stderr" | "stdout" | "fallback" | "none"; message: string } => {
  const stderrText = clipDiagnostic(stderr, limit, truncationMarker);
  if (stderrText) return { type: "stderr", message: stderrText };
  const stdoutText = clipDiagnostic(stdout, limit, truncationMarker);
  if (stdoutText) return { type: "stdout", message: stdoutText };
  const fallbackText = clipDiagnostic(fallback, limit, truncationMarker);
  if (fallbackText) return { type: "fallback", message: fallbackText };
  return { type: "none", message: "" };
};

// --- Tool definition ---

const ERROR_SNIPPET_LIMIT = 500;
const TRUNCATION_MARKER = DEFAULT_TRUNCATION_MARKER;

const getCommitHint = (stderr: string, stdout: string): string => {
  const combined = `${stderr}\n${stdout}`.toLowerCase();

  if (
    combined.includes("index.lock") ||
    combined.includes(".lock") ||
    combined.includes("unable to create") ||
    combined.includes("another git process") ||
    combined.includes("could not lock") ||
    combined.includes("lock file")
  ) {
    return "Git lock contention detected; remove stale lock files or wait for other git processes to finish.";
  }

  if (combined.includes("pre-commit") || combined.includes("hook")) {
    return "Commit was rejected by git hooks; review hook output and fix reported issues before retrying.";
  }

  if (combined.includes("nothing to commit") || combined.includes("nothing added")) {
    return "Working tree is clean; stage or modify files before committing.";
  }

  return "Inspect stderr/stdout details and rerun with corrected inputs or repository state.";
};

const buildCommitError = (error: any, worktreePath?: string): string => {
  const stderr = error?.stderr ? normalizeSnippet(error.stderr.toString()) : "";
  const stdout = error?.stdout ? normalizeSnippet(error.stdout.toString()) : "";
  const fallbackMessage =
    typeof error?.message === "string" ? normalizeSnippet(error.message) : "";
  const exitCode =
    typeof error?.exitCode === "number"
      ? error.exitCode
      : typeof error?.code === "number"
        ? error.code
        : "unknown";

  const lines = [
    "ERROR: Failed to execute 'adw git commit'",
    "command: commit",
    `exit_code: ${exitCode}`,
  ];

  if (worktreePath) {
    lines.push(`worktree_path: ${worktreePath}`);
  }

  const selected = selectDiagnostic(
    stderr,
    stdout,
    fallbackMessage,
    ERROR_SNIPPET_LIMIT,
    TRUNCATION_MARKER,
  );

  if (selected.type === "stderr") {
    lines.push(`stderr: ${clipDiagnostic(stderr, ERROR_SNIPPET_LIMIT, TRUNCATION_MARKER)}`);
    const stdoutText = clipDiagnostic(stdout, ERROR_SNIPPET_LIMIT, TRUNCATION_MARKER);
    if (stdoutText) {
      lines.push(`stdout: ${stdoutText}`);
    }
  } else if (selected.type === "stdout") {
    lines.push(`stdout: ${clipDiagnostic(stdout, ERROR_SNIPPET_LIMIT, TRUNCATION_MARKER)}`);
  } else if (selected.type === "fallback") {
    lines.push(`message: ${clipDiagnostic(fallbackMessage, ERROR_SNIPPET_LIMIT, TRUNCATION_MARKER)}`);
  }

  lines.push(`hint: ${getCommitHint(stderr || fallbackMessage, stdout)}`);
  return lines.join("\n");
};

export default tool({
  description: `Execute adw git commit with deterministic validation and diagnostics.

SIMPLE EXAMPLES (copy these patterns):

  Summary only:     { summary: "Add feature" }
  With description: { summary: "Add feature", description: "More details" }
  With no-verify:   { summary: "Hotfix", no_verify: true }
  In worktree:      { summary: "Update", worktree_path: "./trees/abc" }
  Retry config:     { summary: "Update", max_retries: 5 }

RULES:
- summary is required and must be non-empty after trim.
- no_verify must be boolean when provided.
- max_retries must be an integer between 0 and 10 (default 3).
- Uses deterministic error envelope compatible with git commit wrapper parsing.`,

  args: {
    summary: tool.schema.string().describe("Commit summary line (required)."),
    description: tool.schema.string().optional().describe("Optional commit body."),
    adw_id: tool.schema.string().optional().describe("Optional ADW workflow id."),
    worktree_path: tool.schema.string().optional().describe("Optional worktree path."),
    stage_all: tool.schema.boolean().optional().describe("Stage all changes before commit."),
    max_retries: tool.schema
      .number()
      .optional()
      .describe("Retry count for hook-modified commits (0..10, default 3)."),
    no_verify: tool.schema
      .boolean()
      .optional()
      .describe("Bypass hooks only when true (adds --no-verify)."),
  },

  async execute(args) {
    const cmdParts = ["uv", "run", "adw", "git", "commit"];

    const rawArgs = args as Record<string, unknown>;
    const { normalized } = normalizeSparseOptions(rawArgs, {
      optionalKeys: [
        "summary",
        "description",
        "adw_id",
        "worktree_path",
        "stage_all",
        "max_retries",
        "no_verify",
      ],
      falseOverrideKeys: [],
    });

    const summary =
      typeof normalized.summary === "string" ? normalized.summary.trim() : undefined;
    const description =
      normalized.description === undefined || normalized.description === null
        ? undefined
        : String(normalized.description).trim() || undefined;
    const adwId =
      normalized.adw_id === undefined || normalized.adw_id === null
        ? undefined
        : String(normalized.adw_id).trim() || undefined;
    const worktreePath =
      normalized.worktree_path === undefined || normalized.worktree_path === null
        ? undefined
        : String(normalized.worktree_path).trim() || undefined;
    const stageAll = normalized.stage_all;
    const maxRetries = normalized.max_retries;
    const noVerify = normalized.no_verify;

    if (!summary) {
      return "ERROR: 'commit' command requires non-empty 'summary'.";
    }
    if (noVerify !== undefined && typeof noVerify !== "boolean") {
      return "ERROR: 'no_verify' must be a boolean when provided.";
    }
    if (worktreePath?.startsWith("-")) {
      return "ERROR: 'worktree_path' cannot start with '-'.";
    }

    const retries = maxRetries ?? 3;
    if (!Number.isInteger(retries) || Number(retries) < 0 || Number(retries) > 10) {
      return "ERROR: 'max_retries' must be an integer between 0 and 10.";
    }

    cmdParts.push("--summary", summary);
    if (description) {
      cmdParts.push("--description", description);
    }
    if (adwId) {
      cmdParts.push("--adw-id", adwId);
    }
    if (worktreePath) {
      cmdParts.push("--worktree-path", worktreePath);
    }
    if (stageAll === true) {
      cmdParts.push("--stage-all");
    }
    if (noVerify === true) {
      cmdParts.push("--no-verify");
    }
    cmdParts.push("--max-retries", String(retries));

    try {
      const output = await Bun.$`${cmdParts}`.text();
      return `Git Commit Command\n\n${output}`;
    } catch (error: any) {
      return buildCommitError(error, worktreePath);
    }
  },
});
