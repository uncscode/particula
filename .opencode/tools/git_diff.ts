import { tool } from "@opencode-ai/plugin";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";

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

const normalizeRef = (value?: string): string => {
  if (!value) {
    return "";
  }
  const trimmed = value.trim();
  if (trimmed.toLowerCase().startsWith("refs/heads/")) {
    return trimmed.slice("refs/heads/".length).trim();
  }
  return trimmed;
};

const isValidRefToken = (value: string): boolean => {
  if (!value) {
    return false;
  }
  if (value.startsWith("-")) {
    return false;
  }
  if (
    value.includes("..") ||
    value.includes("@{") ||
    value.includes("//") ||
    value.startsWith("/") ||
    value.endsWith("/") ||
    value.endsWith(".") ||
    value.endsWith(".lock") ||
    value.includes("\x00") ||
    /[ :?*\[\]\\]/.test(value)
  ) {
    return false;
  }
  return /^[A-Za-z0-9._/\-~^]+$/.test(value);
};

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

const findRepoRoot = (): string => {
  let current = resolve(process.cwd());
  while (true) {
    if (existsSync(join(current, "AGENTS.md")) && existsSync(join(current, ".opencode"))) {
      return current;
    }
    const parent = resolve(current, "..");
    if (parent === current) {
      return resolve(process.cwd());
    }
    current = parent;
  }
};

const formatDebugValue = (value: unknown): string => {
  if (value === undefined || value === null) {
    return "";
  }
  if (Buffer.isBuffer(value)) {
    return value.toString();
  }
  return String(value);
};

const writeDebugLog = async (
  command: string,
  cmdParts: string[],
  error: any,
): Promise<string | undefined> => {
  try {
    const repoRoot = findRepoRoot();
    const relativeLogPath = join(
      "adforge_local",
      "opencode",
      "tmp",
      `git_diff-${command}-${Date.now()}-${randomUUID().slice(0, 8)}.log`,
    );
    const absoluteLogPath = join(repoRoot, relativeLogPath);
    const content = [
      `command: ${command}`,
      `argv: ${cmdParts.join(" ")}`,
      `cwd: ${process.cwd()}`,
      "",
      "stderr:",
      formatDebugValue(error?.stderr),
      "",
      "stdout:",
      formatDebugValue(error?.stdout),
      "",
      "message:",
      formatDebugValue(error?.message),
      "",
      "stack:",
      formatDebugValue(error?.stack),
      "",
    ].join("\n");

    await mkdir(dirname(absoluteLogPath), { recursive: true });
    await writeFile(absoluteLogPath, content, { encoding: "utf8", flag: "wx" });
    return relative(repoRoot, absoluteLogPath);
  } catch {
    return undefined;
  }
};

const appendDebugLog = (message: string, debugLog?: string): string => {
  if (!debugLog) {
    return message;
  }
  return `${message}\ndebug_log: ${debugLog}`;
};

// --- Tool definition ---

const OPTIONAL_KEYS = [
  "worktree_path",
  "porcelain",
  "stat",
  "base",
  "target",
  "ref",
  "path",
  "max_count",
  "oneline",
  "help",
];

const normalizeOptionalField = (fieldName: string, value?: string): { value?: string; error?: string } => {
  if (value === undefined || value === null) {
    return { value: undefined };
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return { value: undefined };
  }
  if (trimmed.startsWith("-")) {
    return { error: `ERROR: Invalid ${fieldName}: ${value}.` };
  }
  return { value: trimmed };
};

export default tool({
  description: `Execute ADW read-only git inspection commands (status, diff, log, show).

SIMPLE EXAMPLES (copy these patterns):

Status:    { command: "status", porcelain: true, worktree_path: "./trees/abc" }
Diff:      { command: "diff", stat: true, worktree_path: "./trees/abc" }
Diff base: { command: "diff", base: "main", target: "feature", stat: true }
Log:       { command: "log", max_count: 5, oneline: true, ref: "main" }
Show:      { command: "show", ref: "HEAD~1", path: "adw/core/" }

RULES:
- Supports only read-only commands: status, diff, log, show.
- Empty strings, false booleans, and noisy numeric defaults are treated as omitted.
- show requires ref (unless help: true).
- log max_count must be an integer between 1 and 1000.
- Set help: true to view CLI help for any command.`,

  args: {
    command: tool.schema.enum(["status", "diff", "log", "show"]),
    worktree_path: tool.schema.string().optional(),
    porcelain: tool.schema.boolean().optional(),
    stat: tool.schema.boolean().optional(),
    base: tool.schema.string().optional(),
    target: tool.schema.string().optional(),
    ref: tool.schema.string().optional(),
    path: tool.schema.string().optional(),
    max_count: tool.schema.number().optional(),
    oneline: tool.schema.boolean().optional(),
    help: tool.schema.boolean().optional(),
  },

  async execute(rawArgs) {
    const { normalized } = normalizeSparseOptions(rawArgs, { optionalKeys: OPTIONAL_KEYS });
    const {
      command,
      worktree_path,
      porcelain,
      stat,
      base,
      target,
      ref,
      path,
      max_count,
      oneline,
      help,
    } = normalized as {
      command: "status" | "diff" | "log" | "show";
      worktree_path?: string;
      porcelain?: boolean;
      stat?: boolean;
      base?: string;
      target?: string;
      ref?: string;
      path?: string;
      max_count?: number;
      oneline?: boolean;
      help?: boolean;
    };

    const cmdParts: string[] = ["uv", "run", "adw", "git"];
    const appendCommandParts = (skipValidation = false): string | undefined => {
      switch (command) {
        case "status":
          cmdParts.push("status");
          if (porcelain) {
            cmdParts.push("--porcelain");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        case "diff": {
          cmdParts.push("diff");
          if (stat) {
            cmdParts.push("--stat");
          }
          const normalizedBase = normalizeOptionalField("base", base);
          if (!skipValidation && normalizedBase.error) {
            return normalizedBase.error;
          }
          if (normalizedBase.value) {
            cmdParts.push("--base", normalizedBase.value);
          }
          const normalizedTarget = normalizeOptionalField("target", target);
          if (!skipValidation && normalizedTarget.error) {
            return normalizedTarget.error;
          }
          if (normalizedTarget.value) {
            cmdParts.push("--target", normalizedTarget.value);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }
        case "log": {
          cmdParts.push("log");
          const normalizedLogRef = normalizeRef(ref);
          if (!skipValidation && ref !== undefined && !normalizedLogRef) {
            return "ERROR: 'log' command requires non-empty 'ref' when provided.";
          }
          if (!skipValidation && normalizedLogRef && !isValidRefToken(normalizedLogRef)) {
            return `ERROR: Invalid ref: ${ref}.`;
          }
          if (normalizedLogRef) {
            cmdParts.push("--ref", normalizedLogRef);
          }
          const boundedMaxCount = max_count ?? 10;
          if (
            !skipValidation &&
            (!Number.isInteger(boundedMaxCount) || boundedMaxCount < 1 || boundedMaxCount > 1000)
          ) {
            return "ERROR: 'max_count' must be an integer between 1 and 1000.";
          }
          cmdParts.push("--max-count", boundedMaxCount.toString());
          if (oneline) {
            cmdParts.push("--oneline");
          }
          if (stat) {
            cmdParts.push("--stat");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }
        case "show": {
          cmdParts.push("show");
          if (!skipValidation && (!ref || !ref.trim())) {
            return "ERROR: 'show' command requires 'ref'.";
          }
          const normalizedShowRef = normalizeOptionalField("ref", ref);
          if (!skipValidation && normalizedShowRef.error) {
            return normalizedShowRef.error;
          }
          if (normalizedShowRef.value) {
            cmdParts.push("--ref", normalizedShowRef.value);
          }
          const normalizedPath = normalizeOptionalField("path", path);
          if (!skipValidation && normalizedPath.error) {
            return normalizedPath.error;
          }
          if (normalizedPath.value) {
            cmdParts.push("--path", normalizedPath.value);
          }
          if (stat) {
            cmdParts.push("--stat");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }
        default:
          return `ERROR: Unsupported command '${command}'.`;
      }
    };

    if (help) {
      const validationMessage = appendCommandParts(true);
      if (validationMessage?.startsWith("ERROR")) {
        return validationMessage;
      }
      cmdParts.push("--help");
      try {
        const result = await Bun.$`${cmdParts}`.text();
        return `Git Command: ${command} (help)\n\n${result}`;
      } catch (error: any) {
        const debugLog = await writeDebugLog(`${command}-help`, cmdParts, error);
        const selected = selectDiagnostic(
          error?.stderr ? error.stderr.toString() : "",
          error?.stdout ? error.stdout.toString() : "",
          typeof error?.message === "string" ? error.message : "",
        );

        if (selected.type === "stderr" || selected.type === "fallback") {
          return appendDebugLog(
            `ERROR: Failed to execute 'adw git ${command} --help'.\n${selected.message}`,
            debugLog,
          );
        }
        if (selected.type === "stdout") {
          return appendDebugLog(
            `ERROR: Failed to execute 'adw git ${command} --help'.\n${selected.message}`,
            debugLog,
          );
        }
        return appendDebugLog(
          `ERROR: Failed to execute 'adw git ${command} --help'.\nUnknown error occurred during execution. No output or error message was captured.`,
          debugLog,
        );
      }
    }

    const validationMessage = appendCommandParts();
    if (validationMessage?.startsWith("ERROR")) {
      return validationMessage;
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return `Git Command: ${command}\n\n${result}`;
    } catch (error: any) {
      const debugLog = await writeDebugLog(command, cmdParts, error);
      const selected = selectDiagnostic(
        error?.stderr ? error.stderr.toString() : "",
        error?.stdout ? error.stdout.toString() : "",
        typeof error?.message === "string" ? error.message : "",
      );

      if (selected.type === "stderr" || selected.type === "fallback") {
        return appendDebugLog(`Git Command Failed: ${command}\n${selected.message}`, debugLog);
      }
      if (selected.type === "stdout") {
        return appendDebugLog(`Git Command Failed:\n${selected.message}`, debugLog);
      }
      return appendDebugLog(
        `Git Command Failed: ${command}\nCommand: ${cmdParts.join(" ")}\nUnknown error occurred during execution. No output or error message was captured.`,
        debugLog,
      );
    }
  },
});
