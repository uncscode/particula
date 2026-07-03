/**
 * Atomic git staging wrapper for OpenCode integration.
 *
 * This tool is intentionally limited to staging flows:
 * - add (stage files or --all)
 * - restore (unstage files and/or --staged)
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

const OPTIONAL_KEYS = ["files", "stage_all", "staged", "worktree_path", "help"];

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

const normalizeFiles = (
  files?: string[],
): { values?: string[]; error?: string } => {
  if (!files) {
    return { values: undefined };
  }

  const normalized: string[] = [];
  for (const rawFile of files) {
    const trimmed = rawFile.trim();
    if (!trimmed) {
      continue;
    }
    if (trimmed.startsWith("-")) {
      return { error: `ERROR: Invalid files entry: ${trimmed}.` };
    }
    normalized.push(trimmed);
  }

  return { values: normalized };
};

export default tool({
  description: `Atomic git staging wrapper for add/restore flows.

SIMPLE EXAMPLES (copy these patterns):

Add files:       { command: "add", files: ["src/a.ts", "src/b.ts"] }
Add all:         { command: "add", stage_all: true }
Restore file:    { command: "restore", files: ["src/a.ts"] }
Restore staged:  { command: "restore", staged: true }
Help:            { command: "add", help: true }

RULES:
- Supports only staging commands: add, restore.
- add requires exactly one target mode: stage_all OR files.
- restore requires files and/or staged: true.
- worktree_path values starting with '-' are rejected.
- Set help: true to view command help.`,

  args: {
    command: tool.schema.enum(["add", "restore"]),
    files: tool.schema.array(tool.schema.string()).optional(),
    stage_all: tool.schema.boolean().optional(),
    staged: tool.schema.boolean().optional(),
    worktree_path: tool.schema.string().optional(),
    help: tool.schema.boolean().optional(),
  },

  async execute(rawArgs) {
    const { normalized } = normalizeSparseOptions(rawArgs, { optionalKeys: OPTIONAL_KEYS });
    const { command, files, stage_all, staged, worktree_path, help } = normalized as {
      command: "add" | "restore";
      files?: string[];
      stage_all?: boolean;
      staged?: boolean;
      worktree_path?: string;
      help?: boolean;
    };

    const normalizedFiles = normalizeFiles(files);
    if (normalizedFiles.error) {
      return normalizedFiles.error;
    }

    const cmdParts = ["uv", "run", "--active", "adw", "git"];
    const appendCommandParts = (skipValidation = false): string | undefined => {
      const normalizedWorktree = normalizeOptionalField("worktree_path", worktree_path);
      if (normalizedWorktree.error) {
        return normalizedWorktree.error;
      }

      switch (command) {
        case "add": {
          cmdParts.push("add");
          const hasStageAll = Boolean(stage_all);
          const hasFiles = Boolean(normalizedFiles.values && normalizedFiles.values.length > 0);

          if (!skipValidation && hasStageAll && hasFiles) {
            return "ERROR: 'add' command cannot combine 'stage_all' with 'files'.";
          }
          if (!skipValidation && !hasStageAll && !hasFiles) {
            return "ERROR: 'add' command requires either 'stage_all' or 'files'.";
          }

          if (hasStageAll) {
            cmdParts.push("--all");
          }
          if (hasFiles && normalizedFiles.values) {
            normalizedFiles.values.forEach((filePath) => {
              cmdParts.push("--files", filePath);
            });
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "restore": {
          cmdParts.push("restore");
          const hasFiles = Boolean(normalizedFiles.values && normalizedFiles.values.length > 0);
          const hasStaged = staged === true;
          if (!skipValidation && !hasFiles && !hasStaged) {
            return "ERROR: 'restore' command requires 'files' or 'staged: true'.";
          }
          if (hasStaged) {
            cmdParts.push("--staged");
          }
          if (hasFiles && normalizedFiles.values) {
            normalizedFiles.values.forEach((filePath) => {
              cmdParts.push("--files", filePath);
            });
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
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
        const selected = selectDiagnostic(
          error?.stderr ? error.stderr.toString() : "",
          error?.stdout ? error.stdout.toString() : "",
          typeof error?.message === "string" ? error.message : "",
        );

        if (selected.type === "stderr" || selected.type === "fallback") {
          return `ERROR: Failed to execute 'adw git ${command} --help'.\n${selected.message}`;
        }
        if (selected.type === "stdout") {
          return `ERROR: Failed to execute 'adw git ${command} --help'.\n${selected.message}`;
        }
        return `ERROR: Failed to execute 'adw git ${command} --help'.\nUnknown error occurred during execution. No output or error message was captured.`;
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
      const selected = selectDiagnostic(
        error?.stderr ? error.stderr.toString() : "",
        error?.stdout ? error.stdout.toString() : "",
        typeof error?.message === "string" ? error.message : "",
      );

      if (selected.type === "stderr" || selected.type === "fallback") {
        return `Git Command Failed: ${command}\n${selected.message}`;
      }
      if (selected.type === "stdout") {
        return `Git Command Failed:\n${selected.message}`;
      }
      return `Git Command Failed: ${command}\nCommand: ${cmdParts.join(" ")}\nUnknown error occurred during execution. No output or error message was captured.`;
    }
  },
});
