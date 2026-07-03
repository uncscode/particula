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

// --- Tool definition ---

const OPTIONAL_KEYS = ["adw_id", "force", "help"];
const CANONICAL_ADW_ID_PATTERN = /^[0-9a-f]{8}$/;

const isCanonicalAdwId = (value: string): boolean => CANONICAL_ADW_ID_PATTERN.test(value);

export default tool({
  description: `Manage git worktrees with a narrow ADW git wrapper (worktree-list, worktree-prune, worktree-remove).

SIMPLE EXAMPLES (copy these patterns):

List:   { command: "worktree-list" }
Prune:  { command: "worktree-prune" }
Remove: { command: "worktree-remove", adw_id: "abc12345" }
No force remove: { command: "worktree-remove", adw_id: "abc12345", force: false }

RULES:
- Supports only worktree commands: worktree-list, worktree-prune, worktree-remove.
- worktree-remove requires adw_id unless help: true.
- worktree-remove includes --force by default; set force: false to omit.
- Set help: true to view CLI help for any command.`,

  args: {
    command: tool.schema.enum(["worktree-list", "worktree-prune", "worktree-remove"]),
    adw_id: tool.schema.string().optional(),
    force: tool.schema.boolean().optional(),
    help: tool.schema.boolean().optional(),
  },

  async execute(rawArgs) {
    const { normalized } = normalizeSparseOptions(rawArgs, {
      optionalKeys: OPTIONAL_KEYS,
      falseOverrideKeys: ["force"],
    });

    const { command, adw_id, force, help } = normalized as {
      command: "worktree-list" | "worktree-prune" | "worktree-remove";
      adw_id?: string;
      force?: boolean;
      help?: boolean;
    };

    const cmdParts: string[] = ["uv", "run", "--active", "adw", "git"];

    const appendCommandParts = (skipMissingRequiredValidation = false): string | undefined => {
      switch (command) {
        case "worktree-list":
          cmdParts.push("worktree", "list");
          return undefined;
        case "worktree-prune":
          cmdParts.push("worktree", "prune");
          return undefined;
        case "worktree-remove": {
          cmdParts.push("worktree", "remove");
          if (!skipMissingRequiredValidation && !adw_id) {
            return "ERROR: 'worktree-remove' command requires 'adw_id'.";
          }
          if (adw_id && !isCanonicalAdwId(adw_id)) {
            return `ERROR: Invalid adw_id: ${adw_id}. Expected 8 lowercase hex characters.`;
          }
          if (adw_id && !isValidRefToken(adw_id)) {
            return `ERROR: Invalid adw_id: ${adw_id}.`;
          }
          if (adw_id) {
            cmdParts.push(adw_id);
          }
          if (force !== false) {
            cmdParts.push("--force");
          }
          return undefined;
        }
      }
    };

    const executeCommand = async (helpMode = false): Promise<string> => {
      if (helpMode) {
        cmdParts.push("--help");
      }
      try {
        const result = await Bun.$`${cmdParts}`.text();
        const header = helpMode ? `${command} (help)` : command;
        return `Git Command: ${header}\n\n${result}`;
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
    };

    if (help) {
      const validationMessage = appendCommandParts(true);
      if (validationMessage?.startsWith("ERROR")) {
        return validationMessage;
      }
      return executeCommand(true);
    }

    const validationMessage = appendCommandParts();
    if (validationMessage?.startsWith("ERROR")) {
      return validationMessage;
    }

    return executeCommand();
  },
});
