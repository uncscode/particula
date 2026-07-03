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

// --- Tool definition ---

const OPTIONAL_KEYS = [
  "source",
  "target",
  "no_ff",
  "abort_on_conflict",
  "branch",
  "onto",
  "remote",
  "prune",
  "slice_branch",
  "tracking_branch",
  "recover_missing_worktree",
  "ref",
  "hard",
  "worktree_path",
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
  description: `Move git merge/rebase lifecycle with a narrow ADW wrapper (merge, rebase, fetch, sync, accumulate, abort, continue, reset).

SIMPLE EXAMPLES (copy these patterns):

Merge:      { command: "merge", source: "main", target: "develop" }
Rebase:     { command: "rebase", branch: "feature-123", onto: "main" }
Fetch:      { command: "fetch", remote: "origin", branch: "main" }
Sync:       { command: "sync", source: "upstream", target: "main" }
Accumulate: { command: "accumulate", slice_branch: "issue-1", tracking_branch: "accumulate/F1" }
Abort:      { command: "abort" }
Continue:   { command: "continue" }
Reset:      { command: "reset", ref: "HEAD~1", hard: true }

RULES:
- Supports only merge lifecycle commands: merge, rebase, fetch, sync, accumulate, abort, continue, reset.
- merge requires source unless help: true.
- rebase requires branch unless help: true.
- accumulate requires slice_branch and tracking_branch unless help: true.
- reset requires ref unless help: true.
- Set help: true to view CLI help for any command.`,

  args: {
    command: tool.schema.enum([
      "merge",
      "rebase",
      "fetch",
      "sync",
      "accumulate",
      "abort",
      "continue",
      "reset",
    ]),
    source: tool.schema.string().optional(),
    target: tool.schema.string().optional(),
    no_ff: tool.schema.boolean().optional(),
    abort_on_conflict: tool.schema.boolean().optional(),
    branch: tool.schema.string().optional(),
    onto: tool.schema.string().optional(),
    remote: tool.schema.string().optional(),
    prune: tool.schema.boolean().optional(),
    slice_branch: tool.schema.string().optional(),
    tracking_branch: tool.schema.string().optional(),
    recover_missing_worktree: tool.schema.boolean().optional(),
    ref: tool.schema.string().optional(),
    hard: tool.schema.boolean().optional(),
    worktree_path: tool.schema.string().optional(),
    help: tool.schema.boolean().optional(),
  },

  async execute(rawArgs) {
    const { normalized } = normalizeSparseOptions(rawArgs, {
      optionalKeys: OPTIONAL_KEYS,
      falseOverrideKeys: ["abort_on_conflict"],
    });
    const {
      command,
      source,
      target,
      no_ff,
      abort_on_conflict,
      branch,
      onto,
      remote,
      prune,
      slice_branch,
      tracking_branch,
      recover_missing_worktree,
      ref,
      hard,
      worktree_path,
      help,
    } = normalized as {
      command: "merge" | "rebase" | "fetch" | "sync" | "accumulate" | "abort" | "continue" | "reset";
      source?: string;
      target?: string;
      no_ff?: boolean;
      abort_on_conflict?: boolean;
      branch?: string;
      onto?: string;
      remote?: string;
      prune?: boolean;
      slice_branch?: string;
      tracking_branch?: string;
      recover_missing_worktree?: boolean;
      ref?: string;
      hard?: boolean;
      worktree_path?: string;
      help?: boolean;
    };

    const cmdParts: string[] = ["uv", "run", "--active", "adw", "git"];

    const appendCommandParts = (skipRequiredChecks = false): string | undefined => {
      const normalizedWorktree = normalizeOptionalField("worktree_path", worktree_path);
      if (normalizedWorktree.error) {
        return normalizedWorktree.error;
      }

      switch (command) {
        case "merge": {
          cmdParts.push("merge");
          const normalizedSource = normalizeRef(source);
          if (!skipRequiredChecks && !normalizedSource) {
            return "ERROR: 'merge' command requires 'source'.";
          }
          if (normalizedSource && !isValidRefToken(normalizedSource)) {
            return `ERROR: Invalid source: ${source}.`;
          }
          if (normalizedSource) {
            cmdParts.push(normalizedSource);
          }
          const normalizedTarget = normalizeRef(target);
          if (!skipRequiredChecks && target !== undefined && !normalizedTarget) {
            return "ERROR: 'target' must be non-empty when provided.";
          }
          if (normalizedTarget && !isValidRefToken(normalizedTarget)) {
            return `ERROR: Invalid target: ${target}.`;
          }
          if (normalizedTarget) {
            cmdParts.push("--into", normalizedTarget);
          }
          if (no_ff) {
            cmdParts.push("--no-ff");
          }
          if (abort_on_conflict === false) {
            cmdParts.push("--no-abort-on-conflict");
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "rebase": {
          cmdParts.push("rebase");
          const normalizedBranch = normalizeRef(branch);
          if (!skipRequiredChecks && !normalizedBranch) {
            return "ERROR: 'rebase' command requires 'branch'.";
          }
          if (normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (normalizedBranch) {
            cmdParts.push(normalizedBranch);
          }
          const normalizedOnto = normalizeRef(onto);
          if (!skipRequiredChecks && onto !== undefined && !normalizedOnto) {
            return "ERROR: 'onto' must be non-empty when provided.";
          }
          if (normalizedOnto && !isValidRefToken(normalizedOnto)) {
            return `ERROR: Invalid onto: ${onto}.`;
          }
          if (normalizedOnto) {
            cmdParts.push("--onto", normalizedOnto);
          }
          if (abort_on_conflict === false) {
            cmdParts.push("--no-abort-on-conflict");
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "fetch": {
          cmdParts.push("fetch");
          const normalizedRemote = normalizeOptionalField("remote", remote);
          const normalizedBranch = normalizeRef(branch);
          if (normalizedRemote.error) {
            return normalizedRemote.error;
          }
          if (!skipRequiredChecks && branch !== undefined && !normalizedBranch) {
            return "ERROR: 'branch' must be non-empty when provided.";
          }
          if (normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (normalizedRemote.value) {
            cmdParts.push("--remote", normalizedRemote.value);
          } else {
            cmdParts.push("--remote", "origin");
          }
          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (prune) {
            cmdParts.push("--prune");
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "sync": {
          cmdParts.push("sync");
          const normalizedSource = normalizeRef(source);
          if (!skipRequiredChecks && source !== undefined && !normalizedSource) {
            return "ERROR: 'source' must be non-empty when provided.";
          }
          if (normalizedSource && !isValidRefToken(normalizedSource)) {
            return `ERROR: Invalid source: ${source}.`;
          }
          if (normalizedSource) {
            cmdParts.push("--source", normalizedSource);
          }
          const normalizedTarget = normalizeRef(target);
          if (!skipRequiredChecks && target !== undefined && !normalizedTarget) {
            return "ERROR: 'target' must be non-empty when provided.";
          }
          if (normalizedTarget && !isValidRefToken(normalizedTarget)) {
            return `ERROR: Invalid target: ${target}.`;
          }
          if (normalizedTarget) {
            cmdParts.push("--target", normalizedTarget);
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "accumulate": {
          cmdParts.push("accumulate");
          const normalizedSliceBranch = normalizeRef(slice_branch);
          const normalizedTrackingBranch = normalizeRef(tracking_branch);
          if (!skipRequiredChecks && !normalizedSliceBranch) {
            return "ERROR: 'accumulate' command requires 'slice_branch'.";
          }
          if (!skipRequiredChecks && !normalizedTrackingBranch) {
            return "ERROR: 'accumulate' command requires 'tracking_branch'.";
          }
          if (normalizedSliceBranch && !isValidRefToken(normalizedSliceBranch)) {
            return `ERROR: Invalid slice_branch: ${slice_branch}.`;
          }
          if (normalizedTrackingBranch && !isValidRefToken(normalizedTrackingBranch)) {
            return `ERROR: Invalid tracking_branch: ${tracking_branch}.`;
          }
          if (normalizedSliceBranch) {
            cmdParts.push("--slice-branch", normalizedSliceBranch);
          }
          if (normalizedTrackingBranch) {
            cmdParts.push("--tracking-branch", normalizedTrackingBranch);
          }
          cmdParts.push("--json");
          if (recover_missing_worktree) {
            cmdParts.push("--recover-missing-worktree");
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "abort": {
          cmdParts.push("abort");
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "continue": {
          cmdParts.push("continue");
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "reset": {
          cmdParts.push("reset");
          if (!skipRequiredChecks && (!ref || !ref.trim())) {
            return "ERROR: 'reset' command requires 'ref'.";
          }
          const normalizedResetRef = normalizeRef(ref);
          if (normalizedResetRef && !isValidRefToken(normalizedResetRef)) {
            return `ERROR: Invalid ref: ${ref}.`;
          }
          if (normalizedResetRef) {
            cmdParts.push("--ref", normalizedResetRef);
          }
          if (hard) {
            cmdParts.push("--hard");
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
        if (command === "accumulate") {
          return result.trim();
        }
        return `Git Command: ${command} (help)\n\n${result}`;
      } catch (error: any) {
        const selected = selectDiagnostic(
          error?.stderr ? error.stderr.toString() : "",
          error?.stdout ? error.stdout.toString() : "",
          typeof error?.message === "string" ? error.message : "",
        );
        return selected.message
          ? `Git Command Failed: ${command}\n${selected.message}`
          : `Git Command Failed: ${command}`;
      }
    }

    const validationMessage = appendCommandParts();
    if (validationMessage?.startsWith("ERROR")) {
      return validationMessage;
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      if (command === "accumulate") {
        return result.trim();
      }
      return `Git Command: ${command}\n\n${result}`;
    } catch (error: any) {
      const selected = selectDiagnostic(
        error?.stderr ? error.stderr.toString() : "",
        error?.stdout ? error.stdout.toString() : "",
        typeof error?.message === "string" ? error.message : "",
      );
      return selected.message
        ? `Git Command Failed: ${command}\n${selected.message}`
        : `Git Command Failed: ${command}`;
    }
  },
});
