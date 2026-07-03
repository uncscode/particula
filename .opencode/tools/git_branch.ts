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

const isProtectedBranchRef = (value?: string): boolean => {
  const normalized = normalizeRef(value).toLowerCase();
  return normalized === "main" || normalized === "master";
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

const OPTIONAL_KEYS = ["branch", "create", "source", "worktree_path", "help"];

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
  description: `Move branch pointers with a narrow ADW git wrapper (checkout, push, push-force-with-lease).

SIMPLE EXAMPLES (copy these patterns):

Checkout branch:  { command: "checkout", branch: "feature-123" }
Create from src:  { command: "checkout", branch: "feature-123", create: true, source: "origin/develop" }
Push branch:      { command: "push", branch: "feature-123" }
Force-with-lease: { command: "push-force-with-lease", branch: "feature-123" }

RULES:
- Supports only branch commands: checkout, push, push-force-with-lease.
- push/checkout/push-force-with-lease require branch unless help: true.
- checkout source requires create: true.
- checkout --create and push-force-with-lease block protected branches (main/master).
- Set help: true to view CLI help for any command.`,

  args: {
    command: tool.schema.enum(["checkout", "push", "push-force-with-lease"]),
    branch: tool.schema.string().optional(),
    create: tool.schema.boolean().optional(),
    source: tool.schema.string().optional(),
    worktree_path: tool.schema.string().optional(),
    help: tool.schema.boolean().optional(),
  },

  async execute(rawArgs) {
    const rawSourceProvided = Object.prototype.hasOwnProperty.call(rawArgs, "source");
    const { normalized } = normalizeSparseOptions(rawArgs, {
      optionalKeys: OPTIONAL_KEYS,
      falseOverrideKeys: [],
    });

    const { command, branch, create, source, worktree_path, help } = normalized as {
      command: "checkout" | "push" | "push-force-with-lease";
      branch?: string;
      create?: boolean;
      source?: string;
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
        case "push": {
          cmdParts.push("push");
          const normalizedBranch = normalizeRef(branch);
          if (!skipRequiredChecks && !normalizedBranch) {
            return "ERROR: 'push' command requires 'branch'.";
          }
          if (normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "checkout": {
          cmdParts.push("checkout");
          const normalizedBranch = normalizeRef(branch);
          const normalizedSource = normalizeRef(source);

          if (!skipRequiredChecks && !normalizedBranch) {
            return "ERROR: 'checkout' command requires 'branch'.";
          }
          if (normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (!skipRequiredChecks && rawSourceProvided && !normalizedSource) {
            return "ERROR: 'checkout' command requires non-empty 'source' when provided.";
          }
          if (!skipRequiredChecks && normalizedSource && create !== true) {
            return "ERROR: 'checkout' command requires 'create' when 'source' is provided.";
          }
          if (create === true && isProtectedBranchRef(normalizedBranch)) {
            return "ERROR: checkout --create to protected branch is blocked.";
          }
          if (normalizedSource && !isValidRefToken(normalizedSource)) {
            return `ERROR: Invalid source: ${source}.`;
          }

          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (normalizedSource) {
            cmdParts.push("--source", normalizedSource);
          }
          if (create) {
            cmdParts.push("--create");
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }

        case "push-force-with-lease": {
          cmdParts.push("push-force-with-lease");
          const normalizedBranch = normalizeRef(branch);
          if (!skipRequiredChecks && !normalizedBranch) {
            return "ERROR: 'push-force-with-lease' command requires 'branch'.";
          }
          if (normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (isProtectedBranchRef(normalizedBranch)) {
            return "ERROR: push-force-with-lease to protected branch is blocked.";
          }
          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (normalizedWorktree.value) {
            cmdParts.push("--worktree-path", normalizedWorktree.value);
          }
          return undefined;
        }
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
