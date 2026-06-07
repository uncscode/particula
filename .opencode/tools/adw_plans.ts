import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

import {
  buildCommandFailureError,
  mergeParsedOptionField,
  parseCommandOptionsString,
  redactPathLikeText,
  sanitizeSuccessOutput,
  stripDefaultArgs,
  validateUpdatePhaseIssueLinkArgs,
  validateRequiredArgs,
} from "./adw_plans_contract_shared";

// --- Inlined from lib/cpp_lint_wrapper_shared.ts (isStatDirectory only) ---

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

// --- Inlined from lib/env_utils ---

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key === "VIRTUAL_ENV" || value === undefined) continue;
    env[key] = value;
  }
  return env;
}

// --- Tool implementation ---

const COMMANDS = ["list", "show", "create", "update", "add-phase", "update-phase", "validate", "schema", "scaffold-sections", "list-sections"] as const;
// Commands that mutate persisted plan artifacts under `plans/` and
// `plans/sections/`. For these, `cwd` is required to prevent silent writes
// to the main repository root from within agent-driven worktree runs.
// Regression guard for the leak where post-execution hooks (and agent prompt
// drift) wrote plan mutations to the primary repo instead of the worktree.
const MUTATING_COMMANDS = new Set<string>([
  "create",
  "update",
  "add-phase",
  "update-phase",
  "scaffold-sections",
]);
const LIFECYCLES = ["active", "completed", "closed"] as const;
const PLAN_STATUSES = [
  "Draft",
  "Proposed",
  "Ready",
  "In Progress",
  "Blocked",
  "Monitoring",
  "Shipped",
  "Cancelled",
  "Superseded",
] as const;
const PHASE_STATUSES = ["Not Started", "In Progress", "Blocked", "Shipped", "Cancelled"] as const;
const MAX_PATCH_PAYLOAD_BYTES = 65_536;

/**
 * Strip LLM-supplied default/inert values from args so downstream logic
 * sees `undefined` for anything the caller didn't meaningfully set.
 *
 * Models frequently send every optional parameter with its schema default
 * (empty string `""`, `false`, `0`, `null`) even when the tool description
 * says "only include parameters you need". This normalizer converts those
 * inert values to `undefined` *before* the command switch, preventing:
 *   - empty strings reaching `normalizeIdentifier` for identifiers/titles
 *   - `false` booleans being forwarded as explicit flags
 *   - `0` numbers being forwarded as explicit values
 *   - `null` values surviving into downstream checks
 *
 * Only `command` is preserved unconditionally (it's always required).
 */
const COMMAND_GUIDE = `Commands:
• list — Filters: type, lifecycle, parent, status; Optional options: json
• show — Required: plan_id; Optional options: json
• create — Required: plan_type, title; Optional: plan_id, parent, status; Optional options: status=<value> priority=<value> size=<value>
• update — Required: plan_id; Optional: status, title, patch; Optional options: status=<value> priority=<value> size=<value>
• add-phase — Required: plan_id, title; Optional: phase_status; Optional options: phase-status=<value> size=<value> after=<phase_id>
• update-phase — Required: plan_id, phase_id; Optional: phase_status, title, patch; Optional options: phase-status=<value> size=<value> issue=<n> clear-issue-number
• validate — No additional parameters
• schema — Optional options: check
• scaffold-sections — Required: plan_id, plan_type. Copies section templates for a plan.
• list-sections — Required: plan_id. Lists section files with repo-relative paths. Optional options: json populate.`;

type PlanCommand = (typeof COMMANDS)[number];

const decoder = new TextDecoder();
const DEFAULT_TIMEOUT_MS = 60_000;
const LONG_COMMAND_TIMEOUT_MS = 180_000;

const COMMAND_TIMEOUTS: Record<PlanCommand, number> = {
  list: DEFAULT_TIMEOUT_MS,
  show: DEFAULT_TIMEOUT_MS,
  create: DEFAULT_TIMEOUT_MS,
  update: DEFAULT_TIMEOUT_MS,
  "add-phase": DEFAULT_TIMEOUT_MS,
  "update-phase": DEFAULT_TIMEOUT_MS,
  validate: LONG_COMMAND_TIMEOUT_MS,
  schema: LONG_COMMAND_TIMEOUT_MS,
  "scaffold-sections": DEFAULT_TIMEOUT_MS,
  "list-sections": DEFAULT_TIMEOUT_MS,
};

// VIRTUAL_ENV stripping is now handled by shared sanitizedEnv() from env_utils.

function getTimeout(command: PlanCommand): number {
  return COMMAND_TIMEOUTS[command] ?? DEFAULT_TIMEOUT_MS;
}

type IdentifierOptions = {
  field: string;
  required: boolean;
  emptyMessage: string;
};

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${COMMAND_GUIDE}`;
}

function normalizeIdentifier(
  raw: unknown,
  options: IdentifierOptions,
): { value?: string; error?: string } {
  const { field, required, emptyMessage } = options;
  if (raw === undefined || raw === null) {
    if (required) {
      return { error: buildError(emptyMessage) };
    }
    return {};
  }

  const normalized = String(raw).trim();
  if (!normalized) {
    if (!required) {
      return {};
    }
    return { error: buildError(emptyMessage) };
  }
  if (normalized.startsWith("-")) {
    return {
      error: buildError(`'${field}' must not start with '-' to avoid CLI option confusion.`),
    };
  }
  return { value: normalized };
}

function normalizeCwd(raw: unknown): { value?: string; error?: string } {
  const normalized = normalizeIdentifier(raw, {
    field: "cwd",
    required: false,
    emptyMessage: "'cwd' must not be empty when provided.",
  });
  if (normalized.error || !normalized.value) {
    return normalized;
  }
  return normalized;
}

function findCurrentRepositoryRoot(): string {
  let currentPath = realpathSync(process.cwd());
  while (true) {
    if (existsSync(path.join(currentPath, ".git"))) {
      return currentPath;
    }
    const parentPath = path.dirname(currentPath);
    if (parentPath === currentPath) {
      return realpathSync(process.cwd());
    }
    currentPath = parentPath;
  }
}

function validateCwdPath(cwdPath: string): string | undefined {
  const redactedPath = redactPathLikeText(cwdPath);
  if (!existsSync(cwdPath)) {
    return `ERROR: cwd path does not exist: ${redactedPath}`;
  }
  let stats;
  try {
    stats = statSync(cwdPath);
  } catch {
    return `ERROR: cwd path does not exist: ${redactedPath}`;
  }
  if (!isStatDirectory(stats)) {
    return `ERROR: cwd path is not a directory: ${redactedPath}`;
  }

  const canonical = realpathSync(cwdPath);
  const gitMetadataPath = `${canonical}/.git`;
  if (!existsSync(gitMetadataPath)) {
    return `ERROR: cwd path is not a repository/worktree root: ${redactedPath} (missing .git metadata at ${redactPathLikeText(gitMetadataPath)})`;
  }
  const repoRoot = findCurrentRepositoryRoot();
  if (canonical !== repoRoot) {
    return `ERROR: cwd path resolves outside repository root: ${redactedPath} (canonical: ${redactPathLikeText(canonical)})`;
  }
  return undefined;
}

function normalizePatch(raw: unknown): { value?: string; error?: string } {
  const normalized = normalizeIdentifier(raw, {
    field: "patch",
    required: false,
    emptyMessage: "'patch' must not be empty when provided.",
  });
  if (normalized.error || !normalized.value) {
    return normalized;
  }

  const patchSize = new TextEncoder().encode(normalized.value).length;
  if (patchSize > MAX_PATCH_PAYLOAD_BYTES) {
    return {
      error: buildError(
        `'patch' exceeds maximum size (${MAX_PATCH_PAYLOAD_BYTES} bytes UTF-8); received ${patchSize} bytes.`,
      ),
    };
  }
  return { value: normalized.value };
}

const REQUIRED_ARGS = {
  show: [{ field: "plan_id", message: "show command requires 'plan_id'." }],
  create: [
    { field: "plan_type", message: "create command requires 'plan_type'." },
    { field: "title", message: "create command requires 'title'." },
  ],
  update: [{ field: "plan_id", message: "update command requires 'plan_id'." }],
  "add-phase": [
    { field: "plan_id", message: "add-phase command requires 'plan_id'." },
    { field: "title", message: "add-phase command requires 'title'." },
  ],
  "update-phase": [
    { field: "plan_id", message: "update-phase command requires 'plan_id'." },
    { field: "phase_id", message: "update-phase command requires 'phase_id'." },
  ],
  "scaffold-sections": [
    { field: "plan_id", message: "scaffold-sections command requires 'plan_id'." },
    { field: "plan_type", message: "scaffold-sections command requires 'plan_type'." },
  ],
  "list-sections": [{ field: "plan_id", message: "list-sections command requires 'plan_id'." }],
} as const;

export default tool({
  description: `Manage structured plan metadata and documents via the \`adw plans\` CLI. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

  List plans:     { command: "list", lifecycle: "active", options: "json" }
  Show plan:      { command: "show", plan_id: "E17-F1", options: "json" }
  Create plan:    { command: "create", plan_type: "feature", title: "Add Auth", parent: "E17", cwd: "./trees/abc" }
  Update status:  { command: "update", plan_id: "E17-F8", status: "Ready", cwd: "./trees/abc" }
  Add phase:      { command: "add-phase", plan_id: "E17-F1", title: "Core impl", cwd: "./trees/abc" }
  Ship phase:     { command: "update-phase", plan_id: "E17-F8", phase_id: "E17-F8-P1", phase_status: "Shipped", cwd: "./trees/abc" }
  List sections:  { command: "list-sections", plan_id: "M25", cwd: "./trees/abc" }
  Validate:       { command: "validate" }

RULES:
- Mutating commands (create/update/add-phase/update-phase/scaffold-sections) REQUIRE cwd.
- Read-only commands (list/show/validate/schema/list-sections) accept optional cwd.
- Returns deterministic ERROR envelopes on CLI failures.

See .opencode/tools/adw_plans.md for full parameter reference and advanced usage.`,

  args: {
    command: tool.schema
      .enum([...COMMANDS])
      .describe(`Which \`adw plans\` subcommand to execute.`),

    plan_id: tool.schema
      .string()
      .optional()
      .describe("Plan identifier (e.g., E17-F1). Required for show/update when targeting a specific plan."),

    plan_type: tool.schema
      .string()
      .optional()
      .describe("Plan type filter or creation target. Runtime acceptance is registry-driven (for example: epic, feature, maintenance, research)."),

    lifecycle: tool.schema
      .enum([...LIFECYCLES])
      .optional()
      .describe("Lifecycle filter for list command (active, completed, closed)."),

    parent: tool.schema
      .string()
      .optional()
      .describe("Optional parent filter/id for list/create commands."),

    status: tool.schema
      .enum([...PLAN_STATUSES])
      .optional()
      .describe("Plan status filter or update value. Matches Click choices in adw_plans CLI."),

    title: tool.schema
      .string()
      .optional()
      .describe("Plan title. Required when creating a plan; optional for update command."),

    phase_id: tool.schema
      .string()
      .optional()
      .describe("Phase identifier (e.g., E17-F1-P1). Required for update-phase command."),

    phase_status: tool.schema
      .enum([...PHASE_STATUSES])
      .optional()
      .describe("Phase status for add-phase/update-phase (Not Started, In Progress, Blocked, Shipped, Cancelled)."),

    options: tool.schema
      .string()
      .optional()
      .describe(
        "Command-scoped bounded options string. Supported tokens include json, populate, check, status=<value>, phase-status=<value>, priority=<value>, size=<value>, after=<id>, issue=<n>, and clear-issue-number when allowlisted for the selected command.",
      ),

    patch: tool.schema
      .string()
      .optional()
      .describe("Raw JSON patch payload forwarded as '--patch <json>' for update/update-phase."),

    cwd: tool.schema
      .string()
      .optional()
      .describe("Optional repository/worktree root. Forwarded only as '--cwd <path>'."),
  },

  async execute(args) {
    const preflightError = validateRequiredArgs(
      args as Record<string, any>,
      REQUIRED_ARGS,
      buildError,
    );
    if (preflightError) {
      return preflightError;
    }
    // Normalize away optional LLM-supplied inert values ("", false, 0, null)
    // only after required-arg preflight checks run.
    const normalized = stripDefaultArgs(args as Record<string, any>);
    const {
      command,
      plan_id,
      plan_type,
      lifecycle,
      parent,
      status,
      title,
      phase_id,
      phase_status,
      options,
      patch,
      cwd,
    } = normalized;

    const parsedOptions = parseCommandOptionsString(String(command ?? ""), options, buildError);
    if (parsedOptions.error) {
      return parsedOptions.error;
    }
    const optionValues = parsedOptions.values ?? {};

    const resolvedJson = mergeParsedOptionField(undefined, optionValues.json, "json", buildError);
    if (resolvedJson.error) return resolvedJson.error;
    const resolvedStatus = mergeParsedOptionField(status, optionValues.status, "status", buildError);
    if (resolvedStatus.error) return resolvedStatus.error;
    const resolvedPriority = mergeParsedOptionField(undefined, optionValues.priority, "priority", buildError);
    if (resolvedPriority.error) return resolvedPriority.error;
    const resolvedSize = mergeParsedOptionField(undefined, optionValues.size, "size", buildError);
    if (resolvedSize.error) return resolvedSize.error;
    const resolvedCheck = mergeParsedOptionField(undefined, optionValues.check, "check", buildError);
    if (resolvedCheck.error) return resolvedCheck.error;
    const resolvedPopulate = mergeParsedOptionField(undefined, optionValues.populate, "populate", buildError);
    if (resolvedPopulate.error) return resolvedPopulate.error;
    const resolvedPhaseStatus = mergeParsedOptionField(
      phase_status,
      optionValues.phase_status,
      "phase_status",
      buildError,
    );
    if (resolvedPhaseStatus.error) return resolvedPhaseStatus.error;
    const resolvedAfter = mergeParsedOptionField(undefined, optionValues.after, "after", buildError);
    if (resolvedAfter.error) return resolvedAfter.error;
    const resolvedIssueNumber = mergeParsedOptionField(
      undefined,
      optionValues.issue_number,
      "issue_number",
      buildError,
    );
    if (resolvedIssueNumber.error) return resolvedIssueNumber.error;
    const resolvedClearIssueNumber = mergeParsedOptionField(
      undefined,
      optionValues.clear_issue_number,
      "clear_issue_number",
      buildError,
    );
    if (resolvedClearIssueNumber.error) return resolvedClearIssueNumber.error;

    // Defer patch normalization to command cases that use it (update,
    // update-phase) so that an empty/whitespace patch supplied alongside
    // commands like create or list does not trigger a spurious error.
    const lazyNormalizePatch = (): { value?: string; error?: string } => normalizePatch(patch);

    const cmdParts = ["uv", "run", "adw", "plans"];
    const executedCommand = command as PlanCommand;

    switch (executedCommand) {
      case "list": {
        cmdParts.push("list");
        if (plan_type) {
          cmdParts.push("--type", plan_type);
        }
        if (lifecycle) {
          cmdParts.push("--lifecycle", lifecycle);
        }
        if (parent !== undefined && parent !== null) {
          const normalizedParent = normalizeIdentifier(parent, {
            field: "parent",
            required: false,
            emptyMessage: "'parent' must not be empty when provided.",
          });
          if (normalizedParent.error) {
            return normalizedParent.error;
          }
          if (normalizedParent.value) {
            cmdParts.push("--parent", normalizedParent.value);
          }
        }
        if (resolvedStatus.value) {
          cmdParts.push("--status", resolvedStatus.value);
        }
        if (resolvedJson.value) {
          cmdParts.push("--json");
        }
        break;
      }

      case "show": {
        const normalizedPlanId = normalizeIdentifier(plan_id, {
          field: "plan_id",
          required: true,
          emptyMessage: "show command requires 'plan_id'.",
        });
        if (normalizedPlanId.error) {
          return normalizedPlanId.error;
        }
        cmdParts.push("show", normalizedPlanId.value!);
        if (resolvedJson.value) {
          cmdParts.push("--json");
        }
        break;
      }

      case "create": {
        if (!plan_type) {
          return buildError("create command requires 'plan_type'.");
        }
        const normalizedTitle = typeof title === "string" ? title.trim() : "";
        if (!normalizedTitle) {
          return buildError("create command requires 'title'.");
        }
        cmdParts.push("create", "--type", plan_type, "--title", normalizedTitle);

        if (plan_id !== undefined && plan_id !== null) {
          const normalizedPlanId = normalizeIdentifier(plan_id, {
            field: "plan_id",
            required: false,
            emptyMessage: "'plan_id' must not be empty when provided.",
          });
          if (normalizedPlanId.error) {
            return normalizedPlanId.error;
          }
          if (normalizedPlanId.value) {
            cmdParts.push("--id", normalizedPlanId.value);
          }
        }
        if (parent !== undefined && parent !== null) {
          const normalizedParent = normalizeIdentifier(parent, {
            field: "parent",
            required: false,
            emptyMessage: "'parent' must not be empty when provided.",
          });
          if (normalizedParent.error) {
            return normalizedParent.error;
          }
          if (normalizedParent.value) {
            cmdParts.push("--parent", normalizedParent.value);
          }
        }
        if (resolvedPriority.value) {
          cmdParts.push("--priority", resolvedPriority.value);
        }
        if (resolvedSize.value) {
          cmdParts.push("--size", resolvedSize.value);
        }
        if (resolvedStatus.value) {
          cmdParts.push("--status", resolvedStatus.value);
        }
        break;
      }

      case "update": {
        const normalizedPlanId = normalizeIdentifier(plan_id, {
          field: "plan_id",
          required: true,
          emptyMessage: "update command requires 'plan_id'.",
        });
        if (normalizedPlanId.error) {
          return normalizedPlanId.error;
        }
        cmdParts.push("update", normalizedPlanId.value!);
        if (resolvedStatus.value) {
          cmdParts.push("--status", resolvedStatus.value);
        }
        if (resolvedPriority.value) {
          cmdParts.push("--priority", resolvedPriority.value);
        }
        if (resolvedSize.value) {
          cmdParts.push("--size", resolvedSize.value);
        }
        if (title !== undefined && title !== null) {
          const normalizedTitle = String(title).trim();
          if (normalizedTitle) {
            cmdParts.push("--title", normalizedTitle);
          }
        }
        {
          const normalizedPatch = lazyNormalizePatch();
          if (normalizedPatch.error) {
            return normalizedPatch.error;
          }
          if (normalizedPatch.value) {
            cmdParts.push("--patch", normalizedPatch.value);
          }
        }
        break;
      }

      case "add-phase": {
        const normalizedPlanId = normalizeIdentifier(plan_id, {
          field: "plan_id",
          required: true,
          emptyMessage: "add-phase command requires 'plan_id'.",
        });
        if (normalizedPlanId.error) {
          return normalizedPlanId.error;
        }
        const normalizedTitle = typeof title === "string" ? title.trim() : "";
        if (!normalizedTitle) {
          return buildError("add-phase command requires 'title'.");
        }
        cmdParts.push("add-phase", normalizedPlanId.value!, "--title", normalizedTitle);
        if (resolvedSize.value) {
          cmdParts.push("--size", resolvedSize.value);
        }
        if (resolvedPhaseStatus.value) {
          cmdParts.push("--status", resolvedPhaseStatus.value);
        }
        if (resolvedAfter.value !== undefined && resolvedAfter.value !== null) {
          const normalizedAfter = normalizeIdentifier(resolvedAfter.value, {
            field: "after",
            required: false,
            emptyMessage: "'after' must not be empty when provided.",
          });
          if (normalizedAfter.error) {
            return normalizedAfter.error;
          }
          if (normalizedAfter.value) {
            cmdParts.push("--after", normalizedAfter.value);
          }
        }
        break;
      }

      case "update-phase": {
        const normalizedPlanId = normalizeIdentifier(plan_id, {
          field: "plan_id",
          required: true,
          emptyMessage: "update-phase command requires 'plan_id'.",
        });
        if (normalizedPlanId.error) {
          return normalizedPlanId.error;
        }
        const normalizedPhaseId = normalizeIdentifier(phase_id, {
          field: "phase_id",
          required: true,
          emptyMessage: "update-phase command requires 'phase_id'.",
        });
        if (normalizedPhaseId.error) {
          return normalizedPhaseId.error;
        }
        const issueLinkValidationError = validateUpdatePhaseIssueLinkArgs(
          resolvedIssueNumber.value,
          resolvedClearIssueNumber.value,
          buildError,
        );
        if (issueLinkValidationError) {
          return issueLinkValidationError;
        }
        cmdParts.push("update-phase", normalizedPlanId.value!, normalizedPhaseId.value!);
        if (resolvedPhaseStatus.value) {
          cmdParts.push("--status", resolvedPhaseStatus.value);
        }
        if (title !== undefined && title !== null) {
          const normalizedTitle = String(title).trim();
          if (normalizedTitle) {
            cmdParts.push("--title", normalizedTitle);
          }
        }
        if (resolvedSize.value) {
          cmdParts.push("--size", resolvedSize.value);
        }
        if (resolvedIssueNumber.value !== undefined && resolvedIssueNumber.value !== null) {
          if (
            typeof resolvedIssueNumber.value !== "number"
            || !Number.isSafeInteger(resolvedIssueNumber.value)
            || resolvedIssueNumber.value <= 0
          ) {
            return buildError("'issue_number' must be a positive safe integer when provided.");
          }
          cmdParts.push("--issue", String(resolvedIssueNumber.value));
        }
        if (resolvedClearIssueNumber.value) {
          cmdParts.push("--clear-issue-number");
        }
        {
          const normalizedPatch = lazyNormalizePatch();
          if (normalizedPatch.error) {
            return normalizedPatch.error;
          }
          if (normalizedPatch.value) {
            cmdParts.push("--patch", normalizedPatch.value);
          }
        }
        break;
      }

      case "validate": {
        cmdParts.push("validate");
        break;
      }

      case "schema": {
        cmdParts.push("schema");
        if (resolvedCheck.value) {
          cmdParts.push("--check");
        }
        break;
      }

      case "scaffold-sections": {
        const normalizedPlanId = normalizeIdentifier(plan_id, {
          field: "plan_id",
          required: true,
          emptyMessage: "scaffold-sections command requires 'plan_id'.",
        });
        if (normalizedPlanId.error) {
          return normalizedPlanId.error;
        }
        if (!plan_type) {
          return buildError("scaffold-sections command requires 'plan_type'.");
        }
        cmdParts.push("scaffold-sections", normalizedPlanId.value!, "--type", plan_type);
        break;
      }

      case "list-sections": {
        const normalizedPlanId = normalizeIdentifier(plan_id, {
          field: "plan_id",
          required: true,
          emptyMessage: "list-sections command requires 'plan_id'.",
        });
        if (normalizedPlanId.error) {
          return normalizedPlanId.error;
        }
        cmdParts.push("list-sections", normalizedPlanId.value!);
        if (resolvedJson.value) {
          cmdParts.push("--json");
        }
        if (resolvedPopulate.value) {
          cmdParts.push("--populate");
        }
        break;
      }

      default:
        return buildError(
          `Unsupported command "${command}". Valid commands: ${COMMANDS.join(", ")}.`,
        );
    }

    const normalizedCwd = normalizeCwd(cwd);
    if (normalizedCwd.error) {
      return normalizedCwd.error;
    }
    // Enforce worktree-scoped writes: mutating commands must supply an
    // explicit cwd so PlanRepository is constructed against the worktree
    // root rather than the main repo root resolved via get_project_root().
    // Treats empty/whitespace cwd (already rejected above) and omitted cwd
    // identically at this layer.
    if (MUTATING_COMMANDS.has(executedCommand) && !normalizedCwd.value) {
      return buildError(`${executedCommand} command requires 'cwd'.`);
    }
    if (normalizedCwd.value) {
      const cwdValidationError = validateCwdPath(normalizedCwd.value);
      if (cwdValidationError) {
        return cwdValidationError;
      }
      cmdParts.push("--cwd", normalizedCwd.value);
    }

    try {
      const commandTimeout = getTimeout(executedCommand);
      const result = Bun.spawnSync({
        cmd: cmdParts,
        stdout: "pipe",
        stderr: "pipe",
        timeout: commandTimeout,
        env: sanitizedEnv(),
      });
      const stdoutRaw = result.stdout ? decoder.decode(result.stdout) : "";
      const stderrRaw = result.stderr ? decoder.decode(result.stderr) : "";
      const safeStdout = sanitizeSuccessOutput(stdoutRaw);
      const timedOut = Boolean((result as { timedOut?: boolean }).timedOut);
      if (timedOut) {
        return buildCommandFailureError(
          executedCommand,
          "timeout",
          { stderr: stderrRaw, stdout: stdoutRaw },
          `Command timed out after ${commandTimeout}ms`,
        );
      }
      if (result.exitCode !== 0) {
        return buildCommandFailureError(
          executedCommand,
          `exit ${result.exitCode}`,
          { stderr: stderrRaw, stdout: stdoutRaw },
          `Exit code ${result.exitCode}`,
        );
      }

      if (!safeStdout.hasVisibleContent) {
        return `ADW Plans Command: ${executedCommand}\n\nadw plans ${executedCommand} completed with no output.`;
      }
      return `ADW Plans Command: ${executedCommand}\n\n${safeStdout.text}`;
    } catch (error: any) {
      const stderr =
        error?.stderr instanceof Uint8Array
          ? decoder.decode(error.stderr)
          : typeof error?.stderr === "string"
            ? error.stderr
            : "";
      const stdout =
        error?.stdout instanceof Uint8Array
          ? decoder.decode(error.stdout)
          : typeof error?.stdout === "string"
            ? error.stdout
          : "";
      const message = error?.message ? String(error.message) : "";
      return buildCommandFailureError(
        executedCommand,
        "execution error",
        { stderr, stdout, message },
        "Unknown execution error",
      );
    }
  },
});
