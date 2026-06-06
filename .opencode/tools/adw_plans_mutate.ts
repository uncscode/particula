import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

import {
  buildCommandFailureError,
  redactPathLikeText,
  sanitizeSuccessOutput,
  stripDefaultArgs,
  validateRequiredArgs,
} from "./adw_plans_contract_shared";

// --- Inlined from adw_plans.ts ---

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

const COMMANDS = ["list", "show", "create", "update", "add-phase", "update-phase", "validate", "schema", "scaffold-sections", "list-sections"] as const;

const MUTATING_COMMANDS = new Set<string>([
  "create",
  "update",
  "add-phase",
  "update-phase",
  "scaffold-sections",
]);

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
const PLAN_PRIORITIES = ["P0", "P1", "P2", "P3", "Backlog"] as const;
const PLAN_SIZES = ["XS", "S", "M", "L", "XL", "XXL"] as const;
const PHASE_STATUSES = ["Not Started", "In Progress", "Blocked", "Shipped", "Cancelled"] as const;
const MAX_PATCH_PAYLOAD_BYTES = 65_536;

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

function sanitizedChildEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key === "VIRTUAL_ENV") {
      continue;
    }
    if (value !== undefined) {
      env[key] = value;
    }
  }
  return env;
}

function getTimeout(command: PlanCommand): number {
  return COMMAND_TIMEOUTS[command] ?? DEFAULT_TIMEOUT_MS;
}

type IdentifierOptions = {
  field: string;
  required: boolean;
  emptyMessage: string;
};

const COMMAND_GUIDE = `Commands:
• create — Required: plan_type, title; Optional: plan_id, parent, priority, size, status
• update — Required: plan_id; Optional: status, priority, size, title, patch
• add-phase — Required: plan_id, title; Optional: size, phase_status, after
• update-phase — Required: plan_id, phase_id; Optional: phase_status, title, size, issue_number, clear_issue_number, patch
• scaffold-sections — Required: plan_id, plan_type. Copies section templates for a plan.`;

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
} as const;

// --- Mutate command gate ---

const MUTATE_COMMANDS = ["create", "update", "add-phase", "update-phase", "scaffold-sections"] as const;

const MUTATE_COMMAND_GUIDE = `Supported mutate commands:
• create
• update
• add-phase
• update-phase
• scaffold-sections

Use adw_plans_read for read-only commands (list/show/validate/schema/list-sections).`;

function isMutateCommand(command: unknown): command is (typeof MUTATE_COMMANDS)[number] {
  return typeof command === "string" && MUTATE_COMMANDS.includes(command as (typeof MUTATE_COMMANDS)[number]);
}

// --- Inlined execute logic from adw_plans.ts (mutating commands only) ---

async function executeAdwPlansMutate(args: Record<string, any>): Promise<string> {
  const preflightError = validateRequiredArgs(args, REQUIRED_ARGS, buildError);
  if (preflightError) {
    return preflightError;
  }
  const normalized = stripDefaultArgs(args);
  const {
    command,
    plan_id,
    plan_type,
    parent,
    status,
    title,
    priority,
    size,
    phase_id,
    phase_status,
    after,
    issue_number,
    clear_issue_number,
    patch,
    cwd,
  } = normalized;

  const lazyNormalizePatch = (): { value?: string; error?: string } => normalizePatch(patch);

  const cmdParts = ["uv", "run", "adw", "plans"];
  const executedCommand = command as PlanCommand;

  switch (executedCommand) {
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
      if (priority) {
        cmdParts.push("--priority", priority);
      }
      if (size) {
        cmdParts.push("--size", size);
      }
      if (status) {
        cmdParts.push("--status", status);
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
      if (status) {
        cmdParts.push("--status", status);
      }
      if (priority) {
        cmdParts.push("--priority", priority);
      }
      if (size) {
        cmdParts.push("--size", size);
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
      if (size) {
        cmdParts.push("--size", size);
      }
      if (phase_status) {
        cmdParts.push("--status", phase_status);
      }
      if (after !== undefined && after !== null) {
        const normalizedAfter = normalizeIdentifier(after, {
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
      cmdParts.push("update-phase", normalizedPlanId.value!, normalizedPhaseId.value!);
      if (phase_status) {
        cmdParts.push("--status", phase_status);
      }
      if (title !== undefined && title !== null) {
        const normalizedTitle = String(title).trim();
        if (normalizedTitle) {
          cmdParts.push("--title", normalizedTitle);
        }
      }
      if (size) {
        cmdParts.push("--size", size);
      }
      if (issue_number !== undefined && issue_number !== null) {
        if (
          typeof issue_number !== "number"
          || !Number.isSafeInteger(issue_number)
          || issue_number < 1
        ) {
          return buildError("'issue_number' must be a positive safe integer when provided.");
        }
        cmdParts.push("--issue", String(issue_number));
      }
      if (clear_issue_number) {
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

    default:
      return buildError(
        `Unsupported command "${command}". Valid commands: ${COMMANDS.join(", ")}.`,
      );
  }

  const normalizedCwd = normalizeCwd(cwd);
  if (normalizedCwd.error) {
    return normalizedCwd.error;
  }
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
      env: sanitizedChildEnv(),
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
}

// --- Tool definition ---

export default tool({
  description: `Mutation wrapper for adw plans metadata/documents.

Supported commands: create, update, add-phase, update-phase, scaffold-sections.
Read commands are rejected; use adw_plans_read for read-only operations.

Contract parity note: command execution and envelopes are delegated to adw_plans.`,

  args: {
    command: tool.schema.enum(["create", "update", "add-phase", "update-phase", "scaffold-sections", "list", "show", "validate", "schema", "list-sections"]),
    plan_id: tool.schema.string().optional(),
    plan_type: tool.schema.string().optional(),
    title: tool.schema.string().optional(),
    parent: tool.schema.string().optional(),
    status: tool.schema.enum(["Draft", "Proposed", "Ready", "In Progress", "Blocked", "Monitoring", "Shipped", "Cancelled", "Superseded"]).optional(),
    priority: tool.schema.enum(["P0", "P1", "P2", "P3", "Backlog"]).optional(),
    size: tool.schema.enum(["XS", "S", "M", "L", "XL", "XXL"]).optional(),
    phase_id: tool.schema.string().optional(),
    phase_status: tool.schema.enum(["Not Started", "In Progress", "Blocked", "Shipped", "Cancelled"]).optional(),
    after: tool.schema.string().optional(),
    issue_number: tool.schema.number().optional(),
    clear_issue_number: tool.schema.boolean().optional(),
    patch: tool.schema.string().optional(),
    cwd: tool.schema.string().optional(),
  },

  async execute(args) {
    if (!isMutateCommand(args.command)) {
      return `ERROR: Unsupported command for adw_plans_mutate: ${String(args.command)}\n\n${MUTATE_COMMAND_GUIDE}`;
    }
    return executeAdwPlansMutate(args as Record<string, any>);
  },
});
