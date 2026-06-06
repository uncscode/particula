import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";

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
• list — Filters: type, lifecycle, parent, status, json
• show — Required: plan_id; Optional: json
• validate — No additional parameters
• schema — Optional: check
• list-sections — Required: plan_id. Lists section files with repo-relative paths. Optional: json, populate.`;

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
  return undefined;
}

const REQUIRED_ARGS = {
  show: [{ field: "plan_id", message: "show command requires 'plan_id'." }],
  "list-sections": [{ field: "plan_id", message: "list-sections command requires 'plan_id'." }],
} as const;

// --- Read-only command gate ---

const READ_COMMANDS = ["list", "show", "validate", "schema", "list-sections"] as const;

const READ_COMMAND_GUIDE = `Supported read commands:
• list
• show
• validate
• schema
• list-sections

Use adw_plans_mutate for mutating commands (create/update/add-phase/update-phase/scaffold-sections).`;

function isReadCommand(command: unknown): command is (typeof READ_COMMANDS)[number] {
  return typeof command === "string" && READ_COMMANDS.includes(command as (typeof READ_COMMANDS)[number]);
}

// --- Inlined execute logic from adw_plans.ts (read-only commands only) ---

async function executeAdwPlansReadOnly(args: Record<string, any>): Promise<string> {
  const preflightError = validateRequiredArgs(args, REQUIRED_ARGS, buildError);
  if (preflightError) {
    return preflightError;
  }
  const normalized = stripDefaultArgs(args);
  const {
    command,
    plan_id,
    plan_type,
    lifecycle,
    parent,
    status,
    json,
    check,
    populate,
    cwd,
  } = normalized;

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
      if (status) {
        cmdParts.push("--status", status);
      }
      if (json) {
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
      if (json) {
        cmdParts.push("--json");
      }
      break;
    }

    case "validate": {
      cmdParts.push("validate");
      break;
    }

    case "schema": {
      cmdParts.push("schema");
      if (check) {
        cmdParts.push("--check");
      }
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
      if (json) {
        cmdParts.push("--json");
      }
      if (populate) {
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
  description: `Read-only wrapper for adw plans metadata/documents.

Supported commands: list, show, validate, schema, list-sections.
Mutating commands are rejected; use adw_plans_mutate for writes.

Contract parity note: successful and failed command output envelopes are delegated to adw_plans.`,

  args: {
    command: tool.schema.enum(["list", "show", "validate", "schema", "list-sections", "create", "update", "add-phase", "update-phase", "scaffold-sections"]),
    plan_id: tool.schema.string().optional(),
    plan_type: tool.schema.string().optional(),
    lifecycle: tool.schema.enum(["active", "completed", "closed"]).optional(),
    parent: tool.schema.string().optional(),
    status: tool.schema.enum(["Draft", "Proposed", "Ready", "In Progress", "Blocked", "Monitoring", "Shipped", "Cancelled", "Superseded"]).optional(),
    json: tool.schema.boolean().optional(),
    check: tool.schema.boolean().optional(),
    populate: tool.schema.boolean().optional(),
    cwd: tool.schema.string().optional(),
  },

  async execute(args) {
    if (!isReadCommand(args.command)) {
      return `ERROR: Unsupported command for adw_plans_read: ${String(args.command)}\n\n${READ_COMMAND_GUIDE}`;
    }
    return executeAdwPlansReadOnly(args as Record<string, any>);
  },
});
