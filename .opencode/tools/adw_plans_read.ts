import { tool } from "@opencode-ai/plugin";

import {
  buildCommandFailureError,
  hasMeaningfulSplitWrapperAliasValue,
  mergeParsedOptionField,
  parseCommandOptionsString,
  sanitizeSuccessOutput,
  stripDefaultArgs,
  validateAndNormalizePlansCwdPath,
  validateRequiredArgs,
} from "./adw_plans_contract_shared";

// --- Inlined from adw_plans.ts ---

const COMMANDS = ["list", "show", "validate", "schema", "list-sections"] as const;

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
  validate: LONG_COMMAND_TIMEOUT_MS,
  schema: LONG_COMMAND_TIMEOUT_MS,
  "list-sections": DEFAULT_TIMEOUT_MS,
};

function sanitizedChildEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
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
• list — Filters: type, lifecycle, parent; Optional options: json status=<value>
• show — Required: plan_id; Optional options: json
• validate — No additional parameters
• schema — Optional options: check
• list-sections — Required: plan_id. Lists section files with repo-relative paths. Optional options: json, populate.`;

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
  if (hasMeaningfulSplitWrapperAliasValue(args.status)) {
    return buildError("'status' is not accepted as a direct field in adw_plans_read; use options: \"status=<value>\".");
  }
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
    options,
    cwd,
  } = normalized;

  const parsedOptions = parseCommandOptionsString(String(command ?? ""), options, buildError);
  if (parsedOptions.error) {
    return parsedOptions.error;
  }
  const optionValues = parsedOptions.values ?? {};
  const resolvedStatus = mergeParsedOptionField(undefined, optionValues.status, "status", buildError);
  if (resolvedStatus.error) return resolvedStatus.error;
  const resolvedJson = mergeParsedOptionField(undefined, optionValues.json, "json", buildError);
  if (resolvedJson.error) return resolvedJson.error;
  const resolvedCheck = mergeParsedOptionField(undefined, optionValues.check, "check", buildError);
  if (resolvedCheck.error) return resolvedCheck.error;
  const resolvedPopulate = mergeParsedOptionField(
    undefined,
    optionValues.populate,
    "populate",
    buildError,
  );
  if (resolvedPopulate.error) return resolvedPopulate.error;

  const cmdParts = ["uv", "run", "--active", "adw", "plans"];
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

  const normalizedCwd = validateAndNormalizePlansCwdPath(cwd);
  if (normalizedCwd.error) {
    return normalizedCwd.error;
  }
  if (normalizedCwd.value) {
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
    command: tool.schema.enum(["list", "show", "validate", "schema", "list-sections"]),
    plan_id: tool.schema.string().optional(),
    plan_type: tool.schema.string().optional(),
    lifecycle: tool.schema.enum(["active", "completed", "closed"]).optional(),
    parent: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    cwd: tool.schema.string().optional(),
  },

  async execute(args) {
    if (!isReadCommand(args.command)) {
      return `ERROR: Unsupported command for adw_plans_read: ${String(args.command)}\n\n${READ_COMMAND_GUIDE}`;
    }
    return executeAdwPlansReadOnly(args as Record<string, any>);
  },
});
