import { tool } from "@opencode-ai/plugin";

import {
  buildCommandFailureError,
  hasMeaningfulSplitWrapperAliasValue,
  mergeParsedOptionField,
  parseCommandOptionsString,
  sanitizeSuccessOutput,
  stripDefaultArgs,
  validateAndNormalizePlansCwdPath,
  validateUpdatePhaseIssueLinkArgs,
  validateRequiredArgs,
} from "./adw_plans_contract_shared";

// --- Inlined from adw_plans.ts ---

const COMMANDS = ["create", "update", "add-phase", "update-phase", "scaffold-sections"] as const;

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
const PHASE_STATUSES = ["Not Started", "In Progress", "Blocked", "Shipped", "Cancelled"] as const;
const MAX_PATCH_PAYLOAD_BYTES = 65_536;

type PlanCommand = (typeof COMMANDS)[number];

const decoder = new TextDecoder();
const DEFAULT_TIMEOUT_MS = 60_000;
const LONG_COMMAND_TIMEOUT_MS = 180_000;

const COMMAND_TIMEOUTS: Record<PlanCommand, number> = {
  create: DEFAULT_TIMEOUT_MS,
  update: DEFAULT_TIMEOUT_MS,
  "add-phase": DEFAULT_TIMEOUT_MS,
  "update-phase": DEFAULT_TIMEOUT_MS,
  "scaffold-sections": DEFAULT_TIMEOUT_MS,
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
• create — Required: plan_type, title; Optional: plan_id, parent; Optional options: status=<value> priority=<value> size=<value>
• update — Required: plan_id; Optional: title, patch; Optional options: status=<value> priority=<value> size=<value>
• add-phase — Required: plan_id, title; Optional options: phase-status=<value> size=<value> after=<phase_id>
• update-phase — Required: plan_id, phase_id; Optional: title, patch; Optional options: phase-status=<value> size=<value> issue=<n> clear-issue-number
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
  if (hasMeaningfulSplitWrapperAliasValue(args.status)) {
    return buildError("'status' is not accepted as a direct field in adw_plans_mutate; use options: \"status=<value>\".");
  }
  if (hasMeaningfulSplitWrapperAliasValue(args.phase_status)) {
    return buildError("'phase_status' is not accepted as a direct field in adw_plans_mutate; use options: \"phase-status=<value>\".");
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
    parent,
    title,
    phase_id,
    options,
    patch,
    cwd,
  } = normalized;

  const parsedOptions = parseCommandOptionsString(String(command ?? ""), options, buildError);
  if (parsedOptions.error) {
    return parsedOptions.error;
  }
  const optionValues = parsedOptions.values ?? {};
  const resolvedStatus = mergeParsedOptionField(undefined, optionValues.status, "status", buildError);
  if (resolvedStatus.error) return resolvedStatus.error;
  const resolvedPriority = mergeParsedOptionField(undefined, optionValues.priority, "priority", buildError);
  if (resolvedPriority.error) return resolvedPriority.error;
  const resolvedSize = mergeParsedOptionField(undefined, optionValues.size, "size", buildError);
  if (resolvedSize.error) return resolvedSize.error;
  const resolvedPhaseStatus = mergeParsedOptionField(
    undefined,
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

  const lazyNormalizePatch = (): { value?: string; error?: string } => normalizePatch(patch);

  const cmdParts = ["uv", "run", "--active", "adw", "plans"];
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
          || resolvedIssueNumber.value < 1
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

  const normalizedCwd = validateAndNormalizePlansCwdPath(cwd);
  if (normalizedCwd.error) {
    return normalizedCwd.error;
  }
  if (MUTATING_COMMANDS.has(executedCommand) && !normalizedCwd.value) {
    return buildError(`${executedCommand} command requires 'cwd'.`);
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
  description: `Mutation wrapper for adw plans metadata/documents.

Supported commands: create, update, add-phase, update-phase, scaffold-sections.
Read commands are rejected; use adw_plans_read for read-only operations.

Contract parity note: command execution and envelopes are delegated to adw_plans.`,

  args: {
    command: tool.schema.enum(["create", "update", "add-phase", "update-phase", "scaffold-sections"]),
    plan_id: tool.schema.string().optional(),
    plan_type: tool.schema.string().optional(),
    title: tool.schema.string().optional(),
    parent: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    phase_id: tool.schema.string().optional(),
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
