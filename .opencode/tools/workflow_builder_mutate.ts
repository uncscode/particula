import { tool } from "@opencode-ai/plugin";

// --- Inlined from workflow_builder.ts ---

const MAX_DIAGNOSTIC_CHARS = 2000;

function sanitizeDiagnostic(value: unknown): string {
  const text = typeof value === "string" ? value : value == null ? "" : String(value);
  const withoutAnsi = text.replace(/\x1b\[[0-9;]*m/g, "");
  const withoutNul = withoutAnsi.replace(/\u0000/g, "");
  if (withoutNul.length <= MAX_DIAGNOSTIC_CHARS) {
    return withoutNul;
  }
  return `${withoutNul.slice(0, MAX_DIAGNOSTIC_CHARS)}... [truncated]`;
}

async function executeWorkflowBuilder(args: Record<string, any>): Promise<string> {
  const {
    command,
    workflow_name,
    description,
    version = "1.0.0",
    workflow_type = "custom",
    step_json,
    step_index,
    step_name,
    position,
    workflow_json,
    output = "summary",
  } = args;

  const cmdParts = [
    "python3",
    ".opencode/tools/workflow_builder.py",
    command,
  ];

  if (workflow_name) {
    cmdParts.push("--workflow-name", workflow_name);
  }

  if (description) {
    cmdParts.push("--description", description);
  }

  if (version && version !== "1.0.0") {
    cmdParts.push("--version", version);
  }

  if (workflow_type && workflow_type !== "custom") {
    cmdParts.push("--workflow-type", workflow_type);
  }

  if (step_json) {
    cmdParts.push("--step-json", step_json);
  }

  if (step_index !== undefined) {
    cmdParts.push("--step-index", step_index.toString());
  }

  if (step_name) {
    cmdParts.push("--step-name", step_name);
  }

  if (position !== undefined) {
    cmdParts.push("--position", position.toString());
  }

  if (workflow_json) {
    cmdParts.push("--workflow-json", workflow_json);
  }

  if (output) {
    cmdParts.push("--output", output);
  }

  try {
    const result = await Bun.$`${cmdParts}`.text();
    return result;
  } catch (error: any) {
    const errorOutput = sanitizeDiagnostic(error?.stdout?.toString?.() ?? error?.stdout);
    const errorStderr = sanitizeDiagnostic(error?.stderr?.toString?.() ?? error?.stderr);
    const errorMsg = sanitizeDiagnostic(error?.message);

    if (errorOutput) {
      const detail = errorStderr || errorMsg || "No additional diagnostics provided.";
      return `Workflow Builder Error:\n${errorOutput}\n\nStderr:\n${detail}`;
    }

    return `Workflow Builder Execution Error:\n${errorStderr || errorMsg || "Unknown execution failure"}`;
  }
}

// --- Mutate command gate ---

const MUTATE_COMMANDS = ["create", "add_step", "remove_step", "update"] as const;
const MUTATE_COMMAND_SET = new Set<string>(MUTATE_COMMANDS);

function isMutateCommand(
  command: unknown,
): command is (typeof MUTATE_COMMANDS)[number] {
  return (
    typeof command === "string" &&
    MUTATE_COMMAND_SET.has(command)
  );
}

function normalizeCommand(command: unknown): string {
  if (typeof command !== "string") {
    return String(command);
  }
  const trimmed = command.trim();
  return trimmed.length > 0 ? trimmed : "";
}

// --- Tool definition ---

export default tool({
  description: `Mutation wrapper for workflow_builder operations.

Supported commands: create, add_step, remove_step, update.
Read-only commands are rejected; use workflow_builder_read for list/get/validate.

Contract parity note: successful and failed command output envelopes are delegated to workflow_builder.`,

  args: {
    command: tool.schema.string().optional(),
    workflow_name: tool.schema.string().optional(),
    description: tool.schema.string().optional(),
    version: tool.schema.string().optional(),
    workflow_type: tool.schema.enum(["complete", "patch", "custom"]).optional(),
    step_json: tool.schema.string().optional(),
    step_index: tool.schema.number().optional(),
    step_name: tool.schema.string().optional(),
    position: tool.schema.number().optional(),
    workflow_json: tool.schema.string().optional(),
    output: tool.schema.enum(["summary", "full", "json"]).optional(),
  },

  async execute(args) {
    const normalizedCommand = normalizeCommand(args.command);
    if (!isMutateCommand(normalizedCommand)) {
      return `ERROR: workflow_builder_mutate does not support command '${normalizedCommand}'. Use: create, add_step, remove_step, update.`;
    }
    return executeWorkflowBuilder({ ...args, command: normalizedCommand });
  },
});
