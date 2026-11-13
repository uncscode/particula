const MAX_DIAGNOSTIC_CHARS = 2000;

const WORKFLOW_BUILDER_COMMANDS = [
  "create",
  "add_step",
  "remove_step",
  "get",
  "list",
  "update",
  "validate",
] as const;

type WorkflowBuilderCommand = (typeof WORKFLOW_BUILDER_COMMANDS)[number];

type WorkflowBuilderArgs = Record<string, unknown>;

function sanitizeDiagnostic(value: unknown): string {
  const text = typeof value === "string" ? value : value == null ? "" : String(value);
  const withoutAnsi = text.replace(/\x1b\[[0-9;]*m/g, "");
  const withoutNul = withoutAnsi.replace(/\u0000/g, "");
  if (withoutNul.length <= MAX_DIAGNOSTIC_CHARS) {
    return withoutNul;
  }
  return `${withoutNul.slice(0, MAX_DIAGNOSTIC_CHARS)}... [truncated]`;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function normalizeCommand(command: unknown): string {
  return normalizeOptionalString(command) ?? "";
}

export function isWorkflowBuilderCommand(command: unknown): command is WorkflowBuilderCommand {
  return typeof command === "string" && WORKFLOW_BUILDER_COMMANDS.includes(command as WorkflowBuilderCommand);
}

export function unsupportedCommandError(command: string): string {
  return `ERROR: workflow_builder does not support command '${command}'. Use: ${WORKFLOW_BUILDER_COMMANDS.join(", ")}.`;
}

function getRequiredArgumentError(command: WorkflowBuilderCommand, args: WorkflowBuilderArgs): string | null {
  const workflowName = normalizeOptionalString(args.workflow_name);
  const description = normalizeOptionalString(args.description);
  const stepJson = normalizeOptionalString(args.step_json);
  const stepName = normalizeOptionalString(args.step_name);
  const workflowJson = normalizeOptionalString(args.workflow_json);
  const stepIndex = typeof args.step_index === "number" ? args.step_index : undefined;

  switch (command) {
    case "create":
      return workflowName && description
        ? null
        : "ERROR: 'create' requires workflow_name and description";
    case "add_step":
      return workflowName && stepJson
        ? null
        : "ERROR: 'add_step' requires workflow_name and step_json";
    case "remove_step":
      if (!workflowName) {
        return "ERROR: 'remove_step' requires workflow_name";
      }
      return stepIndex !== undefined || stepName
        ? null
        : "ERROR: 'remove_step' requires either step_index or step_name";
    case "get":
      return workflowName ? null : "ERROR: 'get' requires workflow_name";
    case "update":
      return workflowName && workflowJson
        ? null
        : "ERROR: 'update' requires workflow_name and workflow_json";
    case "validate":
      return workflowJson ? null : "ERROR: 'validate' requires workflow_json";
    case "list":
      return null;
  }
}

function buildCommandParts(command: WorkflowBuilderCommand, args: WorkflowBuilderArgs): string[] {
  const workflowName = normalizeOptionalString(args.workflow_name);
  const description = normalizeOptionalString(args.description);
  const version = normalizeOptionalString(args.version);
  const workflowType = normalizeOptionalString(args.workflow_type);
  const stepJson = normalizeOptionalString(args.step_json);
  const stepName = normalizeOptionalString(args.step_name);
  const workflowJson = normalizeOptionalString(args.workflow_json);
  const output = normalizeOptionalString(args.output);
  const stepIndex = typeof args.step_index === "number" ? args.step_index : undefined;
  const position = typeof args.position === "number" ? args.position : undefined;

  const cmdParts = ["python3", ".opencode/tools/workflow_builder.py", command];

  if (workflowName) {
    cmdParts.push("--workflow-name", workflowName);
  }
  if (description) {
    cmdParts.push("--description", description);
  }
  if (version && version !== "1.0.0") {
    cmdParts.push("--version", version);
  }
  if (workflowType && workflowType !== "custom") {
    cmdParts.push("--workflow-type", workflowType);
  }
  if (stepJson) {
    cmdParts.push("--step-json", stepJson);
  }
  if (stepIndex !== undefined) {
    cmdParts.push("--step-index", stepIndex.toString());
  }
  if (stepName) {
    cmdParts.push("--step-name", stepName);
  }
  if (position !== undefined) {
    cmdParts.push("--position", position.toString());
  }
  if (workflowJson) {
    cmdParts.push("--workflow-json", workflowJson);
  }
  if (output) {
    cmdParts.push("--output", output);
  }

  return cmdParts;
}

export async function executeWorkflowBuilder(args: WorkflowBuilderArgs): Promise<string> {
  const command = normalizeCommand(args.command);
  if (!isWorkflowBuilderCommand(command)) {
    return unsupportedCommandError(command);
  }

  const requiredArgumentError = getRequiredArgumentError(command, args);
  if (requiredArgumentError) {
    return requiredArgumentError;
  }

  const cmdParts = buildCommandParts(command, args);

  try {
    return await Bun.$`${cmdParts}`.text();
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
