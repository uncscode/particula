import { tool } from "@opencode-ai/plugin";

import {
  executeWorkflowBuilder,
  normalizeCommand as normalizeSharedCommand,
} from "./workflow_builder_shared";

// --- Read command gate ---

const READ_COMMANDS = ["list", "get", "validate"] as const;
const READ_COMMAND_SET = new Set<string>(READ_COMMANDS);

function isReadCommand(command: unknown): command is (typeof READ_COMMANDS)[number] {
  return typeof command === "string" && READ_COMMAND_SET.has(command);
}

// --- Tool definition ---

export default tool({
  description: `Read-only wrapper for workflow_builder operations.

Supported commands: list, get, validate.
Mutating commands are rejected; use workflow_builder_mutate for writes.

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
    const normalizedCommand = normalizeSharedCommand(args.command);
    if (!isReadCommand(normalizedCommand)) {
      return `ERROR: workflow_builder_read does not support command '${normalizedCommand}'. Use: list, get, validate.`;
    }
    return executeWorkflowBuilder({ ...args, command: normalizedCommand });
  },
});
