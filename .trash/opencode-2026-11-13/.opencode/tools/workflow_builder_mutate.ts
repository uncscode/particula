import { tool } from "@opencode-ai/plugin";

import {
  executeWorkflowBuilder,
  normalizeCommand as normalizeSharedCommand,
} from "./workflow_builder_shared";

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
    const normalizedCommand = normalizeSharedCommand(args.command);
    if (!isMutateCommand(normalizedCommand)) {
      return `ERROR: workflow_builder_mutate does not support command '${normalizedCommand}'. Use: create, add_step, remove_step, update.`;
    }
    return executeWorkflowBuilder({ ...args, command: normalizedCommand });
  },
});
