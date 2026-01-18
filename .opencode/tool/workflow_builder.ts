/**
 * Workflow Builder Tool for OpenCode Integration
 * 
 * Provides access to WorkflowBuilderTool for creating and validating ADW workflow JSON files.
 * Enables interactive workflow creation with incremental validation.
 * 
 * See https://opencode.ai/docs/custom-tools/ for OpenCode tool development patterns.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: `Create and validate ADW workflow JSON files with the WorkflowBuilderTool.

AVAILABLE COMMANDS:
• create: Create new workflow file
  Usage: { command: "create", workflow_name: "my-workflow", description: "...", workflow_type: "custom" }

• add_step: Add validated step to workflow
  Usage: { command: "add_step", workflow_name: "my-workflow", step_json: '{"type":"agent",...}' }

• remove_step: Remove step from workflow
  Usage: { command: "remove_step", workflow_name: "my-workflow", step_index: 0 }
  OR: { command: "remove_step", workflow_name: "my-workflow", step_name: "Plan" }

• get: Retrieve workflow details
  Usage: { command: "get", workflow_name: "my-workflow" }

• list: List all available workflows
  Usage: { command: "list" }

• update: Update entire workflow with validated JSON
  Usage: { command: "update", workflow_name: "my-workflow", workflow_json: '{"name":...}' }

• validate: Validate workflow JSON without saving
  Usage: { command: "validate", workflow_json: '{"name":...}' }

WORKFLOW TYPES:
• complete - Full validation workflow (test, review, docs)
• patch - Quick fix workflow (build and ship only)
• custom - User-defined steps

OUTPUT MODES:
• summary - Human-readable summary (default)
• full - Complete details including full JSON
• json - Structured JSON output`,

  args: {
    command: tool.schema
      .enum(["create", "add_step", "remove_step", "get", "list", "update", "validate"])
      .describe(`Command to execute.

COMMANDS:
• create - Create new workflow file (requires: workflow_name, description)
• add_step - Add step to workflow (requires: workflow_name, step_json)
• remove_step - Remove step (requires: workflow_name, step_index OR step_name)
• get - Get workflow details (requires: workflow_name)
• list - List all workflows (no arguments required)
• update - Update workflow (requires: workflow_name, workflow_json)
• validate - Validate JSON (requires: workflow_json)`),

    workflow_name: tool.schema
      .string()
      .optional()
      .describe(`Workflow name (used as filename without .json extension).

REQUIRED FOR: create, add_step, remove_step, get, update
NOT REQUIRED FOR: list, validate

EXAMPLES: "quick-fix", "feature-deploy", "custom-test"`),

    description: tool.schema
      .string()
      .optional()
      .describe(`Short workflow description (required for 'create' command).

Should be concise and descriptive.
EXAMPLE: "Quick fix workflow for minor changes"`),

    version: tool.schema
      .string()
      .optional()
      .describe(`Semantic version for workflow (default: "1.0.0").

Used only for 'create' command.
EXAMPLE: "1.2.0"`),

    workflow_type: tool.schema
      .enum(["complete", "patch", "custom"])
      .optional()
      .describe(`Workflow type (default: "custom").

TYPES:
• complete - Full validation: plan → build → test → review → document → ship
• patch - Quick fix: plan → build → ship (skips validation)
• custom - User-defined steps

Used only for 'create' command.`),

    step_json: tool.schema
      .string()
      .optional()
      .describe(`JSON string of step to add (required for 'add_step' command).

Must be valid JSON conforming to workflow step schema.

EXAMPLE (using slash command):
{
  "type": "agent",
  "name": "Implement",
  "command": "/implement",
  "prompt": "Implement from spec_content",
  "model": "base"
}

EXAMPLE (using agent directly):
{
  "type": "agent",
  "name": "Run Tests",
  "agent": "tester",
  "prompt": "run",
  "model": "base"
}

MODEL TIERS:
• light - Lightweight tasks (linting, simple tests)
• base - Standard tasks (implementation, review) [DEFAULT]
• heavy - Complex tasks (architecture, debugging)

NOTE: Must provide either 'command' OR 'agent', not both.`),

    step_index: tool.schema
      .number()
      .optional()
      .describe(`Zero-based index of step to remove (for 'remove_step' command).

Use this OR step_name, not both.
EXAMPLE: step_index: 0 (removes first step)`),

    step_name: tool.schema
      .string()
      .optional()
      .describe(`Name of step to remove (for 'remove_step' command).

Use this OR step_index, not both.
EXAMPLE: step_name: "Plan"`),

    position: tool.schema
      .number()
      .optional()
      .describe(`Position to insert step (for 'add_step' command).

If not specified, step is appended to end.
EXAMPLE: position: 1 (insert at index 1)`),

    workflow_json: tool.schema
      .string()
      .optional()
      .describe(`Complete workflow JSON string (for 'update' or 'validate' commands).

Must be valid JSON conforming to workflow schema.

EXAMPLE:
{
  "name": "my-workflow",
  "version": "1.0.0",
  "description": "Custom workflow",
  "workflow_type": "custom",
  "steps": [...]
}`),

    output: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe(`Output format (default: "summary").

MODES:
• summary - Human-readable summary with key details
• full - Complete output including full workflow JSON
• json - Structured JSON for programmatic use`),
  },

  async execute(args) {
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

    // Build Python command
    const cmdParts = [
      "python3",
      ".opencode/tool/workflow_builder.py",
      command,
    ];

    // Add arguments based on command
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
      // Execute the Python workflow builder tool
      const result = await Bun.$`${cmdParts}`.text();
      
      // Return result directly - errors are included in output
      return result;
      
    } catch (error: any) {
      // Handle execution errors
      const errorOutput = error.stdout ? error.stdout.toString() : "";
      const errorMsg = error.stderr ? error.stderr.toString() : error.message;
      
      // Return error information for LLM to see
      if (errorOutput) {
        return `Workflow Builder Error:\n${errorOutput}\n\nStderr:\n${errorMsg}`;
      }
      
      return `Workflow Builder Execution Error:\n${errorMsg}`;
    }
  },
});
