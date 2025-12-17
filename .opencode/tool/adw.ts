/**
 * Dynamic ADW Tool for OpenCode Integration
 *
 * Provides unified interface to all ADW CLI commands directly from OpenCode.
 * This tool enables seamless execution of ADW workflows without switching to terminal.
 *
 * See https://opencode.ai/docs/custom-tools/ for OpenCode tool development patterns.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: `Execute ADW (AI Developer Workflow) CLI commands for automated development workflows.

COMMON WORKFLOWS:
• complete: Full workflow with validation (plan→build→test→review→document→ship)
  Usage: { command: "complete", issue_number: 123, model: "base" }

• patch: Quick patch workflow (plan→build→ship, skips tests/review)
  Usage: { command: "patch", issue_number: 456 }

WORKFLOW PHASES (for granular control):
• plan: Create implementation plan
• build: Execute implementation
• test: Run validation tests
• review: Code review and quality checks
• document: Generate documentation
• ship: Push changes and create PR

ISSUE MANAGEMENT:
• create-issue: Create new GitHub issue
  Usage: { command: "create-issue", title: "Add feature", body: "Description..." }

• interpret-issue: Convert text to structured issue
  Usage: { command: "interpret-issue", text: "Add tests for auth module" }

SETUP & TEMPLATE COMMANDS:
• setup: Environment and template management (requires args for subcommands)
  Usage: { command: "setup", args: ["template", "extract", "--diff"] }
  Usage: { command: "setup", args: ["template", "validate"] }
  Usage: { command: "setup", args: ["env"] }

SYSTEM COMMANDS:
• status: Show active workflows
• health: System health check
• init: Initialize ADW config

HELP: Set help: true to see command-specific usage
  Usage: { command: "complete", help: true }`,
  args: {
    command: tool.schema
      .enum([
        "complete", "patch", "plan", "build", "test", "review", "document",
        "ship", "status", "health", "init", "create-issue", "interpret-issue",
        "maintenance", "launch", "stop", "docstring", "finalize-docs", "setup"
      ])
      .describe(`ADW command to execute. Use help: true to see detailed usage.

WORKFLOW COMMANDS (require issue_number):
• complete - Full workflow: plan→build→test→review→document→ship
• patch - Quick workflow: plan→build→ship (skips validation)
• plan - Generate implementation plan only
• build - Execute implementation from plan
• test - Run tests and validation
• review - Code review and quality checks
• document - Generate documentation
• ship - Push changes and create pull request

ISSUE COMMANDS:
• create-issue - Create new GitHub issue (requires: title, body)
• interpret-issue - Convert text to structured issue (requires: text OR issue_number)

SETUP COMMANDS (use args for subcommands):
• setup - Environment and template management
  Subcommands via args: ["env"], ["validate"], ["check"],
    ["template", "init"], ["template", "apply"], ["template", "extract"],
    ["template", "validate"], ["template", "token", "list"],
    ["template", "token", "add", "<NAME>", "--default", "<val>", "--description", "<desc>"],
    ["template", "token", "remove", "<NAME>"]

SYSTEM COMMANDS:
• status - Show active ADW workflows
• health - Run system health diagnostics
• init - Initialize ADW configuration
• maintenance - Run maintenance tasks
• docstring - Update docstrings for files
• finalize-docs - Finalize living documentation`),
    
    issue_number: tool.schema
      .number()
      .optional()
      .describe(`GitHub issue number for workflow commands.

REQUIRED FOR: complete, patch, plan, build, test, review, document, ship
OPTIONAL FOR: interpret-issue (use with --source-issue flag)
EXAMPLE: issue_number: 123`),
    
    adw_id: tool.schema
      .string()
      .optional()
      .describe(`ADW workflow ID to resume existing workflow (8-character hex string).

If not provided, a new ADW ID is automatically generated.
Use this to continue interrupted workflows or run specific phases.

EXAMPLE: adw_id: "a1b2c3d4"
GET CURRENT: Use 'status' command to see active ADW IDs`),
    
    model: tool.schema
      .enum(["base", "heavy"])
      .optional()
      .describe(`Model set for AI operations. Defaults to 'base'.

• base - Uses Sonnet models (faster, cost-effective, recommended for most tasks)
• heavy - Uses Opus models (more capable, use for complex features or debugging)

WHEN TO USE HEAVY:
- Complex architectural changes
- Difficult bug investigations  
- Large refactoring tasks
- When base model struggles

EXAMPLE: model: "heavy"`),
    
    args: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe(`Additional CLI arguments passed directly to ADW command.

GENERAL EXAMPLES:
• ["--dry-run"] - Preview without execution
• ["--verbose"] - Detailed output
• ["--help"] - Show command help

SETUP COMMAND EXAMPLES (args are subcommands + flags):
• ["env"] - Run environment wizard
• ["validate"] - Validate environment configuration
• ["check"] - Run preflight checks
• ["template", "init"] - Initialize template manifest
• ["template", "init", "--yes"] - Initialize with defaults (no prompts)
• ["template", "apply"] - Apply templates to project
• ["template", "apply", "--dry-run"] - Preview template application
• ["template", "apply", "--check"] - Check placeholders without writing
• ["template", "extract", "--diff"] - Show drift between live docs and templates
• ["template", "extract", "--dry-run"] - Preview extraction changes
• ["template", "extract", "--yes"] - Extract without confirmation
• ["template", "validate"] - Validate placeholders against manifest
• ["template", "validate", "--format", "json"] - JSON output for validation
• ["template", "token", "list"] - List all keyword tokens
• ["template", "token", "add", "TOKEN_NAME", "--default", "value", "--description", "desc"]
• ["template", "token", "add", "TOKEN_NAME", "--default", "new", "--description", "d", "--force"]
• ["template", "token", "remove", "TOKEN_NAME", "--yes"]

Can be used for any flags not covered by specific parameters.`),
    
    help: tool.schema
      .boolean()
      .optional()
      .describe(`Show detailed help for the specified command. Skips execution and validation.

When true, displays:
- Command usage and syntax
- Required and optional arguments
- Examples and workflows

EXAMPLE: { command: "complete", help: true }
OUTPUT: Shows 'adw complete --help' information`),
    
    text: tool.schema
      .string()
      .optional()
      .describe(`Text input for interpret-issue command. Human-friendly description of work to be done.

The system will:
1. Analyze the text
2. Determine issue type (patch/feature/multistep)
3. Generate structured GitHub issue with proper formatting
4. Add appropriate labels

EXAMPLE: text: "Add comprehensive tests for the authentication module"
RESULT: Creates properly formatted GitHub issue with workflow:blocked label`),
    
    title: tool.schema
      .string()
      .optional()
      .describe(`Issue title for create-issue command (required with body).

Should be concise, descriptive, and follow repository conventions.

EXAMPLES:
• "Add user authentication feature"
• "Fix IndexError in data parser"
• "Refactor database connection logic"

Used with: command: "create-issue", title: "...", body: "..."`),
    
    body: tool.schema
      .string()
      .optional()
      .describe(`Issue body for create-issue command (required with title).

Should include:
- Problem description or feature requirements
- Acceptance criteria (use checklists: - [ ] item)
- Technical context if relevant
- Links to related issues

MARKDOWN SUPPORTED: Use formatting, code blocks, lists, etc.

EXAMPLE:
body: "## Description\\nImplement user auth\\n\\n## Acceptance Criteria\\n- [ ] Login endpoint\\n- [ ] JWT tokens"`),
  },
  async execute(args) {
    const { command, issue_number, adw_id, model, args: additionalArgs, text, title, body, help } = args;

    // Command validation and argument requirements
    const workflowCommands = [
      "complete", "patch", "plan", "build", "test", "review", "document", "ship"
    ];
    const systemCommands = ["status", "health", "init", "maintenance", "launch", "stop"];
    const issueCommands = ["create-issue", "interpret-issue"];
    const docCommands = ["docstring", "finalize-docs"];
    const setupCommands = ["setup"];

    // Validate required arguments based on command type (skip if help flag is set)
    if (!help) {
      if (workflowCommands.includes(command)) {
        if (!issue_number) {
          return `ERROR: Command '${command}' requires 'issue_number' argument.\n\nUsage: adw ${command} <issue_number> [--adw-id <id>] [--model <base|heavy>]`;
        }
      }

      if (command === "create-issue") {
        if (!title || !body) {
          return `ERROR: Command 'create-issue' requires both 'title' and 'body' arguments.\n\nUsage: adw create-issue --title "Issue Title" --body "Issue description"`;
        }
      }

      if (command === "interpret-issue") {
        if (!text && !issue_number) {
          return `ERROR: Command 'interpret-issue' requires either 'text' argument or 'issue_number' argument.\n\nUsage: adw interpret-issue --text "Description" OR adw interpret-issue --source-issue <number>`;
        }
      }

      // Setup command requires args for subcommands (unless showing help)
      if (command === "setup" && (!additionalArgs || additionalArgs.length === 0)) {
        return `ERROR: Command 'setup' requires 'args' for subcommands.

USAGE EXAMPLES:
• Environment wizard: { command: "setup", args: ["env"] }
• Validate config: { command: "setup", args: ["validate"] }
• Preflight check: { command: "setup", args: ["check"] }

TEMPLATE SUBCOMMANDS:
• Initialize manifest: { command: "setup", args: ["template", "init"] }
• Apply templates: { command: "setup", args: ["template", "apply"] }
• Check for drift: { command: "setup", args: ["template", "extract", "--diff"] }
• Extract to templates: { command: "setup", args: ["template", "extract", "--yes"] }
• Validate placeholders: { command: "setup", args: ["template", "validate"] }
• List tokens: { command: "setup", args: ["template", "token", "list"] }
• Add token: { command: "setup", args: ["template", "token", "add", "NAME", "--default", "val", "--description", "desc"] }
• Remove token: { command: "setup", args: ["template", "token", "remove", "NAME", "--yes"] }

Use { command: "setup", help: true } to see CLI help.`;
      }
    }

    // Build command arguments
    const cmdParts = ["uv", "run", "adw", command];

    // For setup command, args are subcommands and must come immediately after "setup"
    // e.g., "adw setup template extract --diff" not "adw setup --diff template extract"
    if (command === "setup") {
      if (help) {
        cmdParts.push("--help");
      } else if (additionalArgs && additionalArgs.length > 0) {
        cmdParts.push(...additionalArgs);
      }
      // Setup command doesn't use other parameters like issue_number, model, etc.
    } else {
      // Add --help flag if requested
      if (help) {
        cmdParts.push("--help");
      }

      // Add issue number for workflow commands (skip if help flag is set)
      if (workflowCommands.includes(command) && issue_number && !help) {
        cmdParts.push(issue_number.toString());
      }

      // Add optional arguments
      if (adw_id) {
        cmdParts.push("--adw-id", adw_id);
      }

      if (model) {
        cmdParts.push("--model", model);
      }

      // Handle command-specific arguments
      if (command === "create-issue") {
        if (title) {
          cmdParts.push("--title", title);
        }
        if (body) {
          cmdParts.push("--body", body);
        }
      }

      if (command === "interpret-issue") {
        if (text) {
          cmdParts.push("--text", text);
        } else if (issue_number) {
          cmdParts.push("--source-issue", issue_number.toString());
        }
      }

      if (command === "docstring" && issue_number) {
        cmdParts.push(issue_number.toString());
      }

      if (command === "finalize-docs" && issue_number) {
        cmdParts.push(issue_number.toString());
      }

      // Add any additional arguments (for non-setup commands, these go at the end)
      if (additionalArgs && additionalArgs.length > 0) {
        cmdParts.push(...additionalArgs);
      }
    }

    try {
      // Execute the ADW CLI command using Bun's shell API
      const result = await Bun.$`${cmdParts}`.text();
      
      // Check for common error patterns
      if (result.includes("ERROR:") || result.includes("Error:")) {
        return `ADW Command Failed:\n${result}`;
      }

      // Success - return the output
      return `ADW Command: ${command}\n\n${result}`;
      
    } catch (error: any) {
      // Handle execution errors
      const errorOutput = error.stdout ? error.stdout.toString() : "";
      const errorMsg = error.stderr ? error.stderr.toString() : error.message;
      
      // Try to extract meaningful error information
      if (errorOutput && errorOutput.includes("ERROR")) {
        return `ADW Command Failed:\n${errorOutput}`;
      }
      
      if (errorMsg) {
        return `ADW Execution Error:\n${errorMsg}${errorOutput ? `\n\nOutput:\n${errorOutput}` : ""}`;
      }
      
      return `ADW Command Failed: ${command}\nError: ${error.message}`;
    }
  },
});