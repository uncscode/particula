/**
 * Dynamic ADW Tool for OpenCode Integration
 *
 * Provides unified interface to all ADW CLI commands directly from OpenCode.
 * This tool enables seamless execution of ADW workflows without switching to terminal.
 *
 * See https://opencode.ai/docs/custom-tools/ for OpenCode tool development patterns.
 */

import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/env_utils ---

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (key === "VIRTUAL_ENV" || value === undefined) continue;
    env[key] = value;
  }
  return env;
}

// --- Tool implementation ---

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;

function normalizeAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

function adwIdValidationMessage(): string {
  return "'adw_id' must be an 8-character hex string (e.g., \"abc12345\").";
}

const decoder = new TextDecoder();
const ERROR_SNIPPET_LIMIT = 2000;

// Conservative timeouts matching sibling wrapper patterns.
// Most commands are fast; workflow commands may run longer.
const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
const WORKFLOW_TIMEOUT_MS = 600_000; // 10 minutes

const WORKFLOW_COMMAND_SET = new Set([
  "complete", "patch", "plan", "build", "test", "review", "document", "ship",
]);

function getCommandTimeout(command: string): number {
  return WORKFLOW_COMMAND_SET.has(command) ? WORKFLOW_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
}

function sanitizeSnippet(value: string, limit: number = ERROR_SNIPPET_LIMIT): string {
  if (!value) return "";
  const normalized = value.replace(/\r\n?/g, "\n").trim();
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, limit).trimEnd()}... [truncated]`;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const trimmed = String(value).trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

// Protected flags that have dedicated tool parameters. Reject these in
// additional args to prevent smuggling via --flag=value or duplicate injection.
const PROTECTED_FLAGS = new Set([
  "--adw-id",
  "--model",
  "--help",
  "--title",
  "--body",
  "--text",
  "--source-issue",
]);

function validateAdditionalArgs(rawAdditionalArgs: unknown):
  | { ok: true; args: string[] }
  | { ok: false; error: string } {
  if (rawAdditionalArgs === undefined) {
    return { ok: true, args: [] };
  }

  if (!Array.isArray(rawAdditionalArgs)) {
    return {
      ok: false,
      error: "ERROR: Invalid 'args': expected an array of strings.",
    };
  }

  const invalidElement = rawAdditionalArgs.find(
    (value) => typeof value !== "string" || value.trim().length === 0,
  );
  if (invalidElement !== undefined) {
    return {
      ok: false,
      error: "ERROR: Invalid 'args': all entries must be non-empty strings.",
    };
  }

  const parsedArgs = rawAdditionalArgs.map((value) => (value as string).trim());
  if (rawAdditionalArgs.length > 0 && parsedArgs.length === 0) {
    return {
      ok: false,
      error: "ERROR: Invalid 'args': no usable arguments after validation.",
    };
  }

  // Check for protected flags — both exact matches and --flag=value format.
  const protectedFlag = parsedArgs.find((arg) =>
    [...PROTECTED_FLAGS].some(
      (flag) => arg === flag || arg.startsWith(flag + "="),
    ),
  );
  if (protectedFlag) {
    const matchedFlag = [...PROTECTED_FLAGS].find(
      (flag) => protectedFlag === flag || protectedFlag.startsWith(flag + "="),
    );
    return {
      ok: false,
      error: `ERROR: Protected flag '${matchedFlag}' is not allowed in 'args'. Use top-level tool arguments instead.`,
    };
  }

  return { ok: true, args: parsedArgs };
}

export default tool({
  description: `Execute ADW (AI Developer Workflow) CLI commands. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

  Full workflow:  { command: "complete", issue_number: 123, model: "base" }
  Quick patch:    { command: "patch", issue_number: 456 }
  Resume work:    { command: "build", issue_number: 123, adw_id: "a1b2c3d4" }
  Create issue:   { command: "create-issue", title: "Add feature", body: "Description..." }
  Interpret text: { command: "interpret-issue", text: "Add tests for auth module" }
  Check status:   { command: "status" }
  Setup env:      { command: "setup", args: ["env"] }

RULES:
- Workflow commands (complete/patch/plan/build/test/review/document/ship) require issue_number.
- setup requires args for subcommands (e.g., ["template", "validate"]).
- Omit optional fields entirely -- blank strings are treated as omitted.
- Set help: true to see CLI usage for any command.

See .opencode/tools/adw.md for full parameter reference, setup commands, and model tiers.`,
  args: {
    command: tool.schema
      .enum([
        "complete", "patch", "plan", "build", "test", "review", "document",
        "ship", "status", "health", "init", "create-issue", "interpret-issue",
        "maintenance", "launch", "stop", "docstring", "finalize-docs", "setup"
      ])
      .describe(`ADW command to execute. Set help: true to see detailed usage for any command.`),
    
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

If not provided, a new ADW ID is automatically generated. Blank strings are treated as omitted.
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
      .any()
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
    const { command, issue_number, model, args: rawAdditionalArgs, help } = args;
    const adw_id = normalizeOptionalString(args.adw_id);
    const text = normalizeOptionalString(args.text);
    const title = normalizeOptionalString(args.title);
    const body = normalizeOptionalString(args.body);
    const additionalArgsValidation = validateAdditionalArgs(rawAdditionalArgs);
    if (!additionalArgsValidation.ok) {
      return additionalArgsValidation.error;
    }
    const additionalArgs = additionalArgsValidation.args;

    // Command validation and argument requirements
    const workflowCommands = [
      "complete", "patch", "plan", "build", "test", "review", "document", "ship"
    ];

    // Validate required arguments based on command type (skip if help flag is set)
    if (!help) {
      if (workflowCommands.includes(command)) {
        if (issue_number === undefined || issue_number === null) {
          return `ERROR: Command '${command}' requires 'issue_number' argument.\n\nUsage: adw ${command} <issue_number> [--adw-id <id>] [--model <base|heavy>]`;
        }
        if (!Number.isInteger(issue_number) || issue_number <= 0) {
          return `ERROR: Command '${command}' requires a positive integer 'issue_number'.`;
        }
      }

      if (command === "create-issue") {
        if (!title || !body) {
          return `ERROR: Command 'create-issue' requires both 'title' and 'body' arguments.\n\nUsage: adw create-issue --title "Issue Title" --body "Issue description"`;
        }
      }

      if (command === "interpret-issue") {
        if (!text && (issue_number === undefined || issue_number === null)) {
          return `ERROR: Command 'interpret-issue' requires either 'text' argument or 'issue_number' argument.\n\nUsage: adw interpret-issue --text "Description" OR adw interpret-issue --source-issue <number>`;
        }
        if (!text && (!Number.isInteger(issue_number) || issue_number <= 0)) {
          return `ERROR: Command 'interpret-issue' requires a positive integer 'issue_number' when 'text' is omitted.`;
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
    if (command === "setup") {
      if (help) {
        cmdParts.push("--help");
      } else if (additionalArgs && additionalArgs.length > 0) {
        cmdParts.push(...additionalArgs);
      }
    } else {
      if (help) {
        cmdParts.push("--help");
      }

      if (workflowCommands.includes(command) && issue_number !== undefined && issue_number !== null && !help) {
        cmdParts.push(issue_number.toString());
      }

      if (adw_id !== undefined) {
        const normalizedAdwId = normalizeAdwId(adw_id);
        if (!normalizedAdwId) {
          return `ERROR: ${adwIdValidationMessage()}`;
        }
        cmdParts.push("--adw-id", normalizedAdwId);
      }

      if (model) {
        cmdParts.push("--model", model);
      }

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
        } else if (issue_number !== undefined && issue_number !== null) {
          cmdParts.push("--source-issue", issue_number.toString());
        }
      }

      if (command === "docstring" && issue_number !== undefined && issue_number !== null) {
        cmdParts.push(issue_number.toString());
      }

      if (command === "finalize-docs" && issue_number !== undefined && issue_number !== null) {
        cmdParts.push(issue_number.toString());
      }

      if (additionalArgs && additionalArgs.length > 0) {
        cmdParts.push(...additionalArgs);
      }
    }

    try {
      const commandTimeout = getCommandTimeout(command);
      const result = Bun.spawnSync({
        cmd: cmdParts,
        stdout: "pipe",
        stderr: "pipe",
        timeout: commandTimeout,
        env: sanitizedEnv(),
      });

      const stdoutRaw = result.stdout ? decoder.decode(result.stdout) : "";
      const stderrRaw = result.stderr ? decoder.decode(result.stderr) : "";

      const timedOut = Boolean((result as { timedOut?: boolean }).timedOut);
      if (timedOut) {
        const output = sanitizeSnippet(stderrRaw || stdoutRaw || `Command timed out after ${commandTimeout}ms`);
        return `ERROR: Failed to execute 'adw ${command}' (timeout after ${commandTimeout}ms).\n${output}`;
      }

      if (result.exitCode !== 0) {
        const safeStderr = sanitizeSnippet(stderrRaw);
        const safeStdout = sanitizeSnippet(stdoutRaw);
        const output = safeStderr || safeStdout || `Exit code ${result.exitCode}`;
        return `ERROR: Failed to execute 'adw ${command}' (exit ${result.exitCode}).\n${output}`;
      }

      const output = stdoutRaw.trim();
      if (!output) {
        return `ADW Command: ${command}\n\nadw ${command} completed with no output.`;
      }

      if (output.includes("ERROR:") || output.includes("Error:")) {
        return `ADW Command Failed:\n${output}`;
      }

      return `ADW Command: ${command}\n\n${output}`;

    } catch (error: any) {
      const stderr = error?.stderr instanceof Uint8Array
        ? decoder.decode(error.stderr)
        : typeof error?.stderr === "string" ? error.stderr : "";
      const stdout = error?.stdout instanceof Uint8Array
        ? decoder.decode(error.stdout)
        : typeof error?.stdout === "string" ? error.stdout : "";
      const message = error?.message ? String(error.message) : "";

      const safeStderr = sanitizeSnippet(stderr);
      const safeStdout = sanitizeSnippet(stdout);
      const safeMessage = sanitizeSnippet(message);
      const fallback = safeStderr || safeStdout || safeMessage || "Unknown execution error";

      return `ERROR: Failed to execute 'adw ${command}' (execution error).\n${fallback}`;
    }
  },
});
