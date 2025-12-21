/**
 * Platform Operations Tool for OpenCode Integration
 *
 * Provides structured access to issue, PR, label, and comment operations through the
 * ADW PlatformRouter. This tool wraps `adw platform` commands so agents no longer need
 * bash `gh` allowlists and remains platform-agnostic across GitHub/GitLab.
 *
 * See docs/Agent/development_plans/features/tool-only-agent-permissions.md for the
 * overarching permissions rollout plan.
 */

import { tool } from "@opencode-ai/plugin";

const ERROR_SNIPPET_LIMIT = 2000;

const REQUIRED_ISSUE_COMMANDS = new Set([
  "fetch-issue",
  "update-issue",
  "add-labels",
  "remove-labels",
  "comment",
  "pr-comments",
]);

const JSON_CAPABLE_COMMANDS = new Set([
  "fetch-issue",
  "create-issue",
  "update-issue",
  "add-labels",
  "remove-labels",
  "pr-comments",
]);

type PlatformCommand =
  | "create-pr"
  | "fetch-issue"
  | "create-issue"
  | "update-issue"
  | "add-labels"
  | "remove-labels"
  | "comment"
  | "pr-comments";

type OutputFormat = "text" | "json";

function truncate(value: string | undefined): string {
  if (!value) return "";
  return value.length > ERROR_SNIPPET_LIMIT
    ? `${value.slice(0, ERROR_SNIPPET_LIMIT)}...<truncated>`
    : value;
}

function buildMissingArgMessage(message: string): string {
  return `ERROR: ${message}`;
}

async function runCommand(cmdParts: (string | number)[]): Promise<string> {
  return Bun.$`${cmdParts}`.text();
}

export default tool({
  description: `Execute platform operations (GitHub/GitLab) via the ADW PlatformRouter.

Platform operations work with both GitHub and GitLab repositories automatically.
The router handles authentication and API differences between platforms.

AVAILABLE COMMANDS:
• create-pr: Create a pull request (or merge request on GitLab)
  Usage: { command: "create-pr", title: "feat: add feature", head: "feature-123", base: "main" }

• fetch-issue: Get issue details with optional JSON output
  Usage: { command: "fetch-issue", issue_number: "123", output_format: "json" }
  Supports prefer_scope: "fork" | "upstream" for fork/upstream routing.

• create-issue: Create a new issue
  Usage: { command: "create-issue", title: "Bug", body: "Details...", labels: "bug,triage" }

• update-issue: Modify existing issue (title, body, state, labels)
  Usage: { command: "update-issue", issue_number: "123", state: "closed" }

• add-labels: Add labels to issue or PR
  Usage: { command: "add-labels", issue_number: "123", labels: "enhancement,docs" }

• remove-labels: Remove labels from issue or PR
  Usage: { command: "remove-labels", issue_number: "123", labels: "wontfix" }

• comment: Add comment to issue or PR
  Usage: { command: "comment", issue_number: "123", body: "LGTM" }

• pr-comments: Fetch review comments for a pull request
  Usage: { command: "pr-comments", issue_number: "42", output_format: "json" }
  Supports actionable_only: true to filter to unresolved comments with meaningful content.
  Supports prefer_scope: "fork" | "upstream" for fork/upstream routing.

JSON OUTPUT: Use output_format: "json" to get structured responses for parsing.

HELP: Set help: true to see command-specific CLI usage
  Usage: { command: "create-pr", help: true }

Platform-agnostic routing is handled by PlatformRouter and supports GitHub/GitLab.
See docs/Agent/development_plans/features/tool-only-agent-permissions.md for context.`,
  args: {
    command: tool.schema
      .enum([
        "create-pr",
        "fetch-issue",
        "create-issue",
        "update-issue",
        "add-labels",
        "remove-labels",
        "comment",
        "pr-comments",
      ])
      .describe(`Platform operation to execute.

COMMAND DESCRIPTIONS:
• create-pr - Create pull request (GitHub) or merge request (GitLab)
• fetch-issue - Retrieve issue details and metadata
• create-issue - Create new issue with title, body, and labels
• update-issue - Modify issue title, body, state, or labels
• add-labels - Add labels to an issue or PR
• remove-labels - Remove labels from an issue or PR
• comment - Post a comment on an issue or PR
• pr-comments - Fetch review comments for a pull request

REQUIRED PARAMETERS BY COMMAND:
• create-pr: title, head, base (required), optional: body, draft
• fetch-issue: issue_number (required), optional: output_format, prefer_scope
• create-issue: title (required), optional: body, labels, output_format
• update-issue: issue_number (required), at least one of: title, body, state, labels
• add-labels: issue_number, labels (required), optional: output_format
• remove-labels: issue_number, labels (required), optional: output_format
• comment: issue_number, body (required)
• pr-comments: issue_number (required), optional: output_format, actionable_only, prefer_scope`),

    issue_number: tool.schema
      .string()
      .optional()
      .describe(`Issue or PR number to operate on.

REQUIRED FOR: fetch-issue, update-issue, add-labels, remove-labels, comment, pr-comments

Can be a simple number or a string. The CLI accepts both formats.

EXAMPLE: issue_number: "123"`),

    title: tool.schema
      .string()
      .optional()
      .describe(`Title for PR or issue.

REQUIRED FOR: create-pr, create-issue
OPTIONAL FOR: update-issue

For PRs, follow the format: "<type>: #<issue> - <description>"
Example: "feat: #123 - Add user authentication module"

EXAMPLE: title: "feat: #456 - Add data validation"`),

    body: tool.schema
      .string()
      .optional()
      .describe(`Body content for PR, issue, or comment.

OPTIONAL FOR: create-pr, create-issue, update-issue
REQUIRED FOR: comment

Supports markdown formatting.

EXAMPLE: body: "## Summary\\nFixes authentication bug\\n\\n## Testing\\n- Unit tests added"`),

    head: tool.schema
      .string()
      .optional()
      .describe(`Source branch for pull request.

REQUIRED FOR: create-pr

EXAMPLE: head: "feature-issue-123-add-auth"`),

    base: tool.schema
      .string()
      .optional()
      .describe(`Target branch for pull request.

REQUIRED FOR: create-pr

Usually "main" or "master".

EXAMPLE: base: "main"`),

    draft: tool.schema
      .boolean()
      .optional()
      .describe(`Create pull request as draft.

APPLIES TO: create-pr only

EXAMPLE: draft: true`),

    state: tool.schema
      .enum(["open", "closed"])
      .optional()
      .describe(`Issue workflow state.

APPLIES TO: update-issue only

EXAMPLE: state: "closed"`),

    labels: tool.schema
      .string()
      .optional()
      .describe(`Comma-separated list of labels.

OPTIONAL FOR: create-issue, update-issue
REQUIRED FOR: add-labels, remove-labels

EXAMPLE: labels: "bug,type:patch,model:default"`),

    output_format: tool.schema
      .enum(["text", "json"])
      .optional()
      .describe(`Output format for results.

APPLIES TO: fetch-issue, create-issue, update-issue, add-labels, remove-labels, pr-comments

DEFAULT: "text" - Human-readable output
JSON: "json" - Structured JSON for programmatic parsing

EXAMPLE: output_format: "json"`),

    prefer_scope: tool.schema
      .enum(["fork", "upstream"])
      .optional()
      .describe(`Repository scope preference.

APPLIES TO: fetch-issue, pr-comments

For fork workflows where issues live upstream but PRs are on fork.

EXAMPLE: prefer_scope: "upstream"`),

    actionable_only: tool.schema
      .boolean()
      .optional()
      .describe(`Filter to actionable review comments only.

APPLIES TO: pr-comments only

When true, returns only unresolved comments with meaningful content.
Filters out resolved comments and empty/trivial feedback.

EXAMPLE: actionable_only: true`),

    help: tool.schema
      .boolean()
      .optional()
      .describe(`Show CLI help for the specified command.

When true, returns the --help output for the command instead of executing it.

EXAMPLE: { command: "create-issue", help: true }`),
  },

  async execute(args) {
    const command = args.command as PlatformCommand;
    const outputFormat: OutputFormat = (args.output_format as OutputFormat) || "text";
    const cmdParts: (string | number)[] = ["uv", "run", "adw", "platform", command];

    if (args.help) {
      cmdParts.push("--help");
      try {
        return await runCommand(cmdParts);
      } catch (error: any) {
        const stdout = truncate(error.stdout?.toString());
        const stderr = truncate(error.stderr?.toString());
        return `ERROR: Failed to fetch help for '${command}'.${
          stdout ? `\nSTDOUT:\n${stdout}` : ""
        }${stderr ? `\nSTDERR:\n${stderr}` : ""}`;
      }
    }

    if (args.prefer_scope && command !== "fetch-issue" && command !== "pr-comments") {
      return buildMissingArgMessage(
        "'prefer_scope' is only supported for 'fetch-issue' and 'pr-comments' commands"
      );
    }

    if (args.actionable_only && command !== "pr-comments") {
      return buildMissingArgMessage(
        "'actionable_only' is only supported for the 'pr-comments' command"
      );
    }

    if (REQUIRED_ISSUE_COMMANDS.has(command) && !args.issue_number) {
      return buildMissingArgMessage(
        `'issue_number' is required for command '${command}'`
      );
    }

    const labelsProvided = args.labels !== undefined;
    const labelsEmpty = labelsProvided && !args.labels?.trim();

    switch (command) {
      case "create-pr": {
        if (!args.title) {
          return buildMissingArgMessage("'title' is required for command 'create-pr'");
        }
        if (!args.head) {
          return buildMissingArgMessage("'head' is required for command 'create-pr'");
        }
        if (!args.base) {
          return buildMissingArgMessage("'base' is required for command 'create-pr'");
        }

        cmdParts.push("--title", args.title, "--head", args.head, "--base", args.base);
        if (args.body) {
          cmdParts.push("--body", args.body);
        }
        if (args.draft === true) {
          cmdParts.push("--draft");
        } else if (args.draft === false) {
          cmdParts.push("--no-draft");
        }
        break;
      }

      case "fetch-issue": {
        cmdParts.push(args.issue_number as string);
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        if (args.prefer_scope) {
          cmdParts.push("--prefer-scope", args.prefer_scope);
        }
        break;
      }

      case "create-issue": {
        if (!args.title) {
          return buildMissingArgMessage("'title' is required for command 'create-issue'");
        }
        cmdParts.push("--title", args.title);
        if (args.body) {
          cmdParts.push("--body", args.body);
        }
        if (labelsProvided) {
          cmdParts.push("--labels", args.labels as string);
        }
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "update-issue": {
        if (labelsEmpty) {
          return buildMissingArgMessage("Provide at least one label (comma-separated)");
        }
        if (!args.title && !args.body && !args.state && !labelsProvided) {
          return buildMissingArgMessage(
            "Provide at least one field to update (title/body/state/labels)"
          );
        }

        cmdParts.push(args.issue_number as string);
        if (args.title) {
          cmdParts.push("--title", args.title);
        }
        if (args.body) {
          cmdParts.push("--body", args.body);
        }
        if (args.state) {
          cmdParts.push("--state", args.state);
        }
        if (labelsProvided) {
          cmdParts.push("--labels", args.labels as string);
        }
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "add-labels": {
        if (!labelsProvided || labelsEmpty) {
          return buildMissingArgMessage("'labels' is required and must contain at least one label (comma-separated)");
        }
        cmdParts.push(args.issue_number as string, "--labels", args.labels as string);
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "remove-labels": {
        if (!labelsProvided || labelsEmpty) {
          return buildMissingArgMessage("'labels' is required and must contain at least one label (comma-separated)");
        }
        cmdParts.push(args.issue_number as string, "--labels", args.labels as string);
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "comment": {
        if (!args.body) {
          return buildMissingArgMessage("'body' is required for command 'comment'");
        }
        cmdParts.push(args.issue_number as string, "--body", args.body);
        break;
      }

      case "pr-comments": {
        cmdParts.push(args.issue_number as string);
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        if (args.actionable_only === true) {
          cmdParts.push("--actionable-only");
        }
        if (args.prefer_scope) {
          cmdParts.push("--prefer-scope", args.prefer_scope);
        }
        break;
      }

      default:
        return buildMissingArgMessage(`Unsupported command: ${command}`);
    }

    try {
      const result = await runCommand(cmdParts);
      
      // Add clear success signals for create-pr command
      if (command === "create-pr") {
        // Check if the result contains a PR URL (indicates success)
        const prUrlMatch = result.match(/https:\/\/github\.com\/[^\/]+\/[^\/]+\/pull\/(\d+)|https:\/\/gitlab\.com\/[^\/]+\/[^\/]+\/-\/merge_requests\/(\d+)/);
        const prNumberMatch = result.match(/(?:PR|MR|pull request|merge request)\s*#?(\d+)/i);
        
        if (prUrlMatch || prNumberMatch) {
          // Success - add clear signal
          const prNumber = prUrlMatch?.[1] || prUrlMatch?.[2] || prNumberMatch?.[1] || "unknown";
          return `PLATFORM_PR_CREATED

${result}

---
PR_NUMBER: ${prNumber}
STATUS: SUCCESS`;
        } else if (result.toLowerCase().includes("created") || result.toLowerCase().includes("success")) {
          // Likely success but couldn't extract PR number
          return `PLATFORM_PR_CREATED

${result}

---
STATUS: SUCCESS (PR number not parsed - check output above)`;
        }
      }
      
      return result;
    } catch (error: any) {
      const stdout = truncate(error.stdout?.toString());
      const stderr = truncate(error.stderr?.toString());

      if (outputFormat === "json" && stdout) {
        return stdout;
      }

      // Add clear failure signal for create-pr command
      if (command === "create-pr") {
        const parts: string[] = [
          `PLATFORM_PR_FAILED`,
          ``,
          `ERROR: Failed to create pull request via 'adw platform ${command}'`,
        ];
        if (stderr) {
          parts.push(`STDERR:\n${stderr}`);
        }
        if (stdout) {
          parts.push(`STDOUT:\n${stdout}`);
        }
        parts.push(``, `---`, `STATUS: FAILED`);
        return parts.join("\n");
      }

      const parts: string[] = [
        `ERROR: Failed to execute 'adw platform ${command}'`,
      ];
      if (stderr) {
        parts.push(`STDERR:\n${stderr}`);
      }
      if (stdout) {
        parts.push(`STDOUT:\n${stdout}`);
      }
      return parts.join("\n\n");
    }
  },
});
