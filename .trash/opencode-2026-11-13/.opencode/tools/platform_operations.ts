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

import {
  getStructuredJsonPayload,
  normalizeLabels,
  sanitizeAndTruncate,
} from "./lib/platform_wrapper_utils";

const ERROR_SNIPPET_LIMIT = 2000;

const COMPATIBILITY_ROUTE_TO_WRAPPER = {
  "create-pr": "platform_pr_write",
  "fetch-issue": "platform_issue_read",
  "create-issue": "platform_issue_write",
  "update-issue": "platform_issue_write",
  "add-labels": "platform_label_write",
  "remove-labels": "platform_label_write",
  comment: "platform_comment_write",
  "pr-comments": "platform_pr_read",
  "pr-review": "platform_pr_review_write",
  "rate-limit": "platform_rate_limit_read",
} as const;

const SPLIT_ONLY_ROUTE_TO_WRAPPER = {
  "pr-diff": "platform_pr_read",
} as const;

const REQUIRED_ISSUE_COMMANDS = new Set([
  "fetch-issue",
  "update-issue",
  "add-labels",
  "remove-labels",
  "comment",
  "pr-comments",
  "pr-review",
]);

const JSON_CAPABLE_COMMANDS = new Set([
  "fetch-issue",
  "create-issue",
  "update-issue",
  "add-labels",
  "remove-labels",
  "pr-comments",
  "rate-limit",
]);

const PREFER_SCOPE_COMMANDS = [
  "create-pr",
  "fetch-issue",
  "create-issue",
  "update-issue",
  "add-labels",
  "remove-labels",
  "comment",
  "pr-comments",
  "pr-review",
  "rate-limit",
] as const;

const PREFER_SCOPE_COMMAND_SET = new Set(PREFER_SCOPE_COMMANDS);
const PREFER_SCOPE_COMMAND_LIST = PREFER_SCOPE_COMMANDS.join(", ");

 type PlatformCommand = keyof typeof COMPATIBILITY_ROUTE_TO_WRAPPER;


type OutputFormat = "text" | "json";

function isSupportedCompatibilityCommand(value: string): value is PlatformCommand {
  return value in COMPATIBILITY_ROUTE_TO_WRAPPER;
}

function normalizeOutputFormat(value: unknown): OutputFormat | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const token = String(value).trim();
  return token === "text" || token === "json" ? token : undefined;
}

function isStrictPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && Number.isInteger(value) && value > 0;
}

function isSafeInlinePath(pathToken: string): boolean {
  if (!pathToken || pathToken.startsWith("/") || pathToken.startsWith("\\")) {
    return false;
  }
  if (pathToken.includes("\\")) {
    return false;
  }
  const segments = pathToken.split("/");
  if (segments.some((segment) => segment.length === 0 || segment === "." || segment === "..")) {
    return false;
  }
  return true;
}

function isShaLikeToken(value: string): boolean {
  return /^[0-9a-fA-F]{7,64}$/.test(value);
}

function buildMissingArgMessage(message: string): string {
  return `ERROR: ${message}`;
}

function buildCompatibilityGuidance(command: string, wrapper: string): string {
  return buildMissingArgMessage(
    `platform_operations compatibility mode does not support '${command}'. Use '${wrapper}' instead.`
  );
}

function normalizeIssueNumberToken(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }

  const token = String(value).trim();
  return token.length > 0 ? token : undefined;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const token = String(value).trim();
  return token.length > 0 ? token : undefined;
}

function isSafeIssueNumberToken(token: string): boolean {
  return /^[0-9]+$/.test(token);
}

function isStrictPositiveIssueNumberToken(token: string): boolean {
  return isSafeIssueNumberToken(token) && !/^0+$/.test(token);
}

async function runCommand(cmdParts: (string | number)[]): Promise<string> {
  return Bun.$`${cmdParts}`.text();
}

export default tool({
  description: `Execute platform operations (GitHub/GitLab) via the ADW PlatformRouter. Compatibility-only wrapper retained for legacy callers; prefer the mapped split wrappers for migrated commands. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

  Create PR:     { command: "create-pr", title: "feat: #123 - Add auth", head: "feature-123", base: "main" }
  Fetch issue:   { command: "fetch-issue", issue_number: "123", output_format: "json" }
  Create issue:  { command: "create-issue", title: "Bug", body: "Details...", labels: "bug,triage" }
  Close issue:   { command: "update-issue", issue_number: "123", state: "closed" }
  Add labels:    { command: "add-labels", issue_number: "123", labels: "enhancement,docs" }
  Comment:       { command: "comment", issue_number: "123", body: "LGTM" }
  PR comments:   { command: "pr-comments", issue_number: "42", output_format: "json" }

RULES:
- Works with both GitHub and GitLab automatically.
- Omit optional fields entirely -- do NOT pass empty strings or false placeholders.
- create-pr: base auto-resolves from ADW state when adw_id is provided and base is omitted.
- Use output_format: "json" for structured responses.
- Set help: true to see CLI usage for any command.

See .opencode/tools/platform_operations.md for full parameter reference and advanced usage.`,
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
        "pr-review",
        "rate-limit",
      ])
      .describe(`Platform command to execute. Set help: true to see CLI usage for any command.`),

    issue_number: tool.schema
      .string()
      .optional()
      .describe(`Issue or PR number to operate on.

REQUIRED FOR: fetch-issue, update-issue, add-labels, remove-labels, comment, pr-comments
and pr-review

Must be a positive integer token (digits only; leading zeros allowed).

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

OPTIONAL FOR: create-pr (auto-resolved when omitted)

When omitted, the CLI resolves target branch in this order:
1. Read target_branch from ADW state (if adw_id provided)
2. Fall back to "main"

This enables PR stacking: issues with [branch:xxx] prefix in title
automatically target that branch without manual specification.

Usually "main" or "master" when specified explicitly.

EXAMPLE: base: "main"
EXAMPLE: (omitted) - auto-resolves from state or defaults to main`),

    adw_id: tool.schema
      .string()
      .optional()
      .describe(`ADW workflow ID for state lookup.

OPTIONAL FOR: create-pr

When provided with create-pr and base is omitted, the CLI reads
target_branch from the workflow state file (agents/{adw_id}/adw_state.json).

This enables automatic PR targeting for stacked PRs where the issue
title contains [branch:xxx] prefix.

EXAMPLE: adw_id: "abc12345"`),

    draft: tool.schema
      .boolean()
      .optional()
      .describe(`Create pull request as draft.

APPLIES TO: create-pr only

Only draft: true is forwarded. Omit draft, or send draft: false, for a non-draft PR.

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

Blank labels are treated as omitted for optional-label commands.

EXAMPLE: labels: "bug,type:patch,model:default"`),

    output_format: tool.schema
      .enum(["text", "json"])
      .optional()
      .describe(`Output format for results.

APPLIES TO: fetch-issue, create-issue, update-issue, add-labels, remove-labels, pr-comments, rate-limit

DEFAULT: "text" - Human-readable output
JSON: "json" - Structured JSON for programmatic parsing

EXAMPLE: output_format: "json"`),

    prefer_scope: tool.schema
      .enum(["fork", "upstream"])
      .optional()
      .describe(`Repository scope preference.

APPLIES TO: ${PREFER_SCOPE_COMMAND_LIST}

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

    path: tool.schema.string().optional(),
    line: tool.schema.number().optional(),
    commit_sha: tool.schema.string().optional(),
    position: tool.schema.number().optional(),

    help: tool.schema
      .boolean()
      .optional()
      .describe(`Show CLI help for the specified command.

When true, returns the --help output for the command instead of executing it.

EXAMPLE: { command: "create-issue", help: true }`),
  },

  async execute(args) {
    const rawCommand = String(args.command ?? "").trim();
    if (rawCommand in SPLIT_ONLY_ROUTE_TO_WRAPPER) {
      return buildCompatibilityGuidance(
        rawCommand,
        SPLIT_ONLY_ROUTE_TO_WRAPPER[rawCommand as keyof typeof SPLIT_ONLY_ROUTE_TO_WRAPPER]
      );
    }
    if (!isSupportedCompatibilityCommand(rawCommand)) {
      return buildMissingArgMessage(`Unsupported command: ${rawCommand}`);
    }

    const command = rawCommand;
    const normalizedOutputFormat = normalizeOutputFormat(args.output_format);
    const outputFormat: OutputFormat = normalizedOutputFormat || "text";
    const issueNumberToken = normalizeIssueNumberToken(args.issue_number);
    const title = normalizeOptionalString(args.title);
    const body = normalizeOptionalString(args.body);
    const head = normalizeOptionalString(args.head);
    const base = normalizeOptionalString(args.base);
    const adwId = normalizeOptionalString(args.adw_id);
     const labels = normalizeLabels(args.labels);
    const preferScope = normalizeOptionalString(args.prefer_scope);
    const path = normalizeOptionalString(args.path);
    const commitSha = normalizeOptionalString(args.commit_sha);
    const cmdParts: (string | number)[] = ["uv", "run", "--active", "adw", "platform", command];

    if (args.help) {
      cmdParts.push("--help");
      try {
        return await runCommand(cmdParts);
      } catch (error: any) {
        const stderr = sanitizeAndTruncate(error.stderr, ERROR_SNIPPET_LIMIT);
        const stdout = sanitizeAndTruncate(error.stdout, ERROR_SNIPPET_LIMIT);
        return `ERROR: Failed to fetch help for '${command}'.${
          stderr ? `\nSTDERR:\n${stderr}` : ""
        }${stdout ? `\nSTDOUT:\n${stdout}` : ""}`;
      }
    }

    if (args.output_format !== undefined && normalizedOutputFormat === undefined) {
      return buildMissingArgMessage("'output_format' must be either 'text' or 'json'");
    }

    if (args.prefer_scope !== undefined && preferScope === undefined) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (preferScope && !["fork", "upstream"].includes(preferScope)) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }

    if (preferScope && !PREFER_SCOPE_COMMAND_SET.has(command)) {
      return buildMissingArgMessage(
        `'prefer_scope' is only supported for ${PREFER_SCOPE_COMMAND_LIST} commands`
      );
    }

    if (args.actionable_only && command !== "pr-comments") {
      return buildMissingArgMessage(
        "'actionable_only' is only supported for the 'pr-comments' command"
      );
    }
    if (args.actionable_only !== undefined && typeof args.actionable_only !== "boolean") {
      return buildMissingArgMessage("'actionable_only' must be a boolean when provided");
    }

    if (REQUIRED_ISSUE_COMMANDS.has(command) && !issueNumberToken) {
      return buildMissingArgMessage(
        `'issue_number' is required for command '${command}'`
      );
    }

    if (issueNumberToken && !isStrictPositiveIssueNumberToken(issueNumberToken)) {
      return buildMissingArgMessage(
        `'issue_number' must be a positive integer token (digits only; leading zeros allowed)`
      );
    }

     const labelsProvided = labels !== undefined;

    switch (command) {
      case "create-pr": {
        if (!title) {
          return buildMissingArgMessage("'title' is required for command 'create-pr'");
        }
        if (!head) {
          return buildMissingArgMessage("'head' is required for command 'create-pr'");
        }

        cmdParts.push("--title", title, "--head", head);
        
        // Base is optional - CLI will resolve from state or default to main
        if (base) {
          cmdParts.push("--base", base);
        }
        
        // Pass adw_id to CLI for state-based target_branch lookup when base is omitted
        if (adwId) {
          cmdParts.push("--adw-id", adwId);
        }
        if (body) {
          cmdParts.push("--body", body);
        }
        if (args.draft === true) {
          cmdParts.push("--draft");
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        break;
      }

      case "fetch-issue": {
        cmdParts.push(issueNumberToken as string);
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        break;
      }

      case "create-issue": {
        if (!title) {
          return buildMissingArgMessage("'title' is required for command 'create-issue'");
        }
        cmdParts.push("--title", title);
        if (body) {
          cmdParts.push("--body", body);
        }
        if (labelsProvided) {
          cmdParts.push("--labels", labels as string);
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "update-issue": {
        if (!title && !body && !args.state && !labelsProvided) {
          return buildMissingArgMessage(
            "Provide at least one field to update (title/body/state/labels)"
          );
        }

        cmdParts.push(issueNumberToken as string);
        if (title) {
          cmdParts.push("--title", title);
        }
        if (body) {
          cmdParts.push("--body", body);
        }
        if (args.state) {
          cmdParts.push("--state", args.state);
        }
        if (labelsProvided) {
          cmdParts.push("--labels", labels as string);
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "add-labels": {
        if (!labelsProvided) {
          return buildMissingArgMessage("'labels' is required and must contain at least one label (comma-separated)");
        }
        cmdParts.push(issueNumberToken as string, "--labels", labels as string);
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "remove-labels": {
        if (!labelsProvided) {
          return buildMissingArgMessage("'labels' is required and must contain at least one label (comma-separated)");
        }
        cmdParts.push(issueNumberToken as string, "--labels", labels as string);
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        break;
      }

      case "comment": {
        if (!body) {
          return buildMissingArgMessage("'body' is required for command 'comment'");
        }
        cmdParts.push(issueNumberToken as string, "--body", body);
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        break;
      }

      case "pr-comments": {
        cmdParts.push(issueNumberToken as string);
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        if (args.actionable_only === true) {
          cmdParts.push("--actionable-only");
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        break;
      }

      case "pr-review": {
        if (!body) {
          return buildMissingArgMessage("'body' is required for command 'pr-review'");
        }
        const line = args.line;
        const position = args.position;
        if (line !== undefined && path === undefined) {
          return buildMissingArgMessage("'--line' requires '--path' for command 'pr-review'");
        }
        if (position !== undefined && path === undefined) {
          return buildMissingArgMessage("'--position' requires '--path' for command 'pr-review'");
        }
        if (path && line === undefined && position === undefined) {
          return buildMissingArgMessage(
            "'--path' requires '--line' or '--position' for command 'pr-review'"
          );
        }
        if (line !== undefined && !isStrictPositiveInteger(line)) {
          return buildMissingArgMessage("'line' must be a positive integer for command 'pr-review'");
        }
        if (position !== undefined && !isStrictPositiveInteger(position)) {
          return buildMissingArgMessage(
            "'position' must be a positive integer for command 'pr-review'"
          );
        }
        if (path && !isSafeInlinePath(path)) {
          return buildMissingArgMessage(
            "'path' must be a safe repository-relative path without traversal for command 'pr-review'"
          );
        }
        if (commitSha && !isShaLikeToken(commitSha)) {
          return buildMissingArgMessage("'commit_sha' must be a SHA-like hex token (7-64 chars)");
        }

        cmdParts.push(issueNumberToken as string, "--body", body);
        if (path) {
          cmdParts.push("--path", path);
        }
        if (line !== undefined) {
          cmdParts.push("--line", line as number);
        }
        if (commitSha) {
          cmdParts.push("--commit-sha", commitSha);
        }
        if (position !== undefined) {
          cmdParts.push("--position", position as number);
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        break;
      }

      case "rate-limit": {
        if (args.output_format) {
          cmdParts.push("--format", outputFormat);
        }
        if (preferScope) {
          cmdParts.push("--prefer-scope", preferScope);
        }
        break;
      }

      default:
        return buildMissingArgMessage(`Unsupported command: ${command}`);
    }

    try {
      const result = await runCommand(cmdParts);

      if (command === "create-pr") {
        if (result.includes("PLATFORM_PR_CREATED") || result.includes("PLATFORM_PR_FAILED")) {
          return result;
        }

        return `PLATFORM_PR_CREATED

${result}

---
STATUS: SUCCESS`;
      }

      return result;
    } catch (error: any) {
      const stdout = sanitizeAndTruncate(error.stdout, ERROR_SNIPPET_LIMIT);
      const stderr = sanitizeAndTruncate(error.stderr, ERROR_SNIPPET_LIMIT);

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

       if (outputFormat === "json" && JSON_CAPABLE_COMMANDS.has(command)) {
         const structuredJson = getStructuredJsonPayload(error.stdout);
         if (structuredJson) {
           return structuredJson;
         }
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
