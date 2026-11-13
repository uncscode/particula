import { tool } from "@opencode-ai/plugin";

import {
  getStructuredJsonPayload,
  sanitizeAndTruncate,
} from "./lib/platform_wrapper_utils";

const ERROR_SNIPPET_LIMIT = 2000;

type PlatformPrReadCommand = "pr-comments" | "pr-diff";
type OutputFormat = "text" | "json";

function buildMissingArgMessage(message: string): string {
  return `ERROR: ${message}`;
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

function normalizeOutputFormat(value: unknown): OutputFormat | undefined {
  const token = normalizeOptionalString(value);
  if (token === undefined) {
    return undefined;
  }
  return token === "text" || token === "json" ? token : undefined;
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
  description: "Execute read-only adw platform PR read commands with strict validation.",
  args: {
    command: tool.schema.enum(["pr-comments", "pr-diff"]).describe("Command to execute."),
    issue_number: tool.schema.string().optional(),
    output_format: tool.schema.enum(["text", "json"]).optional(),
    actionable_only: tool.schema.boolean().optional(),
    prefer_scope: tool.schema.enum(["fork", "upstream"]).optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const command = args.command as PlatformPrReadCommand;
    const issueNumberToken = normalizeIssueNumberToken(args.issue_number);
    const outputFormat = normalizeOutputFormat(args.output_format);
    const preferScope = normalizeOptionalString(args.prefer_scope);

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

    if (!issueNumberToken) {
      return buildMissingArgMessage(`'issue_number' is required for command '${command}'`);
    }
    if (!isStrictPositiveIssueNumberToken(issueNumberToken)) {
      return buildMissingArgMessage(
        "'issue_number' must be a positive integer token (digits only; leading zeros allowed)",
      );
    }
    if (args.output_format !== undefined && outputFormat === undefined) {
      return buildMissingArgMessage("'output_format' must be either 'text' or 'json'");
    }
    if (args.prefer_scope !== undefined && preferScope === undefined) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (preferScope && !["fork", "upstream"].includes(preferScope)) {
      return buildMissingArgMessage("'prefer_scope' must be either 'fork' or 'upstream'");
    }
    if (args.actionable_only !== undefined && typeof args.actionable_only !== "boolean") {
      return buildMissingArgMessage("'actionable_only' must be a boolean when provided");
    }
    if (command === "pr-diff" && args.actionable_only !== undefined) {
      return buildMissingArgMessage("'actionable_only' is only supported for command 'pr-comments'");
    }

    cmdParts.push(issueNumberToken);
    if (outputFormat) {
      cmdParts.push("--format", outputFormat);
    }
    if (command === "pr-comments" && args.actionable_only === true) {
      cmdParts.push("--actionable-only");
    }
    if (preferScope) {
      cmdParts.push("--prefer-scope", preferScope);
    }

    try {
      return await runCommand(cmdParts);
    } catch (error: any) {
       if (outputFormat === "json") {
         const structuredJson = getStructuredJsonPayload(error.stdout);
         if (structuredJson) {
           return structuredJson;
        }
      }
      const stdout = sanitizeAndTruncate(error.stdout, ERROR_SNIPPET_LIMIT);
      const stderr = sanitizeAndTruncate(error.stderr, ERROR_SNIPPET_LIMIT);
      const parts: string[] = [`ERROR: Failed to execute 'adw platform ${command}'`];
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
