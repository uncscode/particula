import { tool } from "@opencode-ai/plugin";

import { getStructuredJsonPayload, sanitizeAndTruncate } from "./lib/platform_wrapper_utils";

const ERROR_SNIPPET_LIMIT = 2000;

type PlatformIssueReadCommand = "fetch-issue";
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
  description: "Execute read-only adw platform issue operations with strict validation.",
  args: {
    command: tool.schema.enum(["fetch-issue"]).describe("Command to execute."),
    issue_number: tool.schema.string().optional(),
    output_format: tool.schema.enum(["text", "json"]).optional(),
    prefer_scope: tool.schema.enum(["fork", "upstream"]).optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const command = args.command as PlatformIssueReadCommand;
    const issueNumberToken = normalizeIssueNumberToken(args.issue_number);
    const outputFormat = normalizeOutputFormat(args.output_format);
    const preferScope = normalizeOptionalString(args.prefer_scope);

    if (command !== "fetch-issue") {
      return buildMissingArgMessage(`Unsupported command: ${String(command)}`);
    }

    const cmdParts: (string | number)[] = ["uv", "run", "--active", "adw", "platform", command];

    if (args.help) {
      cmdParts.push("--help");
      try {
        return await runCommand(cmdParts);
      } catch (error: any) {
        const stdout = sanitizeAndTruncate(error.stdout, ERROR_SNIPPET_LIMIT);
        const stderr = sanitizeAndTruncate(error.stderr, ERROR_SNIPPET_LIMIT);
        return `ERROR: Failed to fetch help for '${command}'.${
          stdout ? `\nSTDOUT:\n${stdout}` : ""
        }${stderr ? `\nSTDERR:\n${stderr}` : ""}`;
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

    cmdParts.push(issueNumberToken);
    if (outputFormat) {
      cmdParts.push("--format", outputFormat);
    }
    if (preferScope) {
      cmdParts.push("--prefer-scope", preferScope);
    }

    try {
      return await runCommand(cmdParts);
    } catch (error: any) {
      const stdout = sanitizeAndTruncate(error.stdout, ERROR_SNIPPET_LIMIT);
      const stderr = sanitizeAndTruncate(error.stderr, ERROR_SNIPPET_LIMIT);
       if (outputFormat === "json") {
         const structuredJson = getStructuredJsonPayload(error.stdout);
         if (structuredJson) {
           return structuredJson;
         }
       }
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
