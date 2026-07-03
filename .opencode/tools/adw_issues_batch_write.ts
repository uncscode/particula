import { tool } from "@opencode-ai/plugin";

import {
  buildError,
  isProvidedValue,
  normalizeAndValidateAdwId,
  normalizeIssue,
  normalizeSectionToken,
  parseBatchOptions,
  runBatchCommandText,
} from "./adw_issues_spec_shared";

// --- Tool implementation ---

async function executeBatchWrite(args: Record<string, any>): Promise<string> {
  const { adw_id, issue, content, section, options } = args;
  if (!isProvidedValue(adw_id)) return buildError("'adw_id' is required for all commands except batch-init.");
  if (!isProvidedValue(issue)) return buildError("batch-write requires 'issue'.");
  if (!isProvidedValue(content)) return buildError("batch-write requires 'content'.");
  const parsedOptions = parseBatchOptions("batch-write", options);
  if (!parsedOptions.ok) return parsedOptions.error;

  const normalizedAdw = normalizeAndValidateAdwId(adw_id);
  if (!normalizedAdw.ok || !normalizedAdw.value) return normalizedAdw.ok ? buildError("'adw_id' is required for all commands except batch-init.") : normalizedAdw.error;

  const normalizedIssue = normalizeIssue(issue);
  if (!normalizedIssue) return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);

  const cmdParts: (string | number)[] = [
    "uv",
    "run",
    "--active",
    "adw",
    "spec",
    "batch",
    "write",
    "--adw-id",
    normalizedAdw.value,
    "--issue",
    normalizedIssue,
    "--content",
    content,
  ];

  if (section !== undefined && section !== null && String(section).trim() !== "") {
    const normalizedSection = normalizeSectionToken(section);
    if (!normalizedSection) {
      return buildError(
        `Invalid section token "${section}". Section must not contain control/newline characters.`,
      );
    }
    cmdParts.push("--section", normalizedSection);
  }

  return runBatchCommandText("batch-write", cmdParts);
}

export default tool({
  description: "Write batch content via adw spec batch write.",
  args: {
    adw_id: tool.schema.string(),
    issue: tool.schema.string(),
    content: tool.schema.string(),
    section: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    return executeBatchWrite(args as Record<string, any>);
  },
});
