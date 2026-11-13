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

async function executeBatchRead(args: Record<string, any>): Promise<string> {
  const { adw_id, issue, section, options } = args;
  if (!isProvidedValue(adw_id)) {
    return buildError("'adw_id' is required for all commands except batch-init.");
  }
  const normalizedAdw = normalizeAndValidateAdwId(adw_id);
  if (!normalizedAdw.ok || !normalizedAdw.value) return normalizedAdw.ok ? buildError("'adw_id' is required for all commands except batch-init.") : normalizedAdw.error;
  const parsedOptions = parseBatchOptions("batch-read", options);
  if (!parsedOptions.ok) return parsedOptions.error;

  const cmdParts: (string | number)[] = ["uv", "run", "--active", "adw", "spec", "batch", "read", "--adw-id", normalizedAdw.value];
  if (issue !== undefined && issue !== null && String(issue).trim() !== "") {
    const normalizedIssue = normalizeIssue(issue);
    if (!normalizedIssue) return buildError(`Invalid issue "${issue}". Issue must be a positive integer.`);
    cmdParts.push("--issue", normalizedIssue);
  }
  if (section !== undefined && section !== null && String(section).trim() !== "") {
    const normalizedSection = normalizeSectionToken(section);
    if (!normalizedSection) {
      return buildError(
        `Invalid section token "${section}". Section must not contain control/newline characters.`,
      );
    }
    cmdParts.push("--section", normalizedSection);
  }
  if (parsedOptions.options.raw) cmdParts.push("--raw");
  return runBatchCommandText("batch-read", cmdParts);
}

export default tool({
  description: "Read batch content via adw spec batch read. Use options: \"raw\" for raw output.",
  args: {
    adw_id: tool.schema.string(),
    issue: tool.schema.string().optional(),
    section: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    return executeBatchRead(args as Record<string, any>);
  },
});
