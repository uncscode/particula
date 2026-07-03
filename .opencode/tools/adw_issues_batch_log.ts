import { tool } from "@opencode-ai/plugin";

import {
  STATUS_VALUES,
  buildError,
  isProvidedValue,
  normalizeAndValidateAdwId,
  normalizeIssue,
  normalizeReviewNote,
  normalizeReviewer,
  parseBatchOptions,
  runBatchCommandText,
} from "./adw_issues_spec_shared";

// --- Tool implementation ---

async function executeBatchLog(args: Record<string, any>): Promise<string> {
  const { adw_id, issue, reviewer, status, note, options } = args;
  if (!isProvidedValue(adw_id)) return buildError("'adw_id' is required for all commands except batch-init.");
  if (!isProvidedValue(issue)) return buildError("batch-log requires 'issue'.");
  const parsedOptions = parseBatchOptions("batch-log", options);
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
    "log",
    "--adw-id",
    normalizedAdw.value,
    "--issue",
    normalizedIssue,
  ];
  if (parsedOptions.options.read) {
    cmdParts.push("--read");
    return runBatchCommandText("batch-log", cmdParts);
  }

  if (!isProvidedValue(reviewer)) return buildError("batch-log requires 'reviewer' when options is not 'read'.");
  if (!isProvidedValue(status)) return buildError("batch-log requires 'status' when options is not 'read'.");
  if (!STATUS_VALUES.includes(status)) {
    return buildError(`Invalid status "${status}". Valid values: PASS, REVISED.`);
  }
  const normalizedReviewer = normalizeReviewer(reviewer);
  if (!normalizedReviewer) {
    return buildError("Invalid reviewer. Use a non-empty reviewer without control characters.");
  }
  cmdParts.push("--reviewer", normalizedReviewer, "--status", status);

  const normalizedNote = normalizeReviewNote(note);
  if (typeof note === "string" && note.trim().length > 0 && !normalizedNote) {
    return buildError("Invalid note. Use a non-empty note without control characters.");
  }
  if (normalizedNote) cmdParts.push("--note", normalizedNote);
  return runBatchCommandText("batch-log", cmdParts);
}

export default tool({
  description: "Read/append review log via adw spec batch log. Use options: \"read\" for read mode.",
  args: {
    adw_id: tool.schema.string(),
    issue: tool.schema.string(),
    reviewer: tool.schema.string().optional(),
    status: tool.schema.enum([...STATUS_VALUES]).optional(),
    note: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    return executeBatchLog(args as Record<string, any>);
  },
});
