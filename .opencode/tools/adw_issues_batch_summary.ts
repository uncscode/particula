import { tool } from "@opencode-ai/plugin";

import {
  buildError,
  isProvidedValue,
  normalizeAndValidateAdwId,
  parseBatchOptions,
  runBatchCommandText,
} from "./adw_issues_spec_shared";

// --- Tool implementation ---

async function executeBatchSummary(args: Record<string, any>): Promise<string> {
  const { adw_id, options } = args;
  if (!isProvidedValue(adw_id)) return buildError("'adw_id' is required for all commands except batch-init.");
  const parsedOptions = parseBatchOptions("batch-summary", options);
  if (!parsedOptions.ok) return parsedOptions.error;
  const normalizedAdw = normalizeAndValidateAdwId(adw_id);
  if (!normalizedAdw.ok || !normalizedAdw.value) return normalizedAdw.ok ? buildError("'adw_id' is required for all commands except batch-init.") : normalizedAdw.error;
  return runBatchCommandText("batch-summary", [
    "uv",
    "run",
    "--active",
    "adw",
    "spec",
    "batch",
    "summary",
    "--adw-id",
    normalizedAdw.value,
  ]);
}

export default tool({
  description: "Read batch summary via adw spec batch summary.",
  args: { adw_id: tool.schema.string(), options: tool.schema.string().optional() },
  async execute(args) {
    return executeBatchSummary(args as Record<string, any>);
  },
});
