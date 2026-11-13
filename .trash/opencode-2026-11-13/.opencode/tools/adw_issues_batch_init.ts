import { tool } from "@opencode-ai/plugin";

import {
  MAX_TOTAL,
  buildError,
  normalizeAndValidateAdwId,
  normalizeSafeRelativeSourcePath,
  normalizeTotal,
  parseBatchOptions,
  runBatchCommandSpawn,
} from "./adw_issues_spec_shared";

// --- Tool implementation ---

async function executeBatchInit(args: Record<string, any>): Promise<string> {
  const { total, source, adw_id, options } = args;
  if (!(typeof total === "string" ? total.trim() : total)) {
    return buildError("batch-init requires 'total' and 'source'.");
  }
  if (!(typeof source === "string" && source.trim().length > 0)) {
    return buildError("batch-init requires 'total' and 'source'.");
  }
  const normalizedSource = normalizeSafeRelativeSourcePath(source);
  if (!normalizedSource) {
    return buildError("Invalid source path. Use a safe, non-empty relative path without control characters or traversal segments.");
  }
  const normalizedTotal = normalizeTotal(total);
  if (!normalizedTotal) {
    return buildError(`Invalid total "${total}". Must be between 1 and ${MAX_TOTAL}.`);
  }
  const parsedOptions = parseBatchOptions("batch-init", options);
  if (!parsedOptions.ok) return parsedOptions.error;
  const normalized = normalizeAndValidateAdwId(adw_id);
  if (!normalized.ok) return normalized.error;
  const cmdParts: (string | number)[] = [
    "uv",
    "run",
    "--active",
    "adw",
    "spec",
    "batch",
    "init",
    "--total",
    normalizedTotal,
    "--source",
    normalizedSource,
  ];
  if (normalized.value) cmdParts.push("--adw-id", normalized.value);
  return runBatchCommandSpawn("batch-init", cmdParts);
}

export default tool({
  description: "Initialize batch content via adw spec batch init.",
  args: {
    total: tool.schema.string(),
    source: tool.schema.string(),
    adw_id: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    return executeBatchInit(args as Record<string, any>);
  },
});
