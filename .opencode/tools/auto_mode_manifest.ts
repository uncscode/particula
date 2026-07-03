/**
 * ADW Auto-Mode Manifest Tool for OpenCode Integration.
 *
 * Provides structured access to `adw auto-mode` CLI commands with validation
 * and consistent error messaging.
 */

import { tool } from "@opencode-ai/plugin";

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;

function normalizeAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

const COMMANDS = ["init-from-batch", "init", "status", "validate", "reset", "complete"] as const;
const MAX_ISSUES = 500;
const MAX_DEPENDS = 500;
const MAX_BRANCH_LEN = 255;
const BRANCH_REF_PATTERN = /^[A-Za-z0-9._/\-]+$/;
const CONTROL_CHARS_PATTERN = /[\x00-\x1F\x7F]/;
const BOUNDED_OPTION_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;

const COMMAND_OPTION_TOKENS = {
  "init-from-batch": ["force"],
  init: ["force"],
  status: ["json"],
  validate: [],
  reset: ["resume", "force"],
  complete: ["force", "dry-run", "branch-merged", "no-branch-merged"],
} as const satisfies Record<AutoModeCommand, readonly string[]>;

const USAGE_EXAMPLE = `Example usage:
  auto_mode_manifest({ command: "init-from-batch", adw_id: "abc12345", source_branch: "epic/e14-auto", target_branch: "develop", branch_type: "epic", ship_strategy: "accumulate", options: "force" })
  auto_mode_manifest({ command: "init", issues: "42,43", depends: "43:42", ship_strategy: "pr", options: "force" })
  auto_mode_manifest({ command: "status", branch: "epic/e14-auto", options: "json" })
  auto_mode_manifest({ command: "validate", branch: "epic/e14-auto" })
  auto_mode_manifest({ command: "reset", issue: "42", branch: "epic/e14-auto", options: "resume force" })
  auto_mode_manifest({ command: "complete", issue: "42", adw_id: "abc12345", branch: "epic/e14-auto", completed_at: "2026-06-27T23:59:59Z", detail: "Issue completed (branch accumulation).", options: "branch-merged dry-run" })`;

const COMMAND_DESCRIPTIONS = `AVAILABLE COMMANDS:
• init-from-batch: Initialize manifest from batch state
  Usage: { command: "init-from-batch", adw_id: "abc12345", source_branch?: "epic/e14-auto", target_branch?: "develop", branch_type?: "epic", segment_size?: 3, ship_strategy?: "accumulate", options?: "force" }

• init: Initialize manifest from issues list
  Usage: { command: "init", issues: "42,43", depends?: "43:42,44:43", source_branch?: "epic/e14-auto", target_branch?: "develop", branch_type?: "epic", segment_size?: 3, ship_strategy?: "pr", options?: "force" }

• status: Show current manifest state
  Usage: { command: "status", branch?: "epic/e14-auto", options?: "json" }

• validate: Validate current manifest state
  Usage: { command: "validate", branch?: "epic/e14-auto" }

• reset: Reset manifest issue state
  Usage: { command: "reset", issue: "42", branch?: "epic/e14-auto", options?: "resume force" }

• complete: Mark an issue completed for accumulate-mode handoff
  Usage: { command: "complete", issue: "42", adw_id: "abc12345", branch?: "epic/e14-auto", completed_at?: "2026-06-27T23:59:59Z", detail?: "Issue completed (branch accumulation).", options?: "force dry-run branch-merged" }
  Note: completion requires workflow context for the target manifest issue because adw_id must match the persisted issue record.`;

type AutoModeCommand = (typeof COMMANDS)[number];
type AutoModeOptionToken =
  | "force"
  | "json"
  | "resume"
  | "dry-run"
  | "branch-merged"
  | "no-branch-merged";
type ParsedCommandOptions = {
  force?: true;
  json?: true;
  resume?: true;
  "dry-run"?: true;
  "branch-merged"?: true;
  "no-branch-merged"?: true;
};

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE_EXAMPLE}`;
}

function normalizePositiveInt(value: string | number): string | null {
  const normalized = String(value ?? "").trim();
  if (!/^[1-9]\d*$/.test(normalized)) {
    return null;
  }
  return normalized;
}

function normalizeNonNegativeInt(value: string | number): string | null {
  const normalized = String(value ?? "").trim();
  if (!/^\d+$/.test(normalized)) {
    return null;
  }
  return normalized;
}

function validateBranchRef(branchName: string): string | null {
  if (!branchName) {
    return "Branch name is required";
  }
  if (branchName.startsWith("-")) {
    return `Invalid branch name: ${branchName}`;
  }
  if (branchName.length > MAX_BRANCH_LEN) {
    return `Branch name is too long (max ${MAX_BRANCH_LEN} chars).`;
  }
  if (CONTROL_CHARS_PATTERN.test(branchName)) {
    return `Invalid branch name: ${branchName}`;
  }
  if (!BRANCH_REF_PATTERN.test(branchName)) {
    return `Invalid branch name: ${branchName}`;
  }
  if (branchName.startsWith("/") || branchName.endsWith("/")) {
    return `Invalid branch name: ${branchName}`;
  }
  if (branchName.includes("//")) {
    return `Invalid branch name: ${branchName}`;
  }
  if (branchName.includes("@{")) {
    return `Invalid branch name: ${branchName}`;
  }
  const parts = branchName.split("/");
  if (parts.some((part) => part === "." || part === "..")) {
    return `Invalid branch name: ${branchName}`;
  }
  if (parts.some((part) => part.endsWith(".lock"))) {
    return `Invalid branch name: ${branchName}`;
  }
  return null;
}

function normalizeBranchInput(value: string | undefined): {
  normalized: string | null;
  error?: string;
} {
  if (value === undefined) {
    return { normalized: null };
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return { normalized: null };
  }
  const error = validateBranchRef(trimmed);
  if (error) {
    return { normalized: null, error };
  }
  return { normalized: trimmed };
}

function normalizeBranchType(
  value: string | undefined,
): "epic" | "feature" | "maintenance" | null {
  if (value === undefined) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const normalized = trimmed.toLowerCase();
  if (normalized === "epic" || normalized === "feature" || normalized === "maintenance") {
    return normalized;
  }
  return null;
}

function normalizeShipStrategy(
  value: string | undefined,
): "pr" | "accumulate" | null {
  if (value === undefined) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const normalized = trimmed.toLowerCase();
  if (normalized === "pr" || normalized === "accumulate") {
    return normalized;
  }
  return null;
}

function isCommand(command: string): command is AutoModeCommand {
  return COMMANDS.includes(command as AutoModeCommand);
}

function buildCliError(output: string): string {
  return `ERROR: adw auto-mode command failed.\n${output}\n\n${USAGE_EXAMPLE}`;
}

function parseCommandOptions(
  command: AutoModeCommand,
  rawOptions: unknown,
): { values?: ParsedCommandOptions; error?: string } {
  if (rawOptions === undefined) {
    return { values: {} };
  }
  if (rawOptions !== undefined && rawOptions !== null && typeof rawOptions !== "string") {
    return {
      error: "ERROR: 'options' must be a string when provided.",
    };
  }
  if (rawOptions === null) {
    return { values: {} };
  }

  const trimmed = rawOptions.trim();
  if (!trimmed) {
    return { values: {} };
  }

  const allowedTokens = COMMAND_OPTION_TOKENS[command];
  const parsedValues: ParsedCommandOptions = {};

  for (const token of trimmed.split(/\s+/)) {
    const separatorCount = token.split("=").length - 1;
    if (separatorCount > 0) {
      return {
        error: `ERROR: Invalid options token '${token}' for '${command}': bare tokens only; '=value' is not supported.`,
      };
    }
    if (!BOUNDED_OPTION_NAME_PATTERN.test(token)) {
      return {
        error: `ERROR: Invalid options token '${token}' for '${command}': token names must use lowercase-kebab-case.`,
      };
    }
    if (!allowedTokens.includes(token)) {
      return {
        error: `ERROR: Invalid options token '${token}' for '${command}': token is not allowed for command '${command}'.`,
      };
    }

    const fieldName = token as AutoModeOptionToken;
    if (parsedValues[fieldName]) {
      return {
        error: `ERROR: Invalid options token '${token}' for '${command}': duplicate '${token}' token is not allowed.`,
      };
    }
    parsedValues[fieldName] = true;
  }

  return { values: parsedValues };
}

function normalizeIssues(issues: string): { csv: string; items: string[] } | null {
  const trimmed = issues.trim();
  if (!trimmed) {
    return null;
  }
  const parts = trimmed.split(",").map((part) => part.trim());
  if (parts.some((part) => part.length === 0)) {
    return null;
  }
  const normalized = parts.map((part) => normalizePositiveInt(part));
  if (normalized.some((value) => value === null)) {
    return null;
  }
  return { csv: (normalized as string[]).join(","), items: normalized as string[] };
}

function normalizeDepends(depends: string): string[] | null {
  const trimmed = depends.trim();
  if (!trimmed) {
    return null;
  }
  const parts = trimmed.split(",").map((part) => part.trim());
  if (parts.some((part) => part.length === 0)) {
    return null;
  }
  const normalizedPairs: string[] = [];
  for (const part of parts) {
    const pairParts = part.split(":");
    if (pairParts.length !== 2) {
      return null;
    }
    const [left, right] = pairParts;
    if (!left || !right) {
      return null;
    }
    const leftNormalized = normalizePositiveInt(left);
    const rightNormalized = normalizePositiveInt(right);
    if (!leftNormalized || !rightNormalized) {
      return null;
    }
    normalizedPairs.push(`${leftNormalized}:${rightNormalized}`);
  }
  return normalizedPairs;
}

function normalizeOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const trimmed = String(value).trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function validateIssueBounds(items: string[]): string | null {
  if (items.length > MAX_ISSUES) {
    return `Too many issues (${items.length}). Maximum is ${MAX_ISSUES}.`;
  }
  return null;
}

function validateDependsBounds(pairs: string[]): string | null {
  if (pairs.length > MAX_DEPENDS) {
    return `Too many dependency pairs (${pairs.length}). Maximum is ${MAX_DEPENDS}.`;
  }
  return null;
}

function appendBranchMetadata(
  cmdParts: (string | number)[],
  args: {
    source_branch?: string;
    target_branch?: string;
    branch_type?: string;
  },
): string | null {
  const normalizedSourceBranch = normalizeBranchInput(args.source_branch);
  if (normalizedSourceBranch.error) {
    return normalizedSourceBranch.error;
  }
  if (normalizedSourceBranch.normalized) {
    cmdParts.push("--source-branch", normalizedSourceBranch.normalized);
  }

  const normalizedTargetBranch = normalizeBranchInput(args.target_branch);
  if (normalizedTargetBranch.error) {
    return normalizedTargetBranch.error;
  }
  if (normalizedTargetBranch.normalized) {
    cmdParts.push("--target-branch", normalizedTargetBranch.normalized);
  }

  const normalizedBranchType = normalizeBranchType(args.branch_type);
  if (args.branch_type !== undefined) {
    if (args.branch_type.trim() && !normalizedBranchType) {
      return `Invalid branch_type "${args.branch_type}". Use: epic, feature, maintenance.`;
    }
    if (normalizedBranchType) {
      cmdParts.push("--branch-type", normalizedBranchType);
    }
  }
  return null;
}

function appendBranchFilter(cmdParts: (string | number)[], branch?: string): string | null {
  const normalizedBranch = normalizeBranchInput(branch);
  if (normalizedBranch.error) {
    return normalizedBranch.error;
  }
  if (normalizedBranch.normalized) {
    cmdParts.push("--branch", normalizedBranch.normalized);
  }
  return null;
}

function appendSegmentSize(
  cmdParts: (string | number)[],
  segment_size: string | number | undefined,
): string | null {
  if (segment_size === undefined || segment_size === null) {
    return null;
  }
  // Treat blank/whitespace-only strings as omitted (sparse-call parity)
  const asString = String(segment_size).trim();
  if (asString === "") {
    return null;
  }
  const normalizedSegment = normalizeNonNegativeInt(segment_size);
  if (!normalizedSegment) {
    return `Invalid segment_size "${segment_size}". Must be a non-negative integer.`;
  }
  cmdParts.push("--segment-size", normalizedSegment);
  return null;
}

function appendShipStrategy(
  cmdParts: (string | number)[],
  ship_strategy: string | undefined,
): string | null {
  if (ship_strategy === undefined) {
    return null;
  }
  const normalized = normalizeShipStrategy(ship_strategy);
  if (ship_strategy.trim() && !normalized) {
    return `Invalid ship_strategy "${ship_strategy}". Use: pr, accumulate.`;
  }
  if (normalized) {
    cmdParts.push("--ship-strategy", normalized);
  }
  return null;
}

function buildInitFromBatchCommand(
  cmdParts: (string | number)[],
  args: AutoModeManifestArgs,
  options: ParsedCommandOptions,
): string | null {
  if (!args.adw_id) {
    return "'adw_id' is required for init-from-batch.";
  }
  const normalizedAdwId = normalizeAdwId(args.adw_id);
  if (!normalizedAdwId) {
    return `Invalid adw_id "${args.adw_id}". Must be an 8-character hex string (e.g., "abc12345").`;
  }
  cmdParts.push("init-from-batch", "--adw-id", normalizedAdwId);

  const branchError = appendBranchMetadata(cmdParts, args);
  if (branchError) {
    return branchError;
  }

  const segmentError = appendSegmentSize(cmdParts, args.segment_size);
  if (segmentError) {
    return segmentError;
  }
  const shipError = appendShipStrategy(cmdParts, args.ship_strategy);
  if (shipError) {
    return shipError;
  }
  if (options.force) {
    cmdParts.push("--force");
  }
  return null;
}

function buildInitCommand(
  cmdParts: (string | number)[],
  args: AutoModeManifestArgs,
  options: ParsedCommandOptions,
): string | null {
  if (!args.issues) {
    return "'issues' is required for init.";
  }
  const normalizedIssues = normalizeIssues(args.issues);
  if (!normalizedIssues) {
    return `Invalid issues "${args.issues}". Provide a comma-separated list of positive integers.`;
  }
  const issueBoundsError = validateIssueBounds(normalizedIssues.items);
  if (issueBoundsError) {
    return issueBoundsError;
  }
  cmdParts.push("init", "--issues", normalizedIssues.csv);

  const depends = normalizeOptionalString(args.depends);
  if (depends !== undefined) {
    const normalizedDepends = normalizeDepends(depends);
    if (!normalizedDepends) {
      return `Invalid depends "${depends}". Provide comma-separated A:B pairs.`;
    }
    const dependsBoundsError = validateDependsBounds(normalizedDepends);
    if (dependsBoundsError) {
      return dependsBoundsError;
    }
    for (const pair of normalizedDepends) {
      cmdParts.push("--depends", pair);
    }
  }

  const branchError = appendBranchMetadata(cmdParts, args);
  if (branchError) {
    return branchError;
  }

  const segmentError = appendSegmentSize(cmdParts, args.segment_size);
  if (segmentError) {
    return segmentError;
  }
  const shipError = appendShipStrategy(cmdParts, args.ship_strategy);
  if (shipError) {
    return shipError;
  }
  if (options.force) {
    cmdParts.push("--force");
  }
  return null;
}

function buildStatusCommand(
  cmdParts: (string | number)[],
  args: AutoModeManifestArgs,
  options: ParsedCommandOptions,
): string | null {
  cmdParts.push("status");
  const branchError = appendBranchFilter(cmdParts, args.branch);
  if (branchError) {
    return branchError;
  }
  if (options.json) {
    cmdParts.push("--json");
  }
  return null;
}

function buildValidateCommand(
  cmdParts: (string | number)[],
  args: AutoModeManifestArgs,
): string | null {
  cmdParts.push("validate");
  const branchError = appendBranchFilter(cmdParts, args.branch);
  if (branchError) {
    return branchError;
  }
  return null;
}

function buildResetCommand(
  cmdParts: (string | number)[],
  args: AutoModeManifestArgs,
  options: ParsedCommandOptions,
): string | null {
  if (!args.issue) {
    return "'issue' is required for reset.";
  }
  const normalizedIssue = normalizePositiveInt(args.issue);
  if (!normalizedIssue) {
    return `Invalid issue "${args.issue}". Issue must be a positive integer.`;
  }
  cmdParts.push("reset", "--issue", normalizedIssue);
  const branchError = appendBranchFilter(cmdParts, args.branch);
  if (branchError) {
    return branchError;
  }
  if (options.resume) {
    cmdParts.push("--resume");
  }
  if (options.force) {
    cmdParts.push("--force");
  }
  return null;
}

function buildCompleteCommand(
  cmdParts: (string | number)[],
  args: AutoModeManifestArgs,
  options: ParsedCommandOptions,
): string | null {
  if (!args.issue) {
    return "'issue' is required for complete.";
  }
  const normalizedIssue = normalizePositiveInt(args.issue);
  if (!normalizedIssue) {
    return `Invalid issue "${args.issue}". Issue must be a positive integer.`;
  }
  if (!args.adw_id) {
    return "'adw_id' is required for complete.";
  }
  const normalizedAdwId = normalizeAdwId(args.adw_id);
  if (!normalizedAdwId) {
    return `Invalid adw_id "${args.adw_id}". Must be an 8-character hex string (e.g., "abc12345").`;
  }
  if (options["branch-merged"] && options["no-branch-merged"]) {
    return "Invalid options for 'complete': 'branch-merged' and 'no-branch-merged' cannot be combined.";
  }
  cmdParts.push("complete", "--issue", normalizedIssue, "--adw-id", normalizedAdwId);
  const branchError = appendBranchFilter(cmdParts, args.branch);
  if (branchError) {
    return branchError;
  }
  const completedAt = normalizeOptionalString(args.completed_at);
  if (completedAt !== undefined) {
    cmdParts.push("--completed-at", completedAt);
  }
  const detail = normalizeOptionalString(args.detail);
  if (detail !== undefined) {
    cmdParts.push("--detail", detail);
  }
  if (options.force) {
    cmdParts.push("--force");
  }
  if (options["dry-run"]) {
    cmdParts.push("--dry-run");
  }
  if (options["branch-merged"]) {
    cmdParts.push("--branch-merged");
  }
  if (options["no-branch-merged"]) {
    cmdParts.push("--no-branch-merged");
  }
  return null;
}

export default tool({
  description: `Manage ADW auto-mode manifest operations via \`adw auto-mode\`.

This tool wraps the auto-mode CLI commands with validation to provide agents
safe access to manifest creation, inspection, resets, and manual completion.

${COMMAND_DESCRIPTIONS}

NOTES:
• \`issues\` is a comma-separated list of positive integers.
• \`depends\` is a comma-separated list of A:B pairs (e.g., "43:42,44:43").
• Sparse-call rule: omit optional fields unless intentionally set; blank optional strings are treated as omitted.
• Bounded options use command-scoped bare tokens only (e.g., options: "force" or options: "resume force").
• Errors return with an ERROR: prefix and include a usage example.

${USAGE_EXAMPLE}`,

  args: {
    command: tool.schema
      .enum([...COMMANDS])
      .describe(`Auto-mode command to execute.

${COMMAND_DESCRIPTIONS}

REQUIRED PARAMETERS BY COMMAND:
• init-from-batch: adw_id
• init: issues
• status: none
• validate: none
• reset: issue
• complete: issue, adw_id`),

    adw_id: tool.schema
      .string()
      .optional()
      .describe(`ADW workflow ID.

REQUIRED FOR: init-from-batch, complete
For complete, this must match the persisted workflow context recorded on the
target manifest issue.
EXAMPLE: adw_id: "abc12345"`),

    issues: tool.schema
      .string()
      .optional()
      .describe(`Comma-separated list of issue numbers.

REQUIRED FOR: init
EXAMPLE: issues: "42,43,44"`),

    depends: tool.schema
      .string()
      .optional()
      .describe(`Comma-separated dependency pairs (A:B).

OPTIONAL FOR: init
Blank strings are treated as omitted.
EXAMPLE: depends: "43:42,44:43"`),

    segment_size: tool.schema
      .string()
      .or(tool.schema.number())
      .optional()
      .describe(`Segment size for auto-mode scheduling (allows 0). Accepts a number or numeric string.

OPTIONAL FOR: init-from-batch, init
EXAMPLE: segment_size: 0`),

    options: tool.schema
      .string()
      .optional()
      .describe(`Bounded command-scoped toggle carrier.

SUPPORTED TOKENS:
• init-from-batch: force
• init: force
• status: json
• reset: resume, force
• complete: force, dry-run, branch-merged, no-branch-merged

Use space-separated bare tokens and keep payload-bearing fields explicit.

EXAMPLES:
• options: "force"
• options: "json"
• options: "resume force"`),

    source_branch: tool.schema
      .string()
      .optional()
      .describe(`Branch name for the source manifest.

OPTIONAL FOR: init-from-batch, init
MAPS TO: --source-branch`),

    target_branch: tool.schema
      .string()
      .optional()
      .describe(`Branch name for the target manifest.

OPTIONAL FOR: init-from-batch, init
MAPS TO: --target-branch`),

    branch_type: tool.schema
      .string()
      .optional()
      .describe(`Branch type metadata for the manifest.

OPTIONAL FOR: init-from-batch, init
MAPS TO: --branch-type
ALLOWED: epic, feature, maintenance`),

    ship_strategy: tool.schema
      .string()
      .optional()
      .describe(`Ship strategy for the manifest. Controls how completed issues are delivered.

- "pr": Each issue opens its own PR (default).
- "accumulate": Issues rebase onto a tracking branch; one final PR at completion.

OPTIONAL FOR: init-from-batch, init
MAPS TO: --ship-strategy
ALLOWED: pr, accumulate`),

    issue: tool.schema
      .string()
      .optional()
      .describe(`Issue number as a positive integer string.

        REQUIRED FOR: reset, complete
        EXAMPLE: issue: "42"`),

    branch: tool.schema
      .string()
      .optional()
      .describe(`Branch name filter for manifest operations.

APPLIES TO: status, validate, reset, complete
MAPS TO: --branch`),

    completed_at: tool.schema
      .string()
      .optional()
      .describe(`UTC ISO 8601 completion timestamp.

OPTIONAL FOR: complete
MAPS TO: --completed-at`),

    detail: tool.schema
      .string()
      .optional()
      .describe(`Checkpoint detail text.

OPTIONAL FOR: complete
MAPS TO: --detail`),
  },

  async execute(args: AutoModeManifestArgs) {
    const { command } = args;
    if (!isCommand(command)) {
      return buildError(
        `Invalid command "${command}". Valid commands: ${COMMANDS.join(", ")}.`,
      );
    }

    const parsedOptions = parseCommandOptions(command, args.options);
    if (parsedOptions.error) {
      return `${parsedOptions.error}\n\n${USAGE_EXAMPLE}`;
    }

    const cmdParts: (string | number)[] = ["uv", "run", "--active", "adw", "auto-mode"];
    let buildErrorMessage: string | null = null;
    const optionValues = parsedOptions.values ?? {};

    switch (command) {
      case "init-from-batch": {
        buildErrorMessage = buildInitFromBatchCommand(cmdParts, args, optionValues);
        break;
      }

      case "init": {
        buildErrorMessage = buildInitCommand(cmdParts, args, optionValues);
        break;
      }

      case "status": {
        buildErrorMessage = buildStatusCommand(cmdParts, args, optionValues);
        break;
      }

      case "validate": {
        buildErrorMessage = buildValidateCommand(cmdParts, args);
        break;
      }

      case "reset": {
        buildErrorMessage = buildResetCommand(cmdParts, args, optionValues);
        break;
      }

      case "complete": {
        buildErrorMessage = buildCompleteCommand(cmdParts, args, optionValues);
        break;
      }
    }

    if (buildErrorMessage) {
      return buildError(buildErrorMessage);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      if (result.includes("ERROR:") || result.includes("Error:")) {
        return buildCliError(result);
      }
      return result || "adw auto-mode completed but returned no output.";
    } catch (err: unknown) {
      const errorData = err && typeof err === "object" ? (err as Record<string, unknown>) : {};
      const errorOutput = errorData.stdout ? String(errorData.stdout) : "";
      const errorMsg = errorData.stderr ? String(errorData.stderr) : "";
      const errorMessage = errorMsg || (errorData.message ? String(errorData.message) : "");

      if (errorOutput && (errorOutput.includes("ERROR") || errorOutput.includes("Error"))) {
        return buildCliError(errorOutput);
      }

      if (errorMessage || errorOutput) {
        return (
          `ERROR: Failed to run adw auto-mode command.\n${errorMessage || ""}` +
          `${errorOutput ? `\n\nOutput:\n${errorOutput}` : ""}` +
          `\n\n${USAGE_EXAMPLE}`
        );
      }

      return buildCliError(err instanceof Error ? err.message : String(err));
    }
  },
});

interface AutoModeManifestArgs {
  command: string;
  adw_id?: string;
  issues?: string;
  depends?: string;
  segment_size?: string | number;
  options?: string;
  source_branch?: string;
  target_branch?: string;
  branch_type?: string;
  ship_strategy?: string;
  issue?: string;
  branch?: string;
  completed_at?: string;
  detail?: string;
}
