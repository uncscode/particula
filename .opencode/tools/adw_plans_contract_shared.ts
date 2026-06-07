import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

/**
 * Function signature for wrapper-local deterministic error envelopes.
 */
export type BuildError = (message: string) => string;

/**
 * Required-argument metadata for a specific plans command.
 */
export type RequiredArgSpec = {
  field: string;
  message: string;
};

/**
 * Map of command names to their required-argument checks.
 */
export type RequiredArgMap = Record<string, RequiredArgSpec[]>;

export type ParsedOptionsStringValues = {
  json?: true;
  populate?: true;
  check?: true;
  status?: string;
  phase_status?: string;
  priority?: string;
  size?: string;
  after?: string;
  issue_number?: number;
  clear_issue_number?: true;
};

export type ParsedOptionsStringResult = {
  values?: ParsedOptionsStringValues;
  error?: string;
};

export type OptionsStringWrapper = "adw_plans" | "adw_plans_read" | "adw_plans_mutate";

export type OptionsStringDisposition = "token_candidate" | "retained_direct" | "direct_exception";

export type OptionsStringAuditEntry = {
  wrapper: OptionsStringWrapper;
  command: string;
  field: string;
  cliFlag: string;
  disposition: OptionsStringDisposition;
  reason: string;
};

export const ADW_PLANS_OPTION_STRING_AUDIT: readonly OptionsStringAuditEntry[] = [
  { wrapper: "adw_plans", command: "list", field: "plan_type", cliFlag: "--type", disposition: "retained_direct", reason: "Registry-driven plan type filtering stays a retained direct field in P1." },
  { wrapper: "adw_plans", command: "list", field: "lifecycle", cliFlag: "--lifecycle", disposition: "retained_direct", reason: "Lifecycle filtering stays a retained direct field to avoid expanding the token grammar in P1." },
  { wrapper: "adw_plans", command: "list", field: "parent", cliFlag: "--parent", disposition: "retained_direct", reason: "Parent identifiers remain retained direct routing/filter values." },
  { wrapper: "adw_plans", command: "list", field: "status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans", command: "list", field: "json", cliFlag: "--json", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans", command: "show", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "show", field: "json", cliFlag: "--json", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans", command: "create", field: "plan_type", cliFlag: "--type", disposition: "retained_direct", reason: "Plan types remain retained direct routing values for create/scaffold commands." },
  { wrapper: "adw_plans", command: "create", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans", command: "create", field: "plan_id", cliFlag: "--id", disposition: "retained_direct", reason: "Optional explicit IDs stay retained direct fields because they shape persisted plan identity." },
  { wrapper: "adw_plans", command: "create", field: "parent", cliFlag: "--parent", disposition: "retained_direct", reason: "Parent identifiers remain retained direct routing/filter values." },
  { wrapper: "adw_plans", command: "create", field: "priority", cliFlag: "--priority", disposition: "token_candidate", reason: "Bounded priority values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "create", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "create", field: "status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans", command: "update", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "update", field: "status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans", command: "update", field: "priority", cliFlag: "--priority", disposition: "token_candidate", reason: "Bounded priority values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "update", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "update", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans", command: "update", field: "patch", cliFlag: "--patch", disposition: "direct_exception", reason: "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules." },
  { wrapper: "adw_plans", command: "add-phase", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "add-phase", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans", command: "add-phase", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "add-phase", field: "phase_status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded phase status values support options-string phase-status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans", command: "add-phase", field: "after", cliFlag: "--after", disposition: "token_candidate", reason: "Single identifier references fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "update-phase", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "update-phase", field: "phase_id", cliFlag: "<phase_id>", disposition: "retained_direct", reason: "Phase identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "update-phase", field: "phase_status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded phase status values support options-string phase-status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans", command: "update-phase", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans", command: "update-phase", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "update-phase", field: "issue_number", cliFlag: "--issue", disposition: "token_candidate", reason: "Positive safe integer issue links fit command-scoped key=value tokens." },
  { wrapper: "adw_plans", command: "update-phase", field: "clear_issue_number", cliFlag: "--clear-issue-number", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans", command: "update-phase", field: "patch", cliFlag: "--patch", disposition: "direct_exception", reason: "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules." },
  { wrapper: "adw_plans", command: "schema", field: "check", cliFlag: "--check", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans", command: "scaffold-sections", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "scaffold-sections", field: "plan_type", cliFlag: "--type", disposition: "retained_direct", reason: "Plan types remain retained direct routing values for create/scaffold commands." },
  { wrapper: "adw_plans", command: "list-sections", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans", command: "list-sections", field: "json", cliFlag: "--json", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans", command: "list-sections", field: "populate", cliFlag: "--populate", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans", command: "list", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans", command: "create", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans", command: "show", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans", command: "update", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans", command: "validate", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans", command: "add-phase", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans", command: "schema", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans", command: "list-sections", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans", command: "update-phase", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans", command: "scaffold-sections", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans_read", command: "list", field: "plan_type", cliFlag: "--type", disposition: "retained_direct", reason: "Registry-driven plan type filtering stays a retained direct field in P1." },
  { wrapper: "adw_plans_read", command: "list", field: "lifecycle", cliFlag: "--lifecycle", disposition: "retained_direct", reason: "Lifecycle filtering stays a retained direct field to avoid expanding the token grammar in P1." },
  { wrapper: "adw_plans_read", command: "list", field: "parent", cliFlag: "--parent", disposition: "retained_direct", reason: "Parent identifiers remain retained direct routing/filter values." },
  { wrapper: "adw_plans_read", command: "list", field: "status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans_read", command: "list", field: "json", cliFlag: "--json", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans_read", command: "show", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_read", command: "show", field: "json", cliFlag: "--json", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans_read", command: "schema", field: "check", cliFlag: "--check", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans_read", command: "list-sections", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_read", command: "list-sections", field: "json", cliFlag: "--json", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans_read", command: "list-sections", field: "populate", cliFlag: "--populate", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans_read", command: "list", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans_read", command: "show", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans_read", command: "validate", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans_read", command: "schema", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans_read", command: "list-sections", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Worktree/repository scoping stays a retained direct field outside token parsing." },
  { wrapper: "adw_plans_mutate", command: "create", field: "plan_type", cliFlag: "--type", disposition: "retained_direct", reason: "Plan types remain retained direct routing values for create/scaffold commands." },
  { wrapper: "adw_plans_mutate", command: "create", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans_mutate", command: "create", field: "plan_id", cliFlag: "--id", disposition: "retained_direct", reason: "Optional explicit IDs stay retained direct fields because they shape persisted plan identity." },
  { wrapper: "adw_plans_mutate", command: "create", field: "parent", cliFlag: "--parent", disposition: "retained_direct", reason: "Parent identifiers remain retained direct routing/filter values." },
  { wrapper: "adw_plans_mutate", command: "create", field: "priority", cliFlag: "--priority", disposition: "token_candidate", reason: "Bounded priority values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "create", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "create", field: "status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans_mutate", command: "create", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans_mutate", command: "update", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_mutate", command: "update", field: "status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans_mutate", command: "update", field: "priority", cliFlag: "--priority", disposition: "token_candidate", reason: "Bounded priority values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "update", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "update", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans_mutate", command: "update", field: "patch", cliFlag: "--patch", disposition: "direct_exception", reason: "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules." },
  { wrapper: "adw_plans_mutate", command: "update", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans_mutate", command: "add-phase", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_mutate", command: "add-phase", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans_mutate", command: "add-phase", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "add-phase", field: "phase_status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded phase status values support options-string phase-status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans_mutate", command: "add-phase", field: "after", cliFlag: "--after", disposition: "token_candidate", reason: "Single identifier references fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "add-phase", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "phase_id", cliFlag: "<phase_id>", disposition: "retained_direct", reason: "Phase identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "phase_status", cliFlag: "--status", disposition: "token_candidate", reason: "Bounded phase status values support options-string phase-status=<value> tokens while remaining accepted as direct wrapper fields." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "title", cliFlag: "--title", disposition: "retained_direct", reason: "Free-form titles remain retained direct fields rather than entering token parsing." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "size", cliFlag: "--size", disposition: "token_candidate", reason: "Bounded size values fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "issue_number", cliFlag: "--issue", disposition: "token_candidate", reason: "Positive safe integer issue links fit command-scoped key=value tokens." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "clear_issue_number", cliFlag: "--clear-issue-number", disposition: "token_candidate", reason: "Simple boolean flag forwarding maps to a bare token." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "patch", cliFlag: "--patch", disposition: "direct_exception", reason: "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules." },
  { wrapper: "adw_plans_mutate", command: "update-phase", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
  { wrapper: "adw_plans_mutate", command: "scaffold-sections", field: "plan_id", cliFlag: "<plan_id>", disposition: "retained_direct", reason: "Plan identifiers remain retained direct routing values for targeted commands." },
  { wrapper: "adw_plans_mutate", command: "scaffold-sections", field: "plan_type", cliFlag: "--type", disposition: "retained_direct", reason: "Plan types remain retained direct routing values for create/scaffold commands." },
  { wrapper: "adw_plans_mutate", command: "scaffold-sections", field: "cwd", cliFlag: "--cwd", disposition: "retained_direct", reason: "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing." },
] as const;

export const ADW_PLANS_OPTION_STRING_RULES = {
  tokenBooleanFlags: ["json", "populate", "check", "clear-issue-number"],
  tokenKeyValueFields: ["status", "phase-status", "priority", "size", "after", "issue"],
  tokenSeparator: "ascii_whitespace",
  tokenNameStyle: "lowercase-kebab-case",
  tokenValueRule: "exactly one '=' with a non-empty value; status and phase-status may consume subsequent space-separated value words until the next valid bounded token candidate for the selected command",
  commandAllowlist: {
    list: ["json"],
    show: ["json"],
    create: ["status", "priority", "size"],
    update: ["status", "priority", "size"],
    "add-phase": ["phase-status", "size", "after"],
    "update-phase": ["phase-status", "size", "issue", "clear-issue-number"],
    schema: ["check"],
    "list-sections": ["json", "populate"],
    validate: [],
    "scaffold-sections": [],
  },
  retainedDirectFields: ["command", "plan_id", "plan_type", "phase_id", "title", "parent", "cwd", "lifecycle", "patch"],
  mutualExclusionRules: [
    "update-phase tokens 'issue=<n>' and 'clear-issue-number' are mutually exclusive and must fail closed when combined.",
  ],
  duplicateHandling: [
    "Repeated identical boolean tokens collapse to one effective flag.",
    "Repeated identical key/value tokens collapse to one effective value.",
    "Contradictory duplicates for the same key fail closed before subprocess spawn.",
  ],
  malformedTokenRules: [
    "Unknown token names fail closed.",
    "Missing values after '=' fail closed.",
    "Extra '=' characters in a token fail closed.",
    "issue values must parse as positive safe integers.",
    "update-phase must not combine 'issue=<n>' with 'clear-issue-number'.",
    "Tokens not allowlisted for the selected command fail closed.",
  ],
  patchExceptionReason:
    "Raw JSON patch payloads remain direct because whitespace, braces, and quotes exceed the bounded token grammar.",
  behaviorNeutralScope:
    "P2 enables bounded options-string parsing across compatibility and split wrappers while preserving direct-field compatibility and direct patch exceptions.",
} as const;

const TOKEN_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;
const STRICT_DECIMAL_INTEGER_PATTERN = /^[1-9][0-9]*$/;
const PLAN_STATUSES = ["Draft", "Proposed", "Ready", "In Progress", "Blocked", "Monitoring", "Shipped", "Cancelled", "Superseded"] as const;
const PLAN_PRIORITIES = ["P0", "P1", "P2", "P3", "Backlog"] as const;
const PLAN_SIZES = ["XS", "S", "M", "L", "XL", "XXL"] as const;
const PHASE_STATUSES = ["Not Started", "In Progress", "Blocked", "Shipped", "Cancelled"] as const;

function buildOptionsParseError(
  command: string,
  token: string,
  reason: string,
  buildError: BuildError,
): string {
  return buildError(`Invalid options token '${token}' for '${command}': ${reason}`);
}

function setParsedBooleanToken(
  values: ParsedOptionsStringValues,
  command: string,
  token: string,
  field: keyof ParsedOptionsStringValues,
  buildError: BuildError,
): string | undefined {
  const currentValue = values[field];
  if (currentValue === true) {
    return undefined;
  }
  if (currentValue !== undefined) {
    return buildOptionsParseError(command, token, `conflicting duplicate '${token}'`, buildError);
  }
  values[field] = true;
  return undefined;
}

function setParsedValueToken(
  values: ParsedOptionsStringValues,
  command: string,
  token: string,
  field: keyof ParsedOptionsStringValues,
  value: string | number,
  buildError: BuildError,
): string | undefined {
  const currentValue = values[field];
  if (currentValue === value) {
    return undefined;
  }
  if (currentValue !== undefined) {
    return buildOptionsParseError(command, token, `conflicting duplicate '${token}'`, buildError);
  }
  values[field] = value as never;
  return undefined;
}

function validateTokenLikeValue(
  command: string,
  token: string,
  fieldLabel: string,
  rawValue: string,
  buildError: BuildError,
): string | undefined {
  if (rawValue.startsWith("-")) {
    return buildOptionsParseError(
      command,
      token,
      `${fieldLabel} values must not start with '-' to avoid CLI option confusion`,
      buildError,
    );
  }
  return undefined;
}

function isPotentialBoundedTokenCandidate(
  token: string,
  allowedTokens: readonly string[],
): boolean {
  const separatorCount = token.split("=").length - 1;
  if (separatorCount > 1) {
    return false;
  }
  if (separatorCount === 0) {
    return (
      TOKEN_NAME_PATTERN.test(token)
      && allowedTokens.includes(token)
      && ADW_PLANS_OPTION_STRING_RULES.tokenBooleanFlags.includes(token as never)
    );
  }

  const [tokenName, rawValue] = token.split("=");
  return (
    TOKEN_NAME_PATTERN.test(tokenName)
    && allowedTokens.includes(tokenName)
    && (Boolean(rawValue) || rawValue === "")
  );
}

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as { mode?: number }).mode ?? 0) & S_IFMT) === S_IFDIR;
}

export function findCurrentRepositoryRoot(): string {
  let currentPath = realpathSync(process.cwd());
  while (true) {
    if (existsSync(path.join(currentPath, ".git"))) {
      return currentPath;
    }
    const parentPath = path.dirname(currentPath);
    if (parentPath === currentPath) {
      return realpathSync(process.cwd());
    }
    currentPath = parentPath;
  }
}

export function validatePlansCwdPath(cwdPath: string): string | undefined {
  const redactedPath = redactPathLikeText(cwdPath);
  if (!existsSync(cwdPath)) {
    return `ERROR: cwd path does not exist: ${redactedPath}`;
  }
  let stats;
  try {
    stats = statSync(cwdPath);
  } catch {
    return `ERROR: cwd path does not exist: ${redactedPath}`;
  }
  if (!isStatDirectory(stats)) {
    return `ERROR: cwd path is not a directory: ${redactedPath}`;
  }

  const canonical = realpathSync(cwdPath);
  const gitMetadataPath = `${canonical}/.git`;
  if (!existsSync(gitMetadataPath)) {
    return `ERROR: cwd path is not a repository/worktree root: ${redactedPath} (missing .git metadata at ${redactPathLikeText(gitMetadataPath)})`;
  }
  const repoRoot = findCurrentRepositoryRoot();
  if (canonical !== repoRoot) {
    return `ERROR: cwd path resolves outside repository root: ${redactedPath} (canonical: ${redactPathLikeText(canonical)})`;
  }
  return undefined;
}

export function mergeParsedOptionField<T>(
  directValue: T | undefined,
  parsedValue: T | undefined,
  field: string,
  buildError: BuildError,
): { value?: T; error?: string } {
  if (parsedValue === undefined) {
    return { value: directValue };
  }
  if (directValue === undefined) {
    return { value: parsedValue };
  }
  if (directValue === parsedValue) {
    return { value: directValue };
  }
  return {
    error: buildError(`'${field}' cannot conflict between direct input and options string.`),
  };
}

export function parseCommandOptionsString(
  command: string,
  options: unknown,
  buildError: BuildError,
): ParsedOptionsStringResult {
  if (options === undefined || options === null) {
    return {};
  }
  if (typeof options !== "string") {
    return { error: buildError("'options' must be a string when provided.") };
  }

  const normalizedOptions = options.trim();
  if (!normalizedOptions) {
    return {};
  }

  const allowedTokens = ADW_PLANS_OPTION_STRING_RULES.commandAllowlist[
    command as keyof typeof ADW_PLANS_OPTION_STRING_RULES.commandAllowlist
  ];
  if (!allowedTokens) {
    return { error: buildError(`Unsupported command '${command}' for options parsing.`) };
  }

  const values: ParsedOptionsStringValues = {};
  const rawTokens = normalizedOptions.split(/\s+/);
  for (let index = 0; index < rawTokens.length; index += 1) {
    let token = rawTokens[index];
    if (!token) {
      continue;
    }

    const separatorCount = token.split("=").length - 1;
    if (separatorCount > 1) {
      return {
        error: buildOptionsParseError(
          command,
          token,
          "tokens must contain at most one '=' separator",
          buildError,
        ),
      };
    }

    if (separatorCount === 0) {
      if (!TOKEN_NAME_PATTERN.test(token)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            "token names must use lowercase-kebab-case",
            buildError,
          ),
        };
      }
      if (!allowedTokens.includes(token as never)) {
        return {
          error: buildOptionsParseError(command, token, "token is not allowed for this command", buildError),
        };
      }
      if (!ADW_PLANS_OPTION_STRING_RULES.tokenBooleanFlags.includes(token as never)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            "token requires a non-empty '=value' suffix",
            buildError,
          ),
        };
      }

      if (token === "json") {
        const error = setParsedBooleanToken(values, command, token, "json", buildError);
        if (error) return { error };
        continue;
      }
      if (token === "populate") {
        const error = setParsedBooleanToken(values, command, token, "populate", buildError);
        if (error) return { error };
        continue;
      }
      if (token === "check") {
        const error = setParsedBooleanToken(values, command, token, "check", buildError);
        if (error) return { error };
        continue;
      }
      if (token === "clear-issue-number") {
        const error = setParsedBooleanToken(
          values,
          command,
          token,
          "clear_issue_number",
          buildError,
        );
        if (error) return { error };
      }
      continue;
    }

    const [tokenName, initialRawValue] = token.split("=");
    let rawValue = initialRawValue;
    if (!TOKEN_NAME_PATTERN.test(tokenName)) {
      return {
        error: buildOptionsParseError(
          command,
          token,
          "token names must use lowercase-kebab-case",
          buildError,
        ),
      };
    }
    if (!rawValue) {
      return {
        error: buildOptionsParseError(command, token, "token value must not be empty", buildError),
      };
    }
    if (!allowedTokens.includes(tokenName as never)) {
      return {
        error: buildOptionsParseError(command, token, "token is not allowed for this command", buildError),
      };
    }
    if (!ADW_PLANS_OPTION_STRING_RULES.tokenKeyValueFields.includes(tokenName as never)) {
      return {
        error: buildOptionsParseError(command, token, "token must be provided without '=value'", buildError),
      };
    }

    if (tokenName === "status" || tokenName === "phase-status") {
      while (
        index + 1 < rawTokens.length
        && !isPotentialBoundedTokenCandidate(rawTokens[index + 1], allowedTokens)
      ) {
        rawValue += ` ${rawTokens[index + 1]}`;
        token += ` ${rawTokens[index + 1]}`;
        index += 1;
      }
    }

    if (tokenName === "status") {
      const tokenLikeValueError = validateTokenLikeValue(
        command,
        token,
        "status",
        rawValue,
        buildError,
      );
      if (tokenLikeValueError) return { error: tokenLikeValueError };
      if (!(PLAN_STATUSES as readonly string[]).includes(rawValue)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            `status values must be one of: ${PLAN_STATUSES.join(", ")}`,
            buildError,
          ),
        };
      }
      const error = setParsedValueToken(values, command, token, "status", rawValue, buildError);
      if (error) return { error };
      continue;
    }

    if (tokenName === "phase-status") {
      const tokenLikeValueError = validateTokenLikeValue(
        command,
        token,
        "phase-status",
        rawValue,
        buildError,
      );
      if (tokenLikeValueError) return { error: tokenLikeValueError };
      if (!(PHASE_STATUSES as readonly string[]).includes(rawValue)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            `phase-status values must be one of: ${PHASE_STATUSES.join(", ")}`,
            buildError,
          ),
        };
      }
      const error = setParsedValueToken(values, command, token, "phase_status", rawValue, buildError);
      if (error) return { error };
      continue;
    }

    if (tokenName === "issue") {
      if (!STRICT_DECIMAL_INTEGER_PATTERN.test(rawValue)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            "issue values must be positive safe integers",
            buildError,
          ),
        };
      }
      const parsedIssueNumber = Number(rawValue);
      const error = setParsedValueToken(
        values,
        command,
        token,
        "issue_number",
        parsedIssueNumber,
        buildError,
      );
      if (error) return { error };
      continue;
    }

    if (tokenName === "priority") {
      const tokenLikeValueError = validateTokenLikeValue(
        command,
        token,
        "priority",
        rawValue,
        buildError,
      );
      if (tokenLikeValueError) return { error: tokenLikeValueError };
      if (!(PLAN_PRIORITIES as readonly string[]).includes(rawValue)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            `priority values must be one of: ${PLAN_PRIORITIES.join(", ")}`,
            buildError,
          ),
        };
      }
      const error = setParsedValueToken(values, command, token, "priority", rawValue, buildError);
      if (error) return { error };
      continue;
    }
    if (tokenName === "size") {
      const tokenLikeValueError = validateTokenLikeValue(
        command,
        token,
        "size",
        rawValue,
        buildError,
      );
      if (tokenLikeValueError) return { error: tokenLikeValueError };
      if (!(PLAN_SIZES as readonly string[]).includes(rawValue)) {
        return {
          error: buildOptionsParseError(
            command,
            token,
            `size values must be one of: ${PLAN_SIZES.join(", ")}`,
            buildError,
          ),
        };
      }
      const error = setParsedValueToken(values, command, token, "size", rawValue, buildError);
      if (error) return { error };
      continue;
    }
    if (tokenName === "after") {
      const tokenLikeValueError = validateTokenLikeValue(
        command,
        token,
        "after",
        rawValue,
        buildError,
      );
      if (tokenLikeValueError) return { error: tokenLikeValueError };
      const error = setParsedValueToken(values, command, token, "after", rawValue, buildError);
      if (error) return { error };
    }
  }

  if (values.issue_number !== undefined && values.clear_issue_number) {
    return {
      error: buildError(
        "'issue_number' and 'clear_issue_number' are mutually exclusive for update-phase.",
      ),
    };
  }

  return { values };
}

/**
 * Sanitized command-failure text and visibility state.
 */
export type SanitizedOutput = {
  text: string;
  hasVisibleContent: boolean;
};

type CommandFailureSources = {
  stderr?: string;
  stdout?: string;
  message?: string;
};

const OUTPUT_CHAR_LIMIT = 4_000;
const CONTROL_CHAR_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WINDOWS_ABSOLUTE_PATH_PATTERN = /[A-Za-z]:\\(?:[^\\\r\n"']+\\)*[^\\\r\n"']+/g;
const QUOTED_UNIX_ABSOLUTE_PATH_PATTERN = /(["'])(\/[^\r\n"']+)\1/g;
const UNIX_COLON_PATH_PATTERN = /(^|[\s(\[])(\/(?:[^:\r\n]|:(?!\s))+?)(?=:\s|:\d|$)/gm;
const UNIX_BARE_PATH_PATTERN = /(^|[\s(\[])(\/(?:[^\s)\]"']|\s+(?![-\w]+:))+)/gm;
const REDACTED_SECRET = "<redacted-secret>";
const SECRET_ASSIGNMENT_PATTERNS = [
  /\b(authorization\s*:\s*bearer\s+)([^\s]+)/gi,
  /\b((?:token|secret|password|passwd|api(?:_|-)?key|access(?:_|-)?token|refresh(?:_|-)?token)\s*[:=]\s*)("?)([^\s",']+)("?)/gi,
  /\b(gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]+)\b/g,
];

function redactAbsolutePaths(raw: string): string {
  return raw
    .replace(WINDOWS_ABSOLUTE_PATH_PATTERN, "<path>")
    .replace(QUOTED_UNIX_ABSOLUTE_PATH_PATTERN, "$1<path>$1")
    .replace(UNIX_COLON_PATH_PATTERN, (_, prefix) => `${prefix}<path>`)
    .replace(UNIX_BARE_PATH_PATTERN, (_, prefix) => `${prefix}<path>`);
}

function redactSecrets(raw: string): string {
  return SECRET_ASSIGNMENT_PATTERNS.reduce((output, pattern) => {
    if (pattern.global && pattern.source.includes("authorization")) {
      return output.replace(pattern, `$1${REDACTED_SECRET}`);
    }
    if (pattern.global && pattern.source.includes("token|secret|password")) {
      return output.replace(pattern, `$1$2${REDACTED_SECRET}$4`);
    }
    return output.replace(pattern, REDACTED_SECRET);
  }, raw);
}

export function redactPathLikeText(raw: string): string {
  const sanitized = sanitizeCommandFailureOutput(raw);
  return sanitized.hasVisibleContent ? sanitized.text : "<path>";
}

/**
 * Sanitize spawned-command diagnostics for the adw_plans wrapper family.
 *
 * Removes control characters, redacts absolute path-like substrings, and
 * preserves the existing bounded truncation contract.
 *
 * Args:
 *   raw: Raw diagnostic text from stderr/stdout/message.
 *
 * Returns:
 *   A normalized text payload plus a visibility flag.
 */
export function sanitizeCommandFailureOutput(raw: string): SanitizedOutput {
  if (!raw) {
    return { text: "", hasVisibleContent: false };
  }

  const normalized = raw.replace(CONTROL_CHAR_PATTERN, "");
  const redacted = redactSecrets(redactAbsolutePaths(normalized)).replace(
    /<path>(?:\s+<path>)+/g,
    "<path>",
  );
  if (!redacted) {
    return { text: "", hasVisibleContent: false };
  }

  const hasVisibleContent = Boolean(redacted.trim());
  let output = redacted;
  if (output.length > OUTPUT_CHAR_LIMIT) {
    const originalLength = output.length;
    output = output.slice(0, OUTPUT_CHAR_LIMIT);
    output += `\n...[output truncated to ${OUTPUT_CHAR_LIMIT} characters; original length ${originalLength}]`;
  }

  return { text: output, hasVisibleContent };
}

export function sanitizeSuccessOutput(raw: string): SanitizedOutput {
  if (!raw) {
    return { text: "", hasVisibleContent: false };
  }

  const normalized = raw.replace(CONTROL_CHAR_PATTERN, "");
  if (!normalized) {
    return { text: "", hasVisibleContent: false };
  }

  const hasVisibleContent = Boolean(normalized.trim());
  let output = normalized;
  if (output.length > OUTPUT_CHAR_LIMIT) {
    const originalLength = output.length;
    output = output.slice(0, OUTPUT_CHAR_LIMIT);
    output += `\n...[output truncated to ${OUTPUT_CHAR_LIMIT} characters; original length ${originalLength}]`;
  }

  return { text: output, hasVisibleContent };
}

/**
 * Select the canonical spawned-command diagnostic using fixed precedence.
 *
 * Args:
 *   sources: Candidate stderr/stdout/message strings.
 *   fallback: Fallback text when every source is empty.
 *
 * Returns:
 *   The selected sanitized diagnostic text.
 */
export function selectCommandFailureDiagnostic(
  sources: CommandFailureSources,
  fallback: string,
): string {
  const safeStderr = sanitizeCommandFailureOutput(sources.stderr ?? "");
  const safeStdout = sanitizeCommandFailureOutput(sources.stdout ?? "");
  const safeMessage = sanitizeCommandFailureOutput(sources.message ?? "");
  if (safeStderr.hasVisibleContent) {
    return safeStderr.text;
  }
  if (safeStdout.hasVisibleContent) {
    return safeStdout.text;
  }
  if (safeMessage.hasVisibleContent) {
    return safeMessage.text;
  }
  return fallback;
}

/**
 * Derive a bounded next-safe-action hint for recognized failure classes.
 *
 * Args:
 *   diagnostic: Selected sanitized diagnostic text.
 *
 * Returns:
 *   A hint line when the failure class is recognized; otherwise undefined.
 */
export function deriveCommandFailureHint(diagnostic: string): string | undefined {
  if (!diagnostic) {
    return undefined;
  }

  if (
    /(?:ENOENT|python3(?:\s*:)?\s+not found|uv(?:\s*:)?\s+not found|can't open file|No such file or directory|Cannot find module)/i.test(
      diagnostic,
    )
  ) {
    return "hint: verify the required runtime/tooling is installed and the backend script exists in this repository.";
  }

  if (/(?:\bcwd path does not exist\b|\bcwd path is not a directory\b|\bcwd path resolves outside repository root\b|\b--cwd\b)/i.test(diagnostic)) {
    return "hint: verify --cwd points to an existing in-repository repository/worktree root.";
  }

  return undefined;
}

/**
 * Build the shared adw_plans spawned-command failure envelope.
 *
 * Args:
 *   command: Plans subcommand name.
 *   reason: Stable failure reason segment.
 *   sources: Candidate stderr/stdout/message strings.
 *   fallback: Fallback text when diagnostics are empty.
 *
 * Returns:
 *   Deterministic ERROR envelope with optional recognized-action hint.
 */
export function buildCommandFailureError(
  command: string,
  reason: string,
  sources: CommandFailureSources,
  fallback: string,
): string {
  const diagnostic = selectCommandFailureDiagnostic(sources, fallback);
  const hint = deriveCommandFailureHint(diagnostic);
  const suffix = hint ? `\n${hint}` : "";
  return `ERROR: adw plans ${command} failed (${reason}).\n${diagnostic}${suffix}`;
}

export function stripDefaultArgs(raw: Record<string, any>): Record<string, any> {
  const cleaned: Record<string, any> = { command: raw.command };
  for (const [key, value] of Object.entries(raw)) {
    if (key === "command") continue;
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.trim() === "") continue;
    if (value === false) continue;
    cleaned[key] = value;
  }
  return cleaned;
}

export function validateRequiredArgs(
  raw: Record<string, any>,
  requirements: RequiredArgMap,
  buildError: BuildError,
): string | undefined {
  const command = String(raw.command ?? "");
  const specs = requirements[command] ?? [];
  for (const spec of specs) {
    const value = raw[spec.field];
    if (typeof value !== "string" || value.trim() === "") {
      return buildError(spec.message);
    }
  }
  return undefined;
}

export function validateUpdatePhaseIssueLinkArgs(
  issueNumber: unknown,
  clearIssueNumber: unknown,
  buildError: BuildError,
): string | undefined {
  const hasIssueNumber = issueNumber !== undefined && issueNumber !== null;
  const hasClearIssueNumber = clearIssueNumber === true;
  if (hasIssueNumber && hasClearIssueNumber) {
    return buildError(
      "'issue_number' and 'clear_issue_number' are mutually exclusive for update-phase.",
    );
  }
  return undefined;
}
