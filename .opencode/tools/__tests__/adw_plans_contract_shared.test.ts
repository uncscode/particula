import { describe, expect, it } from "bun:test";
import { existsSync, mkdirSync, realpathSync, rmSync, symlinkSync } from "node:fs";
import path from "node:path";

import {
  ADW_PLANS_OPTION_STRING_AUDIT,
  ADW_PLANS_OPTION_STRING_RULES,
  buildCommandFailureError,
  deriveCommandFailureHint,
  hasMeaningfulSplitWrapperAliasValue,
  mergeParsedOptionField,
  parseCommandOptionsString,
  redactPathLikeText,
  sanitizeCommandFailureOutput,
  sanitizeSuccessOutput,
  selectCommandFailureDiagnostic,
  stripDefaultArgs,
  validateAndNormalizePlansCwdPath,
  validateUpdatePhaseIssueLinkArgs,
  validateRequiredArgs,
} from "../adw_plans_contract_shared";

const findRepoRoot = (): string => {
  let current = path.resolve(process.cwd());
  while (true) {
    if (existsSync(path.join(current, "AGENTS.md")) && existsSync(path.join(current, ".opencode"))) {
      return current;
    }
    const parent = path.resolve(current, "..");
    if (parent === current) {
      return path.resolve(process.cwd());
    }
    current = parent;
  }
};

describe("adw_plans_contract_shared", () => {
  it("sanitizes diagnostics by removing control characters and redacting absolute paths", () => {
    const result = sanitizeCommandFailureOutput("boom\u0000 /tmp/example/file.py\nnext");

    expect(result).toEqual({
      text: "boom <path>\nnext",
      hasVisibleContent: true,
    });
  });

  it("redacts Windows and spaced absolute paths", () => {
    const result = sanitizeCommandFailureOutput(
      'open "C:\\Users\\me\\Project Files\\backend.py" and /tmp/path with spaces/backend.ts: failed',
    );

    expect(result.text).toContain("<path>");
    expect(result.text).toContain("<path>: failed");
    expect(result.text).not.toContain("C:\\Users\\me");
    expect(result.text).not.toContain("/tmp/path with spaces/backend.ts");
  });

  it("redacts token-like secrets in failure diagnostics", () => {
    const result = sanitizeCommandFailureOutput(
      'Authorization: Bearer github_pat_secretToken123 token=abc123 password="letmein"',
    );

    expect(result.text).toContain("Authorization: Bearer <redacted-secret>");
    expect(result.text).toContain("token=<redacted-secret>");
    expect(result.text).toContain('password="<redacted-secret>"');
    expect(result.text).not.toContain("github_pat_secretToken123");
    expect(result.text).not.toContain("abc123");
    expect(result.text).not.toContain("letmein");
  });

  it("returns no visible content for empty diagnostics", () => {
    expect(sanitizeCommandFailureOutput("")).toEqual({
      text: "",
      hasVisibleContent: false,
    });
  });

  it("sanitizes success output without redacting path-like substrings", () => {
    expect(sanitizeSuccessOutput("sections/root/path\u0000")).toEqual({
      text: "sections/root/path",
      hasVisibleContent: true,
    });
  });

  it("returns no visible success content when control characters sanitize to empty", () => {
    expect(sanitizeSuccessOutput("\u0000\u0001\u0002")).toEqual({
      text: "",
      hasVisibleContent: false,
    });
  });

  it("preserves bounded truncation marker for long success output", () => {
    const result = sanitizeSuccessOutput(`success:${"x".repeat(5000)}`);

    expect(result.hasVisibleContent).toBe(true);
    expect(result.text).toContain("...[output truncated to 4000 characters; original length");
  });

  it("preserves bounded truncation marker for long diagnostics after sanitization", () => {
    const result = sanitizeCommandFailureOutput(`failure:${"x".repeat(5000)}`);

    expect(result.hasVisibleContent).toBe(true);
    expect(result.text).toContain("...[output truncated to 4000 characters; original length");
  });

  it("selects diagnostics with stderr to stdout to message precedence", () => {
    expect(
      selectCommandFailureDiagnostic(
        { stderr: "stderr first", stdout: "stdout second", message: "message third" },
        "fallback",
      ),
    ).toBe("stderr first");

    expect(
      selectCommandFailureDiagnostic(
        { stderr: "", stdout: "stdout second", message: "message third" },
        "fallback",
      ),
    ).toBe("stdout second");

    expect(
      selectCommandFailureDiagnostic(
        { stderr: "", stdout: "", message: "message third" },
        "fallback",
      ),
    ).toBe("message third");

    expect(
      selectCommandFailureDiagnostic({ stderr: "", stdout: "", message: "" }, "fallback"),
    ).toBe("fallback");
  });

  it("ignores whitespace-only higher-priority diagnostics before falling back", () => {
    expect(
      selectCommandFailureDiagnostic(
        { stderr: "   \n\t", stdout: "stdout second", message: "message third" },
        "fallback",
      ),
    ).toBe("stdout second");
    expect(
      selectCommandFailureDiagnostic(
        { stderr: "   ", stdout: "   ", message: "message third" },
        "fallback",
      ),
    ).toBe("message third");
  });

  it("derives runtime and cwd hints for recognized diagnostics", () => {
    expect(deriveCommandFailureHint("ENOENT: python3 not found")).toContain(
      "required runtime/tooling",
    );
    expect(
      deriveCommandFailureHint("cwd path resolves outside repository root"),
    ).toContain("--cwd");
    expect(deriveCommandFailureHint("plan worktree metadata is stale")).toBeUndefined();
    expect(deriveCommandFailureHint("target file does not exist")).toBeUndefined();
    expect(deriveCommandFailureHint("plain failure")).toBeUndefined();
  });

  it("returns undefined hints for empty diagnostics", () => {
    expect(deriveCommandFailureHint("")).toBeUndefined();
  });

  it("builds deterministic failure envelopes with hints", () => {
    const result = buildCommandFailureError(
      "list",
      "execution error",
      { stderr: "python3: can't open file /tmp/backend.py: No such file or directory" },
      "fallback",
    );

    expect(result).toContain("ERROR: adw plans list failed (execution error).");
    expect(result).toContain("<path>");
    expect(result).toContain(
      "hint: verify the required runtime/tooling is installed",
    );
    expect(result).not.toContain("/tmp/backend.py");
  });

  it("builds deterministic failure envelopes without hints for unrecognized failures", () => {
    const result = buildCommandFailureError(
      "list",
      "exit 2",
      { stderr: "plain failure" },
      "fallback",
    );

    expect(result).toBe("ERROR: adw plans list failed (exit 2).\nplain failure");
  });

  it("redacts path-like text and falls back to a canonical token when nothing remains visible", () => {
    expect(redactPathLikeText("/tmp/example/file.py")).toBe("<path>");
    expect(redactPathLikeText("\u0000\u0001")).toBe("<path>");
  });

  it("strips inert optional args while preserving command and meaningful values", () => {
    expect(
      stripDefaultArgs({
        command: "list",
        empty: "   ",
        disabled: false,
        missing: null,
        ignored: undefined,
        zero: 0,
        keep: "value",
      }),
    ).toEqual({ command: "list", zero: 0, keep: "value" });
  });

  it("validates required args from raw inputs before sparse stripping", () => {
    const requirements = {
      create: [
        { field: "plan_type", message: "create command requires 'plan_type'." },
        { field: "title", message: "create command requires 'title'." },
      ],
    };
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(
      validateRequiredArgs(
        { command: "create", plan_type: "feature", title: "   " },
        requirements,
        buildError,
      ),
    ).toBe("ERROR: create command requires 'title'.");
    expect(
      validateRequiredArgs(
        { command: "create", plan_type: "feature", title: "ok" },
        requirements,
        buildError,
      ),
    ).toBeUndefined();
    expect(
      validateRequiredArgs(
        { command: "create", plan_type: 123, title: "ok" },
        requirements,
        buildError,
      ),
    ).toBe("ERROR: create command requires 'plan_type'.");
    expect(
      validateRequiredArgs(
        { command: "list", title: "ignored" },
        requirements,
        buildError,
      ),
    ).toBeUndefined();
  });

  it("classifies split-wrapper direct alias values as meaningful or inert", () => {
    expect(hasMeaningfulSplitWrapperAliasValue(undefined)).toBe(false);
    expect(hasMeaningfulSplitWrapperAliasValue(null)).toBe(false);
    expect(hasMeaningfulSplitWrapperAliasValue(false)).toBe(false);
    expect(hasMeaningfulSplitWrapperAliasValue(0)).toBe(false);
    expect(hasMeaningfulSplitWrapperAliasValue("   ")).toBe(false);

    expect(hasMeaningfulSplitWrapperAliasValue("Ready")).toBe(true);
    expect(hasMeaningfulSplitWrapperAliasValue(true)).toBe(true);
    expect(hasMeaningfulSplitWrapperAliasValue(1)).toBe(true);
    expect(hasMeaningfulSplitWrapperAliasValue({ alias: true })).toBe(true);
  });

  it("canonicalizes validated cwd paths", () => {
    const tempRoot = path.resolve(process.cwd(), "adforge_local/opencode/tmp");
    mkdirSync(tempRoot, { recursive: true });
    const aliasPath = path.join(tempRoot, "adw-plans-shared-alias");
    rmSync(aliasPath, { recursive: true, force: true });
    const repoRoot = findRepoRoot();
    symlinkSync(repoRoot, aliasPath, "dir");

    try {
      expect(validateAndNormalizePlansCwdPath(aliasPath)).toEqual({
        value: realpathSync(repoRoot),
      });
    } finally {
      rmSync(aliasPath, { recursive: true, force: true });
    }
  });

  it("captures the wrapper-family optional field inventory with explicit dispositions", () => {
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans",
      command: "list",
      field: "status",
      cliFlag: "--status",
      disposition: "token_candidate",
      reason:
        "Bounded plan status values support options-string status=<value> tokens while remaining accepted as direct wrapper fields.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans_mutate",
      command: "update-phase",
      field: "phase_status",
      cliFlag: "--status",
      disposition: "token_candidate",
      reason:
        "Bounded phase status values support options-string phase-status=<value> tokens while split wrappers route them through options only.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans",
      command: "update",
      field: "patch",
      cliFlag: "--patch",
      disposition: "direct_exception",
      reason: "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans_read",
      command: "schema",
      field: "check",
      cliFlag: "--check",
      disposition: "token_candidate",
      reason: "Simple boolean flag forwarding maps to a bare token.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans_mutate",
      command: "create",
      field: "plan_id",
      cliFlag: "--id",
      disposition: "retained_direct",
      reason: "Optional explicit IDs stay retained direct fields because they shape persisted plan identity.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans",
      command: "add-phase",
      field: "title",
      cliFlag: "--title",
      disposition: "retained_direct",
      reason: "Free-form titles remain retained direct fields rather than entering token parsing.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans",
      command: "scaffold-sections",
      field: "cwd",
      cliFlag: "--cwd",
      disposition: "retained_direct",
      reason:
        "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans_mutate",
      command: "update-phase",
      field: "cwd",
      cliFlag: "--cwd",
      disposition: "retained_direct",
      reason:
        "Mutating worktree/repository scoping stays a retained direct field and command-required outside token parsing.",
    });
    expect(ADW_PLANS_OPTION_STRING_AUDIT.filter((entry) => entry.field === "cwd")).toHaveLength(20);
    expect(ADW_PLANS_OPTION_STRING_AUDIT).toContainEqual({
      wrapper: "adw_plans_read",
      command: "list",
      field: "status",
      cliFlag: "--status",
      disposition: "token_candidate",
      reason:
        "Bounded plan status values support options-string status=<value> tokens while split wrappers route them through options only.",
    });
    expect(
      ADW_PLANS_OPTION_STRING_AUDIT.filter((entry) => entry.disposition === "direct_exception"),
    ).toEqual([
      {
        wrapper: "adw_plans",
        command: "update",
        field: "patch",
        cliFlag: "--patch",
        disposition: "direct_exception",
        reason:
          "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules.",
      },
      {
        wrapper: "adw_plans",
        command: "update-phase",
        field: "patch",
        cliFlag: "--patch",
        disposition: "direct_exception",
        reason:
          "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules.",
      },
      {
        wrapper: "adw_plans_mutate",
        command: "update",
        field: "patch",
        cliFlag: "--patch",
        disposition: "direct_exception",
        reason:
          "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules.",
      },
      {
        wrapper: "adw_plans_mutate",
        command: "update-phase",
        field: "patch",
        cliFlag: "--patch",
        disposition: "direct_exception",
        reason:
          "Raw JSON patch payloads remain direct because quoting and whitespace exceed bounded token rules.",
      },
    ]);
  });

  it("defines the bounded command allowlists and malformed token rules for the later parser slice", () => {
    expect(ADW_PLANS_OPTION_STRING_RULES.commandAllowlist).toEqual({
      list: ["json", "status"],
      show: ["json"],
      create: ["status", "priority", "size"],
      update: ["status", "priority", "size"],
      "add-phase": ["phase-status", "size", "after"],
      "update-phase": ["phase-status", "size", "issue", "clear-issue-number"],
      schema: ["check"],
      "list-sections": ["json", "populate"],
      validate: [],
      "scaffold-sections": [],
    });
    expect(ADW_PLANS_OPTION_STRING_RULES.tokenBooleanFlags).toEqual(["json", "populate", "check", "clear-issue-number"]);
    expect(ADW_PLANS_OPTION_STRING_RULES.tokenKeyValueFields).toEqual([
      "status",
      "phase-status",
      "priority",
      "size",
      "after",
      "issue",
    ]);
    expect(ADW_PLANS_OPTION_STRING_RULES.retainedDirectFields).not.toContain("status");
    expect(ADW_PLANS_OPTION_STRING_RULES.retainedDirectFields).not.toContain("phase_status");
    expect(ADW_PLANS_OPTION_STRING_RULES.compatibilityRetainedDirectFields).toEqual([
      "status",
      "phase_status",
    ]);
    expect(ADW_PLANS_OPTION_STRING_RULES.splitWrapperOptionOnlyFields).toEqual([
      "status",
      "phase_status",
    ]);
    expect(ADW_PLANS_OPTION_STRING_RULES.malformedTokenRules).toContain(
      "issue values must parse as positive safe integers.",
    );
    expect(ADW_PLANS_OPTION_STRING_RULES.malformedTokenRules).toContain(
      "update-phase must not combine 'issue=<n>' with 'clear-issue-number'.",
    );
    expect(ADW_PLANS_OPTION_STRING_RULES.mutualExclusionRules).toEqual([
      "update-phase tokens 'issue=<n>' and 'clear-issue-number' are mutually exclusive and must fail closed when combined.",
    ]);
    expect(ADW_PLANS_OPTION_STRING_RULES.patchExceptionReason).toContain("whitespace, braces, and quotes");
    expect(ADW_PLANS_OPTION_STRING_RULES.behaviorNeutralScope).toContain(
      "compatibility and split wrappers",
    );
  });

  it("parses allowlisted boolean and key-value options", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(parseCommandOptionsString("list", "json", buildError)).toEqual({
      values: { json: true },
    });
    expect(parseCommandOptionsString("list", "status=Ready json", buildError)).toEqual({
      values: { status: "Ready", json: true },
    });
    expect(parseCommandOptionsString("list-sections", "json populate", buildError)).toEqual({
      values: { json: true, populate: true },
    });
    expect(parseCommandOptionsString("add-phase", "after=M37-P1 size=M", buildError)).toEqual({
      values: { after: "M37-P1", size: "M" },
    });
    expect(parseCommandOptionsString("update", "status=Ready priority=P1", buildError)).toEqual({
      values: { status: "Ready", priority: "P1" },
    });
    expect(
      parseCommandOptionsString("update-phase", "phase-status=In Progress size=M issue=42", buildError),
    ).toEqual({
      values: { phase_status: "In Progress", size: "M", issue_number: 42 },
    });
    expect(
      parseCommandOptionsString(
        "update-phase",
        "phase-status=Blocked clear-issue-number",
        buildError,
      ),
    ).toEqual({
      values: { phase_status: "Blocked", clear_issue_number: true },
    });
    expect(parseCommandOptionsString("update", "status=In Progress json", buildError)).toEqual({
      error:
        "ERROR: Invalid options token 'status=In Progress json' for 'update': status values must be one of: Draft, Proposed, Ready, In Progress, Blocked, Monitoring, Shipped, Cancelled, Superseded",
    });
    expect(
      parseCommandOptionsString("update-phase", "issue=42 clear-issue-number", buildError),
    ).toEqual({
      error:
        "ERROR: 'issue_number' and 'clear_issue_number' are mutually exclusive for update-phase.",
    });
  });

  it("treats whitespace-only options as omitted", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(parseCommandOptionsString("list", "   \n\t  ", buildError)).toEqual({});
  });

  it("collapses repeated identical options tokens to one effective value", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(parseCommandOptionsString("list", "json json", buildError)).toEqual({
      values: { json: true },
    });
    expect(parseCommandOptionsString("update", "size=M size=M", buildError)).toEqual({
      values: { size: "M" },
    });
  });

  it("rejects non-string options payloads and unsupported commands", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(parseCommandOptionsString("list", { bad: true }, buildError)).toEqual({
      error: "ERROR: 'options' must be a string when provided.",
    });
    expect(parseCommandOptionsString("nope", "json", buildError)).toEqual({
      error: "ERROR: Unsupported command 'nope' for options parsing.",
    });
  });

  it("fails closed for malformed and unsupported tokens", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(parseCommandOptionsString("list", "json=true", buildError).error).toContain(
      "token must be provided without '=value'",
    );
    expect(parseCommandOptionsString("list", "priority=P1", buildError).error).toContain(
      "token is not allowed for this command",
    );
    expect(parseCommandOptionsString("list", "status=Unknown", buildError).error).toContain(
      "status values must be one of",
    );
    expect(parseCommandOptionsString("update-phase", "issue=0", buildError).error).toContain(
      "issue values must be positive safe integers",
    );
    expect(parseCommandOptionsString("update", "size=M size=L", buildError).error).toContain(
      "conflicting duplicate 'size=L'",
    );
    expect(parseCommandOptionsString("schema", "check=yes=no", buildError).error).toContain(
      "at most one '=' separator",
    );
    expect(parseCommandOptionsString("create", "priority=P9", buildError).error).toContain(
      "priority values must be one of: P0, P1, P2, P3, Backlog",
    );
    expect(parseCommandOptionsString("update", "size=HUGE", buildError).error).toContain(
      "size values must be one of: XS, S, M, L, XL, XXL",
    );
    expect(parseCommandOptionsString("create", "priority=-P1", buildError).error).toContain(
      "priority values must not start with '-'",
    );
    expect(parseCommandOptionsString("update", "status=Unknown", buildError).error).toContain(
      "status values must be one of",
    );
    expect(
      parseCommandOptionsString("update-phase", "phase-status=Blocked clear-issue-number=no", buildError)
        .error,
    ).toContain("token must be provided without '=value'");
    expect(
      parseCommandOptionsString("update-phase", "phase-status=Unknown Value", buildError).error,
    ).toContain("phase-status values must be one of");
    expect(parseCommandOptionsString("add-phase", "after=--later", buildError).error).toContain(
      "after values must not start with '-'",
    );
    expect(parseCommandOptionsString("update-phase", "issue=01", buildError).error).toContain(
      "issue values must be positive safe integers",
    );
  });

  it("validates update-phase issue linking mutual exclusion", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(validateUpdatePhaseIssueLinkArgs(123, false, buildError)).toBeUndefined();
    expect(validateUpdatePhaseIssueLinkArgs(undefined, true, buildError)).toBeUndefined();
    expect(validateUpdatePhaseIssueLinkArgs(123, true, buildError)).toBe(
      "ERROR: 'issue_number' and 'clear_issue_number' are mutually exclusive for update-phase.",
    );
  });

  it("merges direct and parsed option values with deterministic conflict handling", () => {
    const buildError = (message: string): string => `ERROR: ${message}`;

    expect(mergeParsedOptionField("Ready", undefined, "status", buildError)).toEqual({
      value: "Ready",
    });
    expect(mergeParsedOptionField(undefined, "Ready", "status", buildError)).toEqual({
      value: "Ready",
    });
    expect(mergeParsedOptionField("Ready", "Ready", "status", buildError)).toEqual({
      value: "Ready",
    });
    expect(mergeParsedOptionField("Ready", "Blocked", "status", buildError)).toEqual({
      error: "ERROR: 'status' cannot conflict between direct input and options string.",
    });
  });
});
