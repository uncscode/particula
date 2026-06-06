import { describe, expect, it } from "bun:test";

import {
  buildCommandFailureError,
  deriveCommandFailureHint,
  sanitizeCommandFailureOutput,
  sanitizeSuccessOutput,
  selectCommandFailureDiagnostic,
  stripDefaultArgs,
  validateRequiredArgs,
} from "../adw_plans_contract_shared";

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
        { command: "list", title: "ignored" },
        requirements,
        buildError,
      ),
    ).toBeUndefined();
  });
});
