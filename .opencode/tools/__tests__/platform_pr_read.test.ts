import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_pr_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires issue_number for pr-comments", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    const result = await execute({ command: "pr-comments" });
    assertContains(String(result), "'issue_number' is required");
  });

  it("requires issue_number for pr-diff", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    const result = await execute({ command: "pr-diff" });
    assertContains(String(result), "'issue_number' is required");
  });

  it("assembles pr-comments", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    await execute({ command: "pr-comments", issue_number: "12" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw platform pr-comments 12");
  });

  it("accepts leading-zero issue_number and assembles prefer_scope/actionable_only", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    await execute({
      command: "pr-comments",
      issue_number: "00012",
      prefer_scope: "fork",
      actionable_only: true,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform pr-comments 00012 --actionable-only --prefer-scope fork",
    );
  });

  it("assembles pr-diff with format and prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    await execute({
      command: "pr-diff",
      issue_number: "12",
      output_format: "json",
      prefer_scope: "upstream",
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform pr-diff 12 --format json --prefer-scope upstream",
    );
  });

  it("rejects actionable_only for pr-diff", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    const result = await execute({
      command: "pr-diff",
      issue_number: "12",
      actionable_only: true,
    });
    assertContains(String(result), "'actionable_only' is only supported for command 'pr-comments'");
  });

  it("rejects blank prefer_scope when explicitly provided", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    const result = await execute({
      command: "pr-diff",
      issue_number: "12",
      prefer_scope: "   ",
    });
    assertContains(String(result), "'prefer_scope' must be either 'fork' or 'upstream'");
  });

  it("rejects invalid output_format and actionable_only types before spawn", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    const before = getInvocations().length;

    const badFormat = await execute({
      command: "pr-diff",
      issue_number: "12",
      output_format: "yaml",
    });
    assertContains(String(badFormat), "'output_format' must be either 'text' or 'json'");

    const badActionable = await execute({
      command: "pr-comments",
      issue_number: "12",
      actionable_only: "yes",
    } as any);
    assertContains(String(badActionable), "'actionable_only' must be a boolean when provided");

    expect(getInvocations().length).toBe(before);
  });

  it("rejects zero-only issue_number token", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    const result = await execute({ command: "pr-comments", issue_number: "000" });
    assertContains(String(result), "must be a positive integer token");
  });

  it("help-path failure prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(buildDollarFailure({ stderr: "help stderr", stdout: "help stdout" }));
    const result = await execute({ command: "pr-comments", help: true });
    const text = String(result);
    assertContains(text, "help stderr");
    expect(text.indexOf("help stderr")).toBeLessThan(text.indexOf("help stdout"));
  });

  it("pr-diff failure envelope prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(buildDollarFailure({ stderr: "diff stderr", stdout: "diff stdout" }));
    const result = await execute({ command: "pr-diff", issue_number: "12" });
    const text = String(result);
    assertContains(text, "Failed to execute 'adw platform pr-diff'");
    assertContains(text, "diff stderr");
    expect(text.indexOf("diff stderr")).toBeLessThan(text.indexOf("diff stdout"));
  });

  it("preserves sanitized structured json failure payloads for pr-diff json mode", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(
      buildDollarFailure({
        stdout: JSON.stringify({
          success: false,
          error: "diff fetch failed",
          scope: "upstream",
          token: "ghp_secretsecretsecret",
        }),
        stderr: "ignored stderr",
      }),
    );

    const result = await execute({ command: "pr-diff", issue_number: "12", output_format: "json" });

    expect(JSON.parse(String(result))).toEqual({
      success: false,
      error: "diff fetch failed",
      scope: "upstream",
      token: "[REDACTED]",
    });
  });

  it("preserves structured json failures without truncating machine-readable payloads", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(
      buildDollarFailure({
        stdout: JSON.stringify({ ok: false, detail: "x".repeat(2500) }),
      }),
    );

    const result = String(
      await execute({ command: "pr-diff", issue_number: "12", output_format: "json" }),
    );

    expect(JSON.parse(result)).toEqual({ ok: false, detail: "x".repeat(2500) });
  });

  it("falls back to text envelope when json mode failure lacks structured stdout", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(buildDollarFailure({ stdout: "not json", stderr: "diff stderr" }));

    const result = await execute({ command: "pr-diff", issue_number: "12", output_format: "json" });
    const text = String(result);

    assertContains(text, "Failed to execute 'adw platform pr-diff'");
    assertContains(text, "diff stderr");
  });

  it("sanitizes and redacts failure diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(
      buildDollarFailure({
        stderr: "Authorization: Bearer secret-token\napi_key=abc1234567890\u0007",
      }),
    );
    const result = await execute({ command: "pr-diff", issue_number: "12" });
    const text = String(result);
    assertContains(text, "Authorization: Bearer [REDACTED]");
    assertContains(text, "api_key= [REDACTED]");
    expect(text).not.toContain("secret-token");
    expect(text).not.toContain("abc1234567890");
  });

  it("redacts quoted json-style secret fields inside failure diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_pr_read.ts");
    setDollarError(
      buildDollarFailure({
        stderr: 'upstream payload: {"token":"abc123","client_secret":"def456"}',
      }),
    );
    const result = await execute({ command: "pr-comments", issue_number: "12" });
    const text = String(result);

    assertContains(text, '"token":"[REDACTED]"');
    assertContains(text, '"client_secret":"[REDACTED]"');
    expect(text).not.toContain("abc123");
    expect(text).not.toContain("def456");
  });
});
