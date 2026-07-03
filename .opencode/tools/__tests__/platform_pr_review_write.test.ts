import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure, buildSuccessOutput } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_pr_review_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles plain pr-review command", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    await execute({ command: "pr-review", issue_number: "42", body: "review" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform pr-review 42 --body review",
    );
  });

  it("assembles inline review with path+line", async () => {
    setDollarText("ok");
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "x.ts",
      line: 9,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--path x.ts --line 9");
  });

  it("assembles inline review with path+position", async () => {
    setDollarText("ok");
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "x.ts",
      position: 3,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--path x.ts --position 3");
  });

  it("enforces relation checks", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const noPathWithLine = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      line: 1,
    });
    assertContains(String(noPathWithLine), "'--line' requires '--path'");

    const noPathWithPosition = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      position: 1,
    });
    assertContains(String(noPathWithPosition), "'--position' requires '--path'");

    const missingLineOrPosition = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "x.ts",
    });
    assertContains(String(missingLineOrPosition), "'--path' requires '--line' or '--position'");
  });

  it("help mode bypasses required args", async () => {
    setDollarText("usage");
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const result = await execute({ command: "pr-review", help: true });
    expect(String(result)).toContain("usage");
  });

  it("help-path failure prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    setDollarError(buildDollarFailure({ stderr: "help stderr", stdout: "help stdout" }));
    const result = await execute({ command: "pr-review", help: true });
    const text = String(result);
    assertContains(text, "help stderr");
    expect(text.indexOf("help stderr")).toBeLessThan(text.indexOf("help stdout"));
  });

  it("redacts and truncates sensitive diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const longToken = "ghp_" + "a".repeat(10000);
    setDollarError(buildDollarFailure({ stderr: `token=${longToken}` }));
    const result = await execute({ command: "pr-review", issue_number: "42", body: "review" });
    const text = String(result);
    assertContains(text, "[REDACTED]");
    expect(text.length).toBeLessThan(longToken.length);
    expect(text).not.toContain(longToken);
  });

  it("prefers stderr then stdout for diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));
    const result = await execute({ command: "pr-review", issue_number: "42", body: "review" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text.indexOf("fatal stderr")).toBeLessThan(text.indexOf("shadow stdout"));
  });

  it("falls back to stdout when stderr is empty", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    setDollarError(buildDollarFailure({ stderr: "", stdout: "stdout diagnostic" }));
    const result = await execute({ command: "pr-review", issue_number: "42", body: "review" });
    assertContains(String(result), "stdout diagnostic");
  });

  it("validates required fields", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const before = getInvocations().length;
    const missingIssue = await execute({ command: "pr-review", body: "review" });
    assertErrorPrefix(String(missingIssue), "ERROR:");
    assertContains(String(missingIssue), "'issue_number' is required");
    const whitespaceIssue = await execute({ command: "pr-review", issue_number: "   ", body: "review" });
    assertErrorPrefix(String(whitespaceIssue), "ERROR:");
    assertContains(String(whitespaceIssue), "'issue_number' is required");
    const missingBody = await execute({ command: "pr-review", issue_number: "42", body: "   " });
    assertErrorPrefix(String(missingBody), "ERROR:");
    expect(getInvocations().length).toBe(before);
  });

  it("rejects all-zero issue token and invalid prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");

    const zeroIssue = await execute({ command: "pr-review", issue_number: "000", body: "review" });
    assertErrorPrefix(String(zeroIssue), "ERROR:");
    assertContains(String(zeroIssue), "positive integer token");

    const badScope = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      prefer_scope: " ",
    });
    assertErrorPrefix(String(badScope), "ERROR:");
    assertContains(String(badScope), "'prefer_scope' must be either 'fork' or 'upstream'");
  });

  it("rejects invalid inline coordinates before spawn", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const before = getInvocations().length;

    const zeroLine = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "x.ts",
      line: 0,
    });
    assertErrorPrefix(String(zeroLine), "ERROR:");
    assertContains(String(zeroLine), "'line' must be a positive integer");

    const fractionalPosition = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "x.ts",
      position: 1.5,
    });
    assertErrorPrefix(String(fractionalPosition), "ERROR:");
    assertContains(String(fractionalPosition), "'position' must be a positive integer");

    expect(getInvocations().length).toBe(before);
  });

  it("rejects unsafe inline path values before spawn", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const before = getInvocations().length;

    const absolutePath = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "/etc/passwd",
      line: 4,
    });
    assertErrorPrefix(String(absolutePath), "ERROR:");
    assertContains(String(absolutePath), "safe repository-relative path");

    const traversalPath = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      path: "../x.ts",
      line: 4,
    });
    assertErrorPrefix(String(traversalPath), "ERROR:");
    assertContains(String(traversalPath), "safe repository-relative path");

    expect(getInvocations().length).toBe(before);
  });

  it("validates commit_sha format before spawn", async () => {
    const execute = await loadToolExecute("../../platform_pr_review_write.ts");
    const before = getInvocations().length;

    const badSha = await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      commit_sha: "abc123g",
    });
    assertErrorPrefix(String(badSha), "ERROR:");
    assertContains(String(badSha), "SHA-like hex token");

    setDollarText("ok");
    await execute({
      command: "pr-review",
      issue_number: "42",
      body: "review",
      commit_sha: "abc1234",
    });
    expect(getInvocations().length).toBe(before + 1);
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--commit-sha abc1234");
  });
});
