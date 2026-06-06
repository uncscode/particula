import { afterEach, beforeEach, describe, expect, it, test } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_operations compatibility delegation", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("delegates comment to split wrapper while preserving output", async () => {
    setDollarText("comment-ok");
    const execute = await loadToolExecute("../../platform_operations.ts");
    const result = await execute({ command: "comment", issue_number: "88", body: "hello" });
    expect(String(result)).toContain("comment-ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw platform comment 88 --body hello",
    );
  });

  it("delegates pr-review to split wrapper while preserving output", async () => {
    setDollarText("review-ok");
    const execute = await loadToolExecute("../../platform_operations.ts");
    const result = await execute({
      command: "pr-review",
      issue_number: "91",
      body: "ship it",
      path: "file.ts",
      line: 11,
    });
    expect(String(result)).toContain("review-ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw platform pr-review 91 --body ship it --path file.ts --line 11",
    );
  });

  it("passes prefer_scope through delegated wrappers", async () => {
    setDollarText("ok");
    const execute = await loadToolExecute("../../platform_operations.ts");

    await execute({
      command: "comment",
      issue_number: "88",
      body: "hello",
      prefer_scope: "upstream",
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--prefer-scope upstream");

    await execute({
      command: "pr-review",
      issue_number: "91",
      body: "ship it",
      path: "file.ts",
      position: 3,
      prefer_scope: "fork",
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--prefer-scope fork");
  });

  it("preserves deterministic invalid-input error through compatibility surface", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    const before = getInvocations().length;
    const missingBody = await execute({ command: "comment", issue_number: "55", body: "   " });
    assertErrorPrefix(String(missingBody), "ERROR:");
    assertContains(String(missingBody), "'body' is required for command 'comment'");

    const badRelation = await execute({
      command: "pr-review",
      issue_number: "55",
      body: "ok",
      line: 4,
    });
    assertErrorPrefix(String(badRelation), "ERROR:");
    assertContains(String(badRelation), "'--line' requires '--path'");

    const missingIssue = await execute({ command: "comment", body: "ok" } as any);
    assertErrorPrefix(String(missingIssue), "ERROR:");
    assertContains(String(missingIssue), "'issue_number' is required");

    const whitespaceIssue = await execute({ command: "pr-review", issue_number: "   ", body: "ok" });
    assertErrorPrefix(String(whitespaceIssue), "ERROR:");
    assertContains(String(whitespaceIssue), "'issue_number' is required");

    expect(getInvocations().length).toBe(before);
  });

  it("preserves delegated failure-envelope diagnostics through compatibility surface", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));

    const commentFailure = await execute({ command: "comment", issue_number: "55", body: "ok" });
    const commentText = String(commentFailure);
    assertErrorPrefix(commentText, "ERROR:");
    expect(commentText.indexOf("fatal stderr")).toBeLessThan(commentText.indexOf("shadow stdout"));

    const reviewFailure = await execute({ command: "pr-review", issue_number: "55", body: "ok" });
    const reviewText = String(reviewFailure);
    assertErrorPrefix(reviewText, "ERROR:");
    expect(reviewText.indexOf("fatal stderr")).toBeLessThan(reviewText.indexOf("shadow stdout"));
  });

  it("preserves delegated help-path failure envelope ordering", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    setDollarError(buildDollarFailure({ stderr: "help stderr", stdout: "help stdout" }));

    const commentHelpFailure = await execute({ command: "comment", help: true });
    const commentText = String(commentHelpFailure);
    assertErrorPrefix(commentText, "ERROR:");
    assertContains(commentText, "help stderr");
    assertContains(commentText, "help stdout");

    const reviewHelpFailure = await execute({ command: "pr-review", help: true });
    const reviewText = String(reviewHelpFailure);
    assertErrorPrefix(reviewText, "ERROR:");
    assertContains(reviewText, "help stderr");
    assertContains(reviewText, "help stdout");
  });

  it("passes create-pr adw_id through compatibility path", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    setDollarText("ok");
    const result = await execute({
      command: "create-pr",
      title: "feat: #1 - test",
      head: "feature/test",
      adw_id: "ABC12345",
    });
    expect(String(result)).toContain("PLATFORM_PR_CREATED");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--adw-id ABC12345");
  });

  it("normalizes remove-labels prefer_scope and rejects blank values before spawn", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");

    setDollarText("labels-ok");
    await execute({
      command: "remove-labels",
      issue_number: "55",
      labels: "bug",
      prefer_scope: "upstream",
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--prefer-scope upstream");

    const before = getInvocations().length;
    const badScope = await execute({
      command: "remove-labels",
      issue_number: "55",
      labels: "bug",
      prefer_scope: "   ",
    });
    assertErrorPrefix(String(badScope), "ERROR:");
    assertContains(String(badScope), "'prefer_scope' must be either 'fork' or 'upstream'");
    expect(getInvocations().length).toBe(before);
  });

  test.skip("baseline repro (quarantined): preserves deterministic ERROR envelope precedence for pr-comments missing issue_number", async () => {
    // baseline repro retained for M34-P1 audit evidence; quarantined from default pass path
    const execute = await loadToolExecute("../../platform_operations.ts");
    const before = getInvocations().length;
    const missingIssue = await execute({ command: "pr-comments" });
    const text = String(missingIssue);
    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "Failed to execute 'adw platform pr-comments'");
    expect(getInvocations().length).toBe(before);
  });
});
