import { afterEach, beforeEach, describe, expect, it } from "bun:test";

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
      "uv run --active adw platform comment 88 --body hello",
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
      "uv run --active adw platform pr-review 91 --body ship it --path file.ts --line 11",
    );
  });

  it("retains pr-comments only as a compatibility delegator", async () => {
    setDollarText("comments-ok");
    const execute = await loadToolExecute("../../platform_operations.ts");
    const result = await execute({ command: "pr-comments", issue_number: "91", output_format: "json" });
    expect(String(result)).toContain("comments-ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform pr-comments 91 --format json",
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

    const unsafePath = await execute({
      command: "pr-review",
      issue_number: "55",
      body: "ok",
      path: "../escape.ts",
      line: 4,
    });
    assertErrorPrefix(String(unsafePath), "ERROR:");
    assertContains(
      String(unsafePath),
      "'path' must be a safe repository-relative path without traversal for command 'pr-review'",
    );

    const badCommitSha = await execute({
      command: "pr-review",
      issue_number: "55",
      body: "ok",
      commit_sha: "not-a-sha",
    });
    assertErrorPrefix(String(badCommitSha), "ERROR:");
    assertContains(String(badCommitSha), "'commit_sha' must be a SHA-like hex token (7-64 chars)");

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
    expect(commentText.indexOf("help stderr")).toBeLessThan(commentText.indexOf("help stdout"));

    const reviewHelpFailure = await execute({ command: "pr-review", help: true });
    const reviewText = String(reviewHelpFailure);
    assertErrorPrefix(reviewText, "ERROR:");
    assertContains(reviewText, "help stderr");
    assertContains(reviewText, "help stdout");
    expect(reviewText.indexOf("help stderr")).toBeLessThan(reviewText.indexOf("help stdout"));
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

  it("preserves create-pr failure marker contract through compatibility path", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    setDollarError(buildDollarFailure({ stderr: "cannot create", stdout: "details" }));

    const result = await execute({
      command: "create-pr",
      title: "feat: #1 - test",
      head: "feature/test",
    });

    const text = String(result);
    expect(text).toContain("PLATFORM_PR_FAILED");
    expect(text).toContain("cannot create");
    expect(text).toContain("details");
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

  it("covers retained compatibility success paths for fetch/create/update issue, add-labels, and rate-limit", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");

    setDollarText("fetch-ok");
    await execute({ command: "fetch-issue", issue_number: "8", output_format: "json" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform fetch-issue 8 --format json",
    );

    setDollarText("create-ok");
    await execute({ command: "create-issue", title: "Bug", labels: " , bug, docs , " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform create-issue --title Bug --labels bug,docs",
    );

    setDollarText("update-ok");
    await execute({ command: "update-issue", issue_number: "9", labels: " , triage , " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform update-issue 9 --labels triage",
    );

    setDollarText("labels-ok");
    await execute({ command: "add-labels", issue_number: "10", labels: " bug , docs " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform add-labels 10 --labels bug,docs",
    );

    setDollarText("rate-ok");
    await execute({ command: "rate-limit", output_format: "json" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform rate-limit --format json",
    );
  });

  it("fails closed before spawn for retained compatibility invalid label inputs", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    const before = getInvocations().length;

    const noOpUpdate = await execute({ command: "update-issue", issue_number: "9", labels: " , , " });
    assertErrorPrefix(String(noOpUpdate), "ERROR:");
    assertContains(String(noOpUpdate), "Provide at least one field to update");

    const missingRequiredLabels = await execute({ command: "add-labels", issue_number: "10", labels: " , , " });
    assertErrorPrefix(String(missingRequiredLabels), "ERROR:");
    assertContains(String(missingRequiredLabels), "must contain at least one label");

    expect(getInvocations().length).toBe(before);
  });

  it("returns parse-safe json failures for retained compatibility commands", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    setDollarError(
      buildDollarFailure({
        stdout: '{"ok":false,"token":"ghp_secretsecretsecret"}',
        stderr: "ignored",
      }),
    );

    const result = await execute({ command: "fetch-issue", issue_number: "8", output_format: "json" });
    expect(JSON.parse(String(result))).toEqual({ ok: false, token: "[REDACTED]" });
  });

  it("fails closed for split-only migrated routes before spawn with migration guidance", async () => {
    const execute = await loadToolExecute("../../platform_operations.ts");
    const before = getInvocations().length;
    const unsupported = await execute({ command: "pr-diff" } as any);
    const text = String(unsupported);
    assertErrorPrefix(text, "ERROR:");
    assertContains(
      text,
      "platform_operations compatibility mode does not support 'pr-diff'. Use 'platform_pr_read' instead.",
    );
    expect(getInvocations().length).toBe(before);
  });
});
