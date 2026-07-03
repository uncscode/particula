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

describe("platform_comment_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles comment command with required args", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    await execute({ command: "comment", issue_number: "123", body: "LGTM" });
    const call = getInvocations().at(-1);
    expect(call?.args.join(" ")).toContain("uv run --active adw platform comment 123 --body LGTM");
  });

  it("rejects all-zero issue token", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    const result = await execute({ command: "comment", issue_number: "000", body: "x" });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "positive integer token");
  });

  it("validates required body", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    const result = await execute({ command: "comment", issue_number: "123", body: "   " });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "'body' is required");
  });

  it("rejects missing and whitespace issue_number before spawn", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    const before = getInvocations().length;

    const missing = await execute({ command: "comment", body: "ok" } as any);
    assertErrorPrefix(String(missing), "ERROR:");
    assertContains(String(missing), "'issue_number' is required");

    const whitespace = await execute({ command: "comment", issue_number: "   ", body: "ok" });
    assertErrorPrefix(String(whitespace), "ERROR:");
    assertContains(String(whitespace), "'issue_number' is required");

    expect(getInvocations().length).toBe(before);
  });

  it("validates prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    const result = await execute({
      command: "comment",
      issue_number: "123",
      body: "ok",
      prefer_scope: " ",
    });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "'prefer_scope' must be either 'fork' or 'upstream'");
  });

  it("help mode bypasses required args", async () => {
    setDollarText("usage");
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    const result = await execute({ command: "comment", help: true });
    expect(String(result)).toContain("usage");
  });

  it("help-path failure prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    setDollarError(buildDollarFailure({ stderr: "help stderr", stdout: "help stdout" }));
    const result = await execute({ command: "comment", help: true });
    const text = String(result);
    assertContains(text, "help stderr");
    expect(text.indexOf("help stderr")).toBeLessThan(text.indexOf("help stdout"));
  });

  it("redacts and truncates sensitive diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    const longToken = "ghp_" + "a".repeat(10000);
    setDollarError(buildDollarFailure({ stderr: `Authorization: Bearer ${longToken}` }));
    const result = await execute({ command: "comment", issue_number: "123", body: "ok" });
    const text = String(result);
    assertContains(text, "[REDACTED]");
    expect(text.length).toBeLessThan(longToken.length);
    expect(text).not.toContain(longToken);
  });

  it("prefers stderr then stdout for diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));
    const result = await execute({ command: "comment", issue_number: "123", body: "ok" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text.indexOf("fatal stderr")).toBeLessThan(text.indexOf("shadow stdout"));
  });

  it("falls back to stdout when stderr empty", async () => {
    const execute = await loadToolExecute("../../platform_comment_write.ts");
    setDollarError(buildDollarFailure({ stderr: "", stdout: "stdout diagnostic" }));
    const result = await execute({ command: "comment", issue_number: "123", body: "ok" });
    assertContains(String(result), "stdout diagnostic");
  });
});
