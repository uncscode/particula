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

describe("platform_issue_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles fetch-issue with format and scope", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    await execute({
      command: "fetch-issue",
      issue_number: "123",
      output_format: "json",
      prefer_scope: "upstream",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform fetch-issue 123 --format json --prefer-scope upstream",
    );
  });

  it("requires issue_number for missing and whitespace-only values", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    const before = getInvocations().length;

    const missing = await execute({ command: "fetch-issue" });
    assertContains(String(missing), "'issue_number' is required");

    const whitespace = await execute({ command: "fetch-issue", issue_number: "   " });
    assertContains(String(whitespace), "'issue_number' is required");

    expect(getInvocations().length).toBe(before);
  });

  it("rejects zero-only issue tokens", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    const result = await execute({ command: "fetch-issue", issue_number: "000" });
    assertContains(String(result), "positive integer token");
  });

  it("validates optional output_format and prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");

    const badFormat = await execute({
      command: "fetch-issue",
      issue_number: "1",
      output_format: "yaml",
    });
    assertContains(String(badFormat), "'output_format' must be either 'text' or 'json'");

    const badScope = await execute({ command: "fetch-issue", issue_number: "1", prefer_scope: " " });
    assertContains(String(badScope), "'prefer_scope' must be either");
  });

  it("help mode bypasses required validation", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    setDollarText("usage");
    const result = await execute({ command: "fetch-issue", help: true });
    expect(String(result)).toContain("usage");
  });

  it("prefers stderr diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal", stdout: "shadow" }));
    const result = await execute({ command: "fetch-issue", issue_number: "1" });
    const text = String(result);

    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "fatal");
    expect(text.indexOf("fatal")).toBeLessThan(text.indexOf("shadow"));
  });

  it("returns redacted structured stdout payload for json failures", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    setDollarError(
      buildDollarFailure({ stdout: '{"ok":false,"token":"ghp_secretsecretsecret"}', stderr: "ignored" }),
    );
    const result = await execute({ command: "fetch-issue", issue_number: "1", output_format: "json" });
    expect(JSON.parse(String(result))).toEqual({ ok: false, token: "[REDACTED]" });
  });

  it("falls back to the deterministic error envelope for non-json json-mode failures", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    setDollarError(buildDollarFailure({ stdout: "not-json", stderr: "fatal stderr" }));
    const result = await execute({ command: "fetch-issue", issue_number: "1", output_format: "json" });
    const text = String(result);

    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "Failed to execute 'adw platform fetch-issue'");
    assertContains(text, "fatal stderr");
  });
});
