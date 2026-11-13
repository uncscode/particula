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

describe("platform_issue_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles create-issue with optional flags in compatibility order", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    await execute({
      command: "create-issue",
      title: "T",
      body: "B",
      labels: "bug",
      prefer_scope: "fork",
      output_format: "json",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform create-issue --title T --body B --labels bug --prefer-scope fork --format json",
    );
  });

  it("assembles update-issue with state and labels", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    await execute({
      command: "update-issue",
      issue_number: "007",
      title: "T",
      body: "B",
      labels: "bug,docs",
      state: "closed",
      prefer_scope: "upstream",
      output_format: "text",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform update-issue 007 --title T --body B --labels bug,docs --state closed --prefer-scope upstream --format text",
    );
  });

  it("omits blank optional fields without masking required validation", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    await execute({
      command: "create-issue",
      title: "T",
      body: "   ",
      labels: "   ",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw platform create-issue --title T",
    );
  });

  it("requires title for create-issue including whitespace-only values", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    const result = await execute({ command: "create-issue", title: "   " });
    assertContains(String(result), "'title' is required");
  });

  it("requires issue_number for update-issue including whitespace-only values", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    const before = getInvocations().length;

    const missing = await execute({ command: "update-issue", title: "x" });
    assertContains(String(missing), "'issue_number' is required");

    const whitespace = await execute({ command: "update-issue", issue_number: "   ", title: "x" });
    assertContains(String(whitespace), "'issue_number' is required");

    expect(getInvocations().length).toBe(before);
  });

  it("rejects all-zero issue identifiers for update-issue", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    const result = await execute({ command: "update-issue", issue_number: "000", title: "x" });
    assertContains(String(result), "positive integer token");
  });

  it("rejects no-op update requests", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    const result = await execute({ command: "update-issue", issue_number: "12", body: "   " });
    assertContains(String(result), "Provide at least one field to update");
  });

  it("validates state and output_format", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");

    const badState = await execute({
      command: "update-issue",
      issue_number: "12",
      title: "x",
      state: "merged",
    });
    assertContains(String(badState), "'state' must be either 'open' or 'closed'");

    const badFormat = await execute({
      command: "create-issue",
      title: "x",
      output_format: "yaml",
    });
    assertContains(String(badFormat), "'output_format' must be either 'text' or 'json'");
  });

  it("help mode bypasses required validation", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    setDollarText("usage");
    const result = await execute({ command: "create-issue", help: true });
    expect(String(result)).toContain("usage");
  });

  it("failure envelope prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));
    const result = await execute({ command: "create-issue", title: "x" });
    const text = String(result);

    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "fatal stderr");
    expect(text.indexOf("fatal stderr")).toBeLessThan(text.indexOf("shadow stdout"));
  });

  it("preserves sanitized stdout payload for json failures", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    setDollarError(buildDollarFailure({ stdout: '{"ok":false,"token":"ghp_secretsecretsecret"}', stderr: "ignored" }));
    const result = await execute({ command: "create-issue", title: "x", output_format: "json" });
    expect(JSON.parse(String(result))).toEqual({ ok: false, token: "[REDACTED]" });
  });

  it("falls back to the deterministic error envelope for non-json json-mode failures", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    setDollarError(buildDollarFailure({ stdout: "not-json", stderr: "fatal stderr" }));
    const result = await execute({ command: "create-issue", title: "x", output_format: "json" });
    const text = String(result);

    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "Failed to execute 'adw platform create-issue'");
    assertContains(text, "fatal stderr");
  });
});
