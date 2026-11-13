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

describe("platform_label_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles add-labels with normalized labels", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    await execute({
      command: "add-labels",
      issue_number: "1",
      labels: " bug, docs ,triage ",
      prefer_scope: "fork",
      output_format: "json",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform add-labels 1 --labels bug,docs,triage --prefer-scope fork --format json",
    );
  });

  it("assembles remove-labels", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    await execute({ command: "remove-labels", issue_number: "2", labels: "bug" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform remove-labels 2 --labels bug",
    );
  });

  it("requires issue_number and labels", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");

    const missingIssue = await execute({ command: "add-labels", labels: "x" });
    assertContains(String(missingIssue), "'issue_number' is required");

    const missingLabels = await execute({ command: "remove-labels", issue_number: "1" });
    assertContains(String(missingLabels), "'labels' is required");
  });

  it("rejects normalization-to-empty labels and invalid issue token", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");

    const emptyLabels = await execute({ command: "add-labels", issue_number: "1", labels: " , , " });
    assertContains(String(emptyLabels), "must contain at least one label");

    const zeroIssue = await execute({ command: "add-labels", issue_number: "000", labels: "x" });
    assertContains(String(zeroIssue), "positive integer token");
  });

  it("validates optional output_format and prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");

    const badFormat = await execute({
      command: "add-labels",
      issue_number: "1",
      labels: "x",
      output_format: "yaml",
    });
    assertContains(String(badFormat), "'output_format' must be either 'text' or 'json'");

    const badScope = await execute({
      command: "add-labels",
      issue_number: "1",
      labels: "x",
      prefer_scope: " ",
    });
    assertContains(String(badScope), "'prefer_scope' must be either 'fork' or 'upstream'");
  });

  it("help mode bypasses required validation", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    setDollarText("usage");
    const result = await execute({ command: "add-labels", help: true });
    expect(String(result)).toContain("usage");
  });

  it("failure envelope prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));
    const result = await execute({ command: "add-labels", issue_number: "1", labels: "x" });
    const text = String(result);

    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "fatal stderr");
    expect(text.indexOf("fatal stderr")).toBeLessThan(text.indexOf("shadow stdout"));
  });

  it("returns sanitized structured stdout for json failures", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    setDollarError(
      buildDollarFailure({
        stdout: '{"ok":false,"token":"ghp_secretsecretsecret"}',
        stderr: "ignored",
      }),
    );
    const result = await execute({
      command: "add-labels",
      issue_number: "1",
      labels: "x",
      output_format: "json",
    });
    expect(JSON.parse(String(result))).toEqual({ ok: false, token: "[REDACTED]" });
  });

  it("falls back to error envelope when json failure stdout is not structured json", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    setDollarError(buildDollarFailure({ stdout: "not-json", stderr: "fatal stderr" }));
    const result = await execute({
      command: "add-labels",
      issue_number: "1",
      labels: "x",
      output_format: "json",
    });
    const text = String(result);
    assertContains(text, "Failed to execute 'adw platform add-labels'");
    assertContains(text, "fatal stderr");
  });
});
