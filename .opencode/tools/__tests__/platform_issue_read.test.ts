import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setDollarError, setDollarText } from "./helpers/mock-subprocess";
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

  it("requires issue_number", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    const result = await execute({ command: "fetch-issue" });
    assertContains(String(result), "'issue_number' is required");
  });

  it("validates prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    const result = await execute({ command: "fetch-issue", issue_number: "1", prefer_scope: " " });
    assertContains(String(result), "'prefer_scope' must be either");
  });

  it("prefers stderr diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    setDollarError({ stderr: "fatal", stdout: "shadow" });
    const result = await execute({ command: "fetch-issue", issue_number: "1" });
    assertContains(String(result), "fatal");
  });

  it("assembles fetch-issue", async () => {
    const execute = await loadToolExecute("../../platform_issue_read.ts");
    await execute({ command: "fetch-issue", issue_number: "123" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw platform fetch-issue 123");
  });
});
