import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse, getInvocations } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_workflow wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires issue_number unless help", async () => {
    const execute = await loadToolExecute("../../adw_workflow.ts");
    const result = await execute({ command: "build" });
    assertContains(String(result), "requires 'issue_number'");
  });

  it("validates adw_id shape", async () => {
    const execute = await loadToolExecute("../../adw_workflow.ts");
    const result = await execute({ command: "build", issue_number: 1, adw_id: "xyz" });
    assertContains(String(result), "8-character hexadecimal");
  });

  it("assembles workflow command", async () => {
    const execute = await loadToolExecute("../../adw_workflow.ts");
    await execute({ command: "build", issue_number: 42, adw_id: "A1B2C3D4" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw build 42 --adw-id a1b2c3d4");
  });

  it("help bypasses issue_number requirement", async () => {
    const execute = await loadToolExecute("../../adw_workflow.ts");
    const result = await execute({ command: "build", help: true });
    expect(String(result)).toContain("ok");
  });

  it("prefers stderr over stdout in non-zero failure diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_workflow.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ command: "build", issue_number: 42 });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
