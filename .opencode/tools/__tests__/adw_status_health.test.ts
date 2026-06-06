import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse, getInvocations } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_status_health wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("rejects non-array args", async () => {
    const execute = await loadToolExecute("../../adw_status_health.ts");
    const result = await execute({ command: "status", args: "--x" });
    assertContains(String(result), "expected an array");
  });

  it("assembles status call", async () => {
    const execute = await loadToolExecute("../../adw_status_health.ts");
    await execute({ command: "status" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw status");
  });

  it("prefers stderr over stdout in non-zero failure diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_status_health.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ command: "status" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
