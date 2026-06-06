import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_service wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("rejects force on launch", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    const result = await execute({ command: "launch", force: true });
    assertContains(String(result), "only supported for command 'stop'");
  });

  it("assembles stop force", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    await execute({ command: "stop", force: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw stop --force");
  });

  it("prefers stderr over stdout in non-zero failure diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ command: "stop" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
