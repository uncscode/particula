import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
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

  it("assembles status and health calls", async () => {
    const execute = await loadToolExecute("../../adw_status_health.ts");
    await execute({ command: "status" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw status");

    await execute({ command: "health" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw health");
  });

  it("help emits the direct command help path", async () => {
    const execute = await loadToolExecute("../../adw_status_health.ts");
    const result = await execute({ command: "status", help: true });
    expect(String(result)).toContain("ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw status --help");
  });

  it("ignores unknown extra payload fields and preserves runtime contract", async () => {
    const execute = await loadToolExecute("../../adw_status_health.ts");
    await execute({ command: "status", adw_id: "a1b2c3d4" } as { command: string; adw_id: string });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw status");
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
