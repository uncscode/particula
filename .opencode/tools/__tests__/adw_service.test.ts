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

  it("supports help flows", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    const result = await execute({ command: "launch", help: true });
    expect(String(result)).toContain("ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw launch --help");
  });

  it("assembles launch mode and background", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    await execute({ command: "launch", mode: "local", background: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw launch --mode local --background");
  });

  it("assembles stop force", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    await execute({ command: "stop", force: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw stop --force");
  });

  it("rejects launch-only options on stop", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    const invocationCount = getInvocations().length;
    assertContains(String(await execute({ command: "stop", mode: "local" })), "only supported for command 'launch'");
    assertContains(String(await execute({ command: "stop", background: true })), "only supported for command 'launch'");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects stop-only options on launch", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    const invocationCount = getInvocations().length;
    const result = await execute({ command: "launch", force: true });
    assertContains(String(result), "only supported for command 'stop'");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects invalid mode values before spawn", async () => {
    const execute = await loadToolExecute("../../adw_service.ts");
    const invocationCount = getInvocations().length;
    const result = await execute({ command: "launch", mode: "bogus" });
    assertContains(String(result), "'mode' must be 'local' or 'remote'");
    expect(getInvocations()).toHaveLength(invocationCount);
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
