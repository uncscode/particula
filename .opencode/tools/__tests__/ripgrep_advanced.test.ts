import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("ripgrep_advanced wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires contentPattern", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({});
    assertContains(String(result), "contentPattern");
  });

  it("validates beforeContext", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "x", beforeContext: -1 });
    assertContains(String(result), "non-negative integer");
  });

  it("assembles content command", async () => {
    setSpawnResponse({ stdout: "a:1:x\n", exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({ contentPattern: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("rg -n -e x");
  });

  it("prefers stderr over stdout on non-zero exit", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ contentPattern: "x" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
