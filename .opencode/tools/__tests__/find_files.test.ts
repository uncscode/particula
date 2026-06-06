import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("find_files wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("rejects contentPattern input", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", contentPattern: "x" });
    assertContains(String(result), "contentPattern");
  });

  it("assembles rg --files", async () => {
    setSpawnResponse({ stdout: "a.ts\n", exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    await execute({ pattern: "**/*.ts" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args[0]).toBe("rg");
    expect(args).toContain("--files");
    expect(args).toContain("--glob");
    expect(args[args.indexOf("--glob") + 1]).toBe("**/*.ts");
  });

  it("prefers stderr over stdout on non-zero exit", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ pattern: "**/*.ts" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
