import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("search_content wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires contentPattern", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({});
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "contentPattern");
  });

  it("validates numeric guardrail", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "x", beforeContext: -1 });
    assertContains(String(result), "not supported by search_content");
  });

  it("assembles rg content search", async () => {
    setSpawnResponse({ stdout: "a:1:x\n", exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({ contentPattern: "x", path: "." });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("rg -n -e x");
  });

  it("prefers stderr over stdout on non-zero exit", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ contentPattern: "x" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
