import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("git_merge wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires source for merge", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "merge" });
    assertContains(String(result), "requires 'source'");
  });

  it("requires branch for rebase", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "rebase" });
    assertContains(String(result), "requires 'branch'");
  });

  it("assembles fetch", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    await execute({ command: "fetch" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw git fetch");
  });

  it("requires ref for reset", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "reset" });
    assertContains(String(result), "requires 'ref'");
  });

});
