import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("git_worktree wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires adw_id for worktree-remove", async () => {
    const execute = await loadToolExecute("../../git_worktree.ts");
    const result = await execute({ command: "worktree-remove" });
    assertContains(String(result), "requires 'adw_id'");
  });

  it("assembles worktree-list", async () => {
    const execute = await loadToolExecute("../../git_worktree.ts");
    await execute({ command: "worktree-list" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw git worktree list");
  });

});
