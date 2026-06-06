import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { setSpawnResponse, installSubprocessMocks, restoreSubprocessMocks, getInvocations } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("git_branch wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires branch for push", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push" });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "requires 'branch'");
  });

  it("blocks protected force push", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push-force-with-lease", branch: "main" });
    assertContains(String(result), "protected branch");
  });

  it("fails closed for source without create and does not spawn", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const before = getInvocations().length;
    const result = await execute({ command: "checkout", branch: "feat-1", source: "origin/main" });
    assertContains(String(result), "source");
    expect(getInvocations().length).toBe(before);
  });


  it("assembles checkout command", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    await execute({ command: "checkout", branch: "feat-1", create: true, source: "origin/main" });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("uv run adw git checkout --branch feat-1 --source origin/main --create");
  });
});
