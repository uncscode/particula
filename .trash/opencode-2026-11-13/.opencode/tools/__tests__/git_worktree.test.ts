import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarText,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
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
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git worktree list");
  });

  it("rejects non-canonical adw_id", async () => {
    const execute = await loadToolExecute("../../git_worktree.ts");
    const result = await execute({ command: "worktree-remove", adw_id: "ABC12345" });

    assertContains(String(result), "Expected 8 lowercase hex characters");
    expect(getInvocations()).toHaveLength(0);
  });

  it("defaults worktree-remove to force", async () => {
    const execute = await loadToolExecute("../../git_worktree.ts");
    await execute({ command: "worktree-remove", adw_id: "abc12345" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git worktree remove abc12345 --force",
    );
  });

  it("omits force when force is false", async () => {
    const execute = await loadToolExecute("../../git_worktree.ts");
    await execute({ command: "worktree-remove", adw_id: "abc12345", force: false });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git worktree remove abc12345",
    );
    expect(getInvocations().at(-1)?.args.join(" ")).not.toContain("--force");
  });

  it("assembles worktree-prune", async () => {
    const execute = await loadToolExecute("../../git_worktree.ts");
    await execute({ command: "worktree-prune" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git worktree prune");
  });

  it("allows help mode without required adw_id", async () => {
    setDollarText("usage");
    const execute = await loadToolExecute("../../git_worktree.ts");
    const result = await execute({ command: "worktree-remove", help: true });

    expect(String(result)).toContain("Git Command: worktree-remove (help)");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git worktree remove --force --help",
    );
  });

});
