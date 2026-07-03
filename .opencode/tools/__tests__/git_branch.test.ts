import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarText,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
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

  it("blocks protected checkout create", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "checkout", branch: "main", create: true });

    assertContains(String(result), "protected branch");
    expect(getInvocations()).toHaveLength(0);
  });

  it("fails closed for source without create and does not spawn", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const before = getInvocations().length;
    const result = await execute({ command: "checkout", branch: "feat-1", source: "origin/main" });
    assertContains(String(result), "source");
    expect(getInvocations().length).toBe(before);
  });

  it("rejects blank source when explicitly provided", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "checkout", branch: "feat-1", create: true, source: "   " });

    assertContains(String(result), "requires non-empty 'source' when provided");
    expect(getInvocations()).toHaveLength(0);
  });


  it("assembles checkout command", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    await execute({ command: "checkout", branch: "feat-1", create: true, source: "origin/main" });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("uv run --active adw git checkout --branch feat-1 --source origin/main --create");
  });

  it("normalizes refs/heads prefixes for checkout branch and source", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    await execute({
      command: "checkout",
      branch: "refs/heads/feat-1",
      create: true,
      source: "refs/heads/origin/main",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git checkout --branch feat-1 --source origin/main --create",
    );
  });

  it("rejects malformed branch refs", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push", branch: "bad ref" });

    assertContains(String(result), "Invalid branch: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed sources", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "checkout", branch: "feat-1", create: true, source: "bad ref" });

    assertContains(String(result), "Invalid source: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects option-like worktree paths", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push", branch: "feat-1", worktree_path: "--repo=/tmp/x" });

    assertContains(String(result), "Invalid worktree_path: --repo=/tmp/x.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("assembles push command", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    await execute({ command: "push", branch: "feat-1" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git push --branch feat-1");
  });

  it("allows help mode without branch validation", async () => {
    setDollarText("usage");
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push", help: true });

    expect(String(result)).toContain("Git Command: push (help)");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git push --help");
  });

  it("keeps branch guardrails active in help mode", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push", branch: "bad ref", help: true });

    assertContains(String(result), "Invalid branch: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("keeps worktree guardrails active in help mode", async () => {
    const execute = await loadToolExecute("../../git_branch.ts");
    const result = await execute({ command: "push", worktree_path: "--repo=/tmp/x", help: true });

    assertContains(String(result), "Invalid worktree_path: --repo=/tmp/x.");
    expect(getInvocations()).toHaveLength(0);
  });
});
