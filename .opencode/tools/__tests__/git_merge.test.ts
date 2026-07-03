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
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git fetch");
  });

  it("assembles abort and continue commands", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");

    await execute({ command: "abort" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git abort");

    await execute({ command: "continue" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git continue");
  });

  it("requires ref for reset", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "reset" });
    assertContains(String(result), "requires 'ref'");
  });

  it("assembles merge with target no_ff and explicit false abort handling", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    await execute({
      command: "merge",
      source: "main",
      target: "develop",
      no_ff: true,
      abort_on_conflict: false,
      worktree_path: "./trees/abc",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git merge main --into develop --no-ff --no-abort-on-conflict --worktree-path ./trees/abc",
    );
  });

  it("assembles rebase with onto and explicit false abort handling", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    await execute({ command: "rebase", branch: "feat-1", onto: "main", abort_on_conflict: false });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git rebase feat-1 --onto main --no-abort-on-conflict",
    );
  });

  it("assembles fetch default remote with prune", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    await execute({ command: "fetch", prune: true });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git fetch --remote origin --prune",
    );
  });

  it("assembles sync with source and target", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    await execute({ command: "sync", source: "upstream", target: "main" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git sync --source upstream --target main",
    );
  });

  it("rejects malformed sync source refs", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "sync", source: "bad ref", target: "main" });

    assertContains(String(result), "Invalid source: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed sync target refs", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "sync", source: "upstream", target: "bad ref" });

    assertContains(String(result), "Invalid target: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed command refs across guarded fields", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");

    const cases = [
      {
        name: "rebase.onto",
        args: { command: "rebase", branch: "feat-1", onto: "bad ref" },
        message: "Invalid onto: bad ref.",
      },
      {
        name: "fetch.branch",
        args: { command: "fetch", branch: "bad ref" },
        message: "Invalid branch: bad ref.",
      },
      {
        name: "accumulate.tracking_branch",
        args: { command: "accumulate", slice_branch: "issue-1", tracking_branch: "bad ref" },
        message: "Invalid tracking_branch: bad ref.",
      },
      {
        name: "merge.source",
        args: { command: "merge", source: "bad ref" },
        message: "Invalid source: bad ref.",
      },
    ] as const;

    for (const testCase of cases) {
      const result = await execute(testCase.args);
      assertContains(String(result), testCase.message);
      expect(getInvocations(), testCase.name).toHaveLength(0);
    }
  });

  it("assembles reset with hard flag", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    await execute({ command: "reset", ref: "HEAD~1", hard: true });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git reset --ref HEAD~1 --hard",
    );
  });

  it("returns raw trimmed accumulate payload and assembles required flags", async () => {
    setDollarText('{"ok":true}\n');
    const execute = await loadToolExecute("../../git_merge.ts");

    const result = await execute({
      command: "accumulate",
      slice_branch: "issue-1",
      tracking_branch: "accumulate/F1",
      recover_missing_worktree: true,
    });

    expect(String(result)).toBe('{"ok":true}');
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git accumulate --slice-branch issue-1 --tracking-branch accumulate/F1 --json --recover-missing-worktree",
    );
  });

  it("rejects blank accumulate identifiers before spawn", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "accumulate", slice_branch: "   ", tracking_branch: "track" });

    assertContains(String(result), "requires 'slice_branch'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects option-like worktree paths", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "fetch", worktree_path: "--repo=/tmp/x" });

    assertContains(String(result), "Invalid worktree_path: --repo=/tmp/x.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed merge targets", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "merge", source: "main", target: "bad ref" });

    assertContains(String(result), "Invalid target: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects invalid reset refs before spawning", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "reset", ref: "bad ref" });

    assertContains(String(result), "Invalid ref: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns raw trimmed accumulate help output without required argument validation", async () => {
    setDollarText('{"usage":"accumulate"}\n');
    const execute = await loadToolExecute("../../git_merge.ts");

    const result = await execute({ command: "accumulate", help: true });

    expect(String(result)).toBe('{"usage":"accumulate"}');
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git accumulate --json --help",
    );
  });

  it("keeps ref guardrails active in help mode", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "reset", ref: "bad ref", help: true });

    assertContains(String(result), "Invalid ref: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("keeps worktree guardrails active in help mode", async () => {
    const execute = await loadToolExecute("../../git_merge.ts");
    const result = await execute({ command: "fetch", worktree_path: "--repo=/tmp/x", help: true });

    assertContains(String(result), "Invalid worktree_path: --repo=/tmp/x.");
    expect(getInvocations()).toHaveLength(0);
  });

});
