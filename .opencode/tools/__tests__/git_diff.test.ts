import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { existsSync } from "node:fs";
import { readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure, buildSuccessOutput } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const findRepoRoot = (): string => {
  let current = resolve(process.cwd());
  while (true) {
    if (existsSync(join(current, "AGENTS.md")) && existsSync(join(current, ".opencode"))) {
      return current;
    }
    const parent = resolve(current, "..");
    if (parent === current) {
      return resolve(process.cwd());
    }
    current = parent;
  }
};

const readDebugLogFromResult = async (result: string): Promise<string> => {
  const match = result.match(/^debug_log: (.+)$/m);
  expect(match).not.toBeNull();
  const logPath = join(findRepoRoot(), match?.[1] ?? "");
  try {
    return await readFile(logPath, "utf8");
  } finally {
    await rm(logPath, { force: true });
  }
};

describe("git_diff wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("returns success envelope for status command", async () => {
    setDollarText(buildSuccessOutput("M file.py"));
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "status", porcelain: true });
    expect(result).toContain("Git Command: status");
    expect(result).toContain("M file.py");

    const calls = getInvocations();
    expect(calls.at(-1)?.args.join(" ")).toContain("uv run --active adw git status --porcelain");
  });

  it("returns deterministic validation error for show without ref", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "show" });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("requires 'ref'");
  });

  it("assembles diff with base target stat and worktree path", async () => {
    setDollarText(buildSuccessOutput("diff"));
    const execute = await loadToolExecute("../../git_diff.ts");
    const trustedWorktree = findRepoRoot();

    await execute({
      command: "diff",
      base: "main",
      target: "feature/test",
      stat: true,
      worktree_path: trustedWorktree,
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      `uv run --active adw git diff --stat --base main --target feature/test --worktree-path ${trustedWorktree}`,
    );
  });

  it("assembles diff with scoped path", async () => {
    setDollarText(buildSuccessOutput("diff"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "diff", path: "adw/git/operations.py" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git diff --path adw/git/operations.py",
    );
  });

  it("keeps blank diff path unscoped", async () => {
    setDollarText(buildSuccessOutput("diff"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "diff", path: "   " });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git diff");
    expect(getInvocations().at(-1)?.args.join(" ")).not.toContain("--path");
  });

  it("uses default max-count for log", async () => {
    setDollarText(buildSuccessOutput("log"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "log", oneline: true });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git log --max-count 10 --oneline",
    );
  });

  it("rejects out-of-range max_count for log", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "log", max_count: 0 });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "'max_count' must be an integer between 1 and 1000.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-integer max_count for log", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "log", max_count: 1.5 });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "'max_count' must be an integer between 1 and 1000.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects max_count above upper bound for log", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "log", max_count: 1001 });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "'max_count' must be an integer between 1 and 1000.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("accepts valid git rev-spec syntax for log refs", async () => {
    setDollarText(buildSuccessOutput("log"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "log", ref: "HEAD@{1}", max_count: 5, oneline: true });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git log --ref HEAD@{1} --max-count 5 --oneline",
    );
  });

  it("rejects malformed log refs", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "log", ref: "bad ref" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid ref: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("omits blank log ref values during sparse normalization", async () => {
    setDollarText(buildSuccessOutput("log"));
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "log", ref: "   " });

    expect(String(result)).toContain("Git Command: log");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git log --max-count 10");
    expect(getInvocations().at(-1)?.args.join(" ")).not.toContain("--ref");
  });

  it("assembles show with path and stat", async () => {
    setDollarText(buildSuccessOutput("show"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "show", ref: "HEAD~1", path: "adw/core/", stat: true });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git show --ref HEAD~1 --path adw/core/ --stat",
    );
  });

  it("accepts valid git rev-spec syntax for diff refs", async () => {
    setDollarText(buildSuccessOutput("diff"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "diff", base: "stash@{1}", target: "HEAD^{tree}" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git diff --base stash@{1} --target HEAD^{tree}",
    );
  });

  it("accepts valid git rev-spec syntax for show refs", async () => {
    setDollarText(buildSuccessOutput("show"));
    const execute = await loadToolExecute("../../git_diff.ts");

    await execute({ command: "show", ref: "HEAD@{1}" });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git show --ref HEAD@{1}",
    );
  });

  it("rejects malformed diff base refs", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "diff", base: "bad ref" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid base: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed show refs", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "show", ref: "bad ref" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid ref: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed diff target refs", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "diff", target: "bad ref" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid target: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed show paths", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "show", ref: "HEAD", path: "--cached" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid path: --cached.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed diff paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "diff", path: "../outside.txt" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid path: ../outside.txt.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects repo-root-equivalent scoped diff paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "diff", path: "." });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid path: .");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects absolute scoped diff paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "diff", path: "/repo/adw/git/operations.py" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid path: /repo/adw/git/operations.py.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects option-like worktree paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "status", worktree_path: "--repo=/tmp/x" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid worktree_path: --repo=/tmp/x.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects unrelated local repository worktree paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "status", worktree_path: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "worktree_path resolves outside repository root or ADW worktree roots");
    expect(getInvocations()).toHaveLength(0);
  });

  it("allows help mode without required show ref", async () => {
    setDollarText("usage: help");
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "show", help: true });

    expect(String(result)).toContain("Git Command: show (help)");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git show --help");
  });

  it("keeps ref guardrails active in help mode", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "show", ref: "bad ref", help: true });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid ref: bad ref.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("keeps path guardrails active in help mode", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "show", ref: "HEAD", path: "--cached", help: true });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Invalid path: --cached.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("keeps max_count bounds active in help mode", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "log", max_count: 0, help: true });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "'max_count' must be an integer between 1 and 1000.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("prefers stderr for failure diagnostics", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));

    const result = await execute({ command: "status" });
    assertContains(String(result), "fatal stderr");
  });

  it("falls back to stdout when stderr is empty", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    setDollarError(buildDollarFailure({ stderr: "", stdout: "stdout diagnostic" }));

    const result = await execute({ command: "status" });
    assertContains(String(result), "stdout diagnostic");
    expect(String(result)).toContain("Git Command Failed:\nstdout diagnostic");
  });

  it("falls back to message when stderr/stdout are empty", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    setDollarError(buildDollarFailure({ stderr: "", stdout: "", message: "fallback message" }));

    const result = await execute({ command: "status" });
    assertContains(String(result), "fallback message");
  });

  it("writes full failure context to repo-local debug log", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const longTraceback = `Traceback ${"frame ".repeat(160)} root cause`;
    setDollarError(buildDollarFailure({ stderr: longTraceback, stdout: "shadow stdout" }));

    const result = String(await execute({ command: "status" }));

    expect(result).toContain("Git Command Failed: status");
    expect(result).toContain("... [truncated]");
    expect(result).toContain("debug_log: adforge_local/opencode/tmp/git_diff-status-");

    const debugLog = await readDebugLogFromResult(result);
    expect(debugLog).toContain(longTraceback);
    expect(debugLog).toContain("stdout:\nshadow stdout");
  });
});
