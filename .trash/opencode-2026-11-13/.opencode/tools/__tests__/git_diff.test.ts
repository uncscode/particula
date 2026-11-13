import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, realpathSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnError,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const selectedRoot = process.cwd();
const testCheckout = realpathSync(join(import.meta.dir, "..", "..", ".."));

const runGit = (worktreePath: string, args: string[]): string => {
  const result = Bun.spawnSync({
    cmd: ["git", "-C", worktreePath, ...args],
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode !== 0) throw new Error("Git test setup failed");
  return Buffer.from(result.stdout).toString("utf8");
};

const resolvePrimaryCheckout = (): string => {
  const commonGitDir = runGit(testCheckout, ["rev-parse", "--git-common-dir"]).trim();
  if (!commonGitDir) throw new Error("Git test setup failed");
  return realpathSync(resolve(testCheckout, commonGitDir, ".."));
};

const expectReadOnlyDiagnostics = async (
  execute: (args: Record<string, unknown>) => Promise<string> | string,
  worktreePath: string,
): Promise<void> => {
  expect(await execute({ command: "status", worktree_path: worktreePath })).toMatch(
    /^Git Command: status \((clean|changed)\)/,
  );
  expect(await execute({ command: "log", ref: "HEAD", max_count: 1, worktree_path: worktreePath })).toMatch(
    /^Git Command: log \(ok\)/,
  );
  expect(await execute({ command: "diff", base: "HEAD", target: "HEAD", worktree_path: worktreePath })).toMatch(
    /^Git Command: diff \(no_diff\)/,
  );
};

describe("git_diff wrapper", () => {
  beforeEach(() => { installSubprocessMocks(); resetSubprocessMocks(); resetCapturedToolDefinition(); });
  afterEach(() => { restoreSubprocessMocks(); resetCapturedToolDefinition(); });

  it("invokes only the local Python diagnostics adapter", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: true, data: { stdout: "", status: "clean" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");
    expect(await execute({ command: "status", worktree_path: selectedRoot })).toContain("Git Command: status (clean)");
    const invocation = getInvocations().at(-1);
    expect(invocation?.args[0]).toBe("python3");
    expect(invocation?.args[1]).toContain("read_only_git_diagnostics.py");
    expect(invocation?.args.join(" ")).not.toContain(" adw ");
    expect(invocation?.stdin).toBe(JSON.stringify({ command: "status", worktree_path: selectedRoot }));
    expect(invocation?.stdout).toBe("pipe");
    expect(invocation?.stderr).toBe("pipe");
    expect(invocation?.timeout).toBe(30_000);
  });

  it("renders adapter errors without command details", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: false, error: { type: "invalid_request", message: "bad request" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");
    expect(await execute({ command: "diff", worktree_path: selectedRoot })).toBe("ERROR: invalid_request: bad request");
  });

  it("renders malformed adapter output as a bounded failure", async () => {
    setSpawnResponse({ stdout: "not-json" });
    const execute = await loadToolExecute("../../git_diff.ts");
    expect(await execute({ command: "log", worktree_path: selectedRoot })).toBe("ERROR: Git diagnostics adapter returned malformed output");
  });

  it("rejects oversized adapter output before JSON decoding", async () => {
    setSpawnResponse({ stdout: "x".repeat(131_073) });
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "log", worktree_path: selectedRoot })).toBe(
      "ERROR: Git diagnostics adapter returned oversized output",
    );
  });

  it("caps successful adapter output before rendering", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: true, data: { stdout: "x".repeat(13_000), diff: "diff_present" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");

    expect((await execute({ command: "diff", worktree_path: selectedRoot }) as string).length).toBeLessThan(12_100);
  });

  it("does not invoke the adapter when worktree admission fails", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "status", worktree_path: "-untrusted" })).toContain(
      'ERROR: invalid or untrusted worktree_path: "-untrusted"',
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("forwards canonical directories for authoritative adapter admission", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: false, error: { type: "invalid_request", message: "not registered" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "status", worktree_path: "/" })).toBe(
      "ERROR: invalid_request: not registered",
    );
    expect(getInvocations()).toHaveLength(1);
  });

  it("defaults an omitted worktree to the wrapper repository root", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: true, data: { stdout: "", status: "clean" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "status" })).toContain("Git Command: status (clean)");
    expect(getInvocations().at(-1)?.stdin).toBe(
      JSON.stringify({ command: "status", worktree_path: testCheckout }),
    );
  });

  it("resolves relative worktree paths against the wrapper repository root", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: true, data: { stdout: "", status: "clean" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "status", worktree_path: "." })).toContain(
      "Git Command: status (clean)",
    );
    expect(getInvocations().at(-1)?.stdin).toBe(
      JSON.stringify({ command: "status", worktree_path: testCheckout }),
    );
  });

  it("reports the resolved candidate for a missing relative directory", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const missing = join(testCheckout, "definitely-missing");

    expect(await execute({ command: "status", worktree_path: "definitely-missing" })).toBe(
      `ERROR: invalid or untrusted worktree_path: ${JSON.stringify(missing)}`,
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("renders a nonzero adapter exit without exposing child output", async () => {
    setSpawnResponse({ stdout: "sensitive adapter detail", stderr: "more detail", exitCode: 1 });
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "show", ref: "HEAD", worktree_path: selectedRoot })).toBe(
      "ERROR: Git diagnostics adapter failed",
    );
  });

  it("redacts adapter launch exceptions", async () => {
    setSpawnError({
      message: "secret launch detail",
      stdout: "sensitive stdout",
      stderr: "sensitive stderr",
    });
    const execute = await loadToolExecute("../../git_diff.ts");

    const result = await execute({ command: "status", worktree_path: selectedRoot });
    expect(result).toBe("ERROR: Git diagnostics adapter unavailable");
    expect(result).not.toContain("secret launch detail");
    expect(result).not.toContain("sensitive stdout");
    expect(result).not.toContain("sensitive stderr");
  });

  it("launches the local adapter against its primary checkout", async () => {
    restoreSubprocessMocks();
    const execute = await loadToolExecute("../../git_diff.ts");

    await expectReadOnlyDiagnostics(execute, resolvePrimaryCheckout());
  }, 3_000);

  it("launches the local adapter against a registered linked worktree", async () => {
    restoreSubprocessMocks();
    const primary = resolvePrimaryCheckout();
    const temporaryParent = join(
      primary,
      "adforge_local",
      "opencode",
      "tmp",
      `git-diff-linked-${process.pid}-${Date.now()}`,
    );
    const linkedWorktree = join(temporaryParent, "linked");
    let added = false;
    let testFailure: unknown;

    try {
      mkdirSync(temporaryParent, { recursive: true });
      const head = runGit(primary, ["rev-parse", "--verify", "HEAD"]).trim();
      runGit(primary, ["worktree", "add", "--detach", linkedWorktree, head]);
      added = true;
      const execute = await loadToolExecute("../../git_diff.ts");

      await expectReadOnlyDiagnostics(execute, linkedWorktree);
    } catch (error) {
      testFailure = error;
      throw error;
    } finally {
      let cleanupFailure: unknown;
      try {
        if (added) runGit(primary, ["worktree", "remove", "--force", linkedWorktree]);
      } catch (error) {
        cleanupFailure = error;
      }
      try {
        rmSync(temporaryParent, { recursive: true, force: true });
      } catch (error) {
        cleanupFailure ??= error;
      }
      if (cleanupFailure) {
        if (testFailure) console.error("Git test cleanup failed", cleanupFailure);
        else throw cleanupFailure;
      }
    }
  }, 3_000);
});
