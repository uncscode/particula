import assert from "node:assert";
import { beforeEach, describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const gitOperationsTool = (await import("../git_operations")).default;

const calls: any[] = [];

function setBunStub({ output = "ok", error }: { output?: string; error?: any } = {}) {
  calls.length = 0;
  (globalThis as any).Bun = {
    $: (...args: any[]) => {
      calls.push(args);
      if (error) {
        throw error;
      }
      return {
        text: async () => output,
      };
    },
  };
}

describe("git_operations tool - sync commands", () => {
  beforeEach(() => {
    setBunStub();
  });

  it("requires source for merge", async () => {
    const result = await gitOperationsTool.execute({ command: "merge" });
    assert.equal(result, "ERROR: 'merge' command requires 'source'.");
  });

  it("assembles merge command with flags and worktree", async () => {
    await gitOperationsTool.execute({
      command: "merge",
      source: "main",
      target: "develop",
      no_ff: true,
      abort_on_conflict: false,
      worktree_path: "./trees/abc",
    });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "merge",
      "main",
      "--into",
      "develop",
      "--no-ff",
      "--no-abort-on-conflict",
      "--worktree-path",
      "./trees/abc",
    ]);
  });

  it("requires branch for rebase", async () => {
    const result = await gitOperationsTool.execute({ command: "rebase" });
    assert.equal(result, "ERROR: 'rebase' command requires 'branch'.");
  });

  it("assembles rebase command with onto and abort flag", async () => {
    await gitOperationsTool.execute({
      command: "rebase",
      branch: "topic",
      onto: "main",
      abort_on_conflict: false,
      worktree_path: "./trees/abc",
    });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "rebase",
      "topic",
      "--onto",
      "main",
      "--no-abort-on-conflict",
      "--worktree-path",
      "./trees/abc",
    ]);
  });

  it("defaults fetch remote to origin and supports prune", async () => {
    await gitOperationsTool.execute({ command: "fetch", branch: "dev", prune: true });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("--remote"));
    assert.ok(cmd.includes("origin"));
    assert.ok(cmd.includes("--branch"));
    assert.ok(cmd.includes("dev"));
    assert.ok(cmd.includes("--prune"));
  });

  it("assembles sync with source, target, and worktree", async () => {
    await gitOperationsTool.execute({
      command: "sync",
      source: "upstream",
      target: "develop",
      worktree_path: "./trees/abc",
    });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "sync",
      "--source",
      "upstream",
      "--target",
      "develop",
      "--worktree-path",
      "./trees/abc",
    ]);
  });

  it("forwards worktree for abort without required params", async () => {
    await gitOperationsTool.execute({ command: "abort", worktree_path: "./trees/abc" });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "abort",
      "--worktree-path",
      "./trees/abc",
    ]);
  });

  it("requires ref for reset and supports --hard", async () => {
    const missing = await gitOperationsTool.execute({ command: "reset" });
    assert.equal(missing, "ERROR: 'reset' command requires 'ref'.");

    await gitOperationsTool.execute({ command: "reset", ref: "HEAD~1", hard: true });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("--ref"));
    assert.ok(cmd.includes("HEAD~1"));
    assert.ok(cmd.includes("--hard"));
  });

  it("blocks push-force-with-lease to protected branches", async () => {
    const result = await gitOperationsTool.execute({ command: "push-force-with-lease", branch: "main" });
    assert.equal(result, "ERROR: push-force-with-lease to protected branch is blocked.");
  });

  it("assembles push-force-with-lease for feature branch", async () => {
    await gitOperationsTool.execute({
      command: "push-force-with-lease",
      branch: "feature-123",
      worktree_path: "./trees/abc",
    });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "push-force-with-lease",
      "--branch",
      "feature-123",
      "--worktree-path",
      "./trees/abc",
    ]);
  });

  it("forwards max_count and oneline for log", async () => {
    await gitOperationsTool.execute({ command: "log", max_count: 3, oneline: true });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "log",
      "--max-count",
      "3",
      "--oneline",
    ]);
  });

  it("defaults max_count and omits oneline when not set", async () => {
    await gitOperationsTool.execute({ command: "log" });

    const cmd = calls[0][1] as string[];
    const maxCountIndex = cmd.indexOf("--max-count");
    assert.ok(maxCountIndex > -1);
    assert.equal(cmd[maxCountIndex + 1], "10");
    assert.equal(cmd.includes("--oneline"), false);
  });

  it("includes --path when provided for show", async () => {
    await gitOperationsTool.execute({ command: "show", ref: "HEAD", path: "README.md" });

    const cmd = calls[0][1] as string[];
    assert.deepEqual(cmd, [
      "uv",
      "run",
      "adw",
      "git",
      "show",
      "--ref",
      "HEAD",
      "--path",
      "README.md",
    ]);
  });

  it("omits --path when not provided for show", async () => {
    await gitOperationsTool.execute({ command: "show", ref: "HEAD" });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("--ref"));
    assert.equal(cmd.includes("--path"), false);
  });

  it("appends --help when help flag is set and bypasses validation", async () => {
    await gitOperationsTool.execute({ command: "merge", help: true });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("--help"));
    assert.ok(cmd.includes("merge"));
  });
});
