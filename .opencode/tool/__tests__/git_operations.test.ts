import assert from "node:assert/strict";
import { describe, it, mock } from "bun:test";

const schema = {
  array: () => schema,
  boolean: () => schema,
  default: () => schema,
  describe: () => schema,
  enum: () => schema,
  literal: () => schema,
  number: () => schema,
  object: () => schema,
  optional: () => schema,
  string: () => schema,
  union: () => schema,
};

const toolStub = Object.assign((definition: unknown) => definition, { schema });

mock.module("@opencode-ai/plugin", () => ({
  tool: toolStub,
}));

// Provide a fake Bun.$ implementation to capture command assembly.
const bunGlobal = (globalThis as { Bun?: { $?: (...args: unknown[]) => unknown } }).Bun;
const bunDollar = (_strings: TemplateStringsArray, ...values: unknown[]) => {
  const parts = values[0] as (string | number)[];
  return {
    text: async () => parts.join(" "),
  };
};

if (bunGlobal) {
  bunGlobal.$ = bunDollar;
} else {
  (globalThis as any).Bun = { $: bunDollar };
}

const gitOperationsTool = (await import("../git_operations")).default as {
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("git_operations worktree commands", () => {
  it("builds worktree list command", async () => {
    const result = await gitOperationsTool.execute({ command: "worktree-list" });

    assert.ok(result.includes("uv run adw git worktree list"));
    assert.ok(result.startsWith("Git Command: worktree-list"));
  });

  it("builds worktree prune command", async () => {
    const result = await gitOperationsTool.execute({ command: "worktree-prune" });

    assert.ok(result.includes("uv run adw git worktree prune"));
    assert.ok(result.startsWith("Git Command: worktree-prune"));
  });

  it("builds worktree remove command with force flag by default", async () => {
    const result = await gitOperationsTool.execute({
      command: "worktree-remove",
      adw_id: "abc123",
    });

    assert.ok(result.includes("uv run adw git worktree remove abc123 --force"));
  });

  it("omits force flag when explicitly disabled", async () => {
    const result = await gitOperationsTool.execute({
      command: "worktree-remove",
      adw_id: "abc123",
      force: false,
    });

    assert.ok(result.includes("uv run adw git worktree remove abc123"));
    assert.ok(!result.includes("--force"));
  });

  it("returns error when worktree-remove missing adw_id", async () => {
    const result = await gitOperationsTool.execute({ command: "worktree-remove" });

    assert.equal(result, "ERROR: 'worktree-remove' command requires 'adw_id'.");
  });

  it("bypasses validation in help mode", async () => {
    const result = await gitOperationsTool.execute({ command: "worktree-remove", help: true });

    assert.ok(result.includes("--help"));
    assert.ok(result.startsWith("Git Command: worktree-remove"));
  });
});

describe("git_operations continue command", () => {
  it("builds continue command", async () => {
    const result = await gitOperationsTool.execute({ command: "continue" });

    assert.ok(result.includes("uv run adw git continue"));
    assert.ok(result.startsWith("Git Command: continue"));
  });

  it("includes worktree-path when provided", async () => {
    const result = await gitOperationsTool.execute({
      command: "continue",
      worktree_path: "/tmp/trees/abc",
    });

    assert.ok(result.includes("--worktree-path /tmp/trees/abc"));
  });
});
