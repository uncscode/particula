import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import {
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

describe("git_commit wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires non-empty summary", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "   " });
    assertContains(String(result), "requires non-empty 'summary'");
  });

  it("documents the narrow wrapper boundary distinctly from native protected git", async () => {
    await loadToolExecute("../../git_commit.ts");

    const description = getCapturedToolDefinition().description ?? "";
    expect(description).toContain("narrow OpenCode adw git commit path");
    expect(description).toContain("distinct from the native TUI/runtime protected-git");
    expect(description).toContain("does not imply arbitrary git verbs, shell execution, auto-push, or");
  });

  it("validates max_retries", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "msg", max_retries: 99 });
    assertContains(String(result), "between 0 and 10");
  });

  it("assembles commit with summary", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({ summary: "test commit" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git commit --summary test commit",
    );
  });

  it("assembles description adw_id worktree_path and stage_all with default retries", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({
      summary: "test commit",
      description: "body",
      adw_id: "dac13a15",
      worktree_path: "./trees/abc",
      stage_all: true,
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git commit --summary test commit --description body --adw-id dac13a15 --worktree-path ./trees/abc --stage-all --max-retries 3",
    );
  });

  it("passes through no_verify and zero retries", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({ summary: "test commit", no_verify: true, max_retries: 0 });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git commit --summary test commit --no-verify --max-retries 0",
    );
  });

  it("rejects non-boolean no_verify values", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "msg", no_verify: "true" });

    assertContains(String(result), "'no_verify' must be a boolean");
  });

  it("rejects option-like worktree paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "msg", worktree_path: "--repo=/tmp/x" });

    assertContains(String(result), "'worktree_path' cannot start with '-'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("does not emit flags for false optional booleans", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({ summary: "test commit", no_verify: false, stage_all: false });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(command).toContain("--max-retries 3");
    expect(command).not.toContain("--no-verify");
    expect(command).not.toContain("--stage-all");
  });

  it("passes through committed status and sha beneath wrapper prefix", async () => {
    setDollarText("✅ Success\nstatus=committed\nsha=abc123def456\ncreated");
    const execute = await loadToolExecute("../../git_commit.ts");

    const result = await execute({ summary: "test commit" });

    expect(String(result)).toContain(
      "Git Commit Command\n\n✅ Success\nstatus=committed\nsha=abc123def456\ncreated",
    );
  });

  it("passes through no-op status without sha", async () => {
    setDollarText("✅ Success\nstatus=no_op\nNo changes to commit");
    const execute = await loadToolExecute("../../git_commit.ts");

    const result = await execute({ summary: "test commit" });

    expect(String(result)).toContain("status=no_op");
    expect(String(result)).not.toContain("sha=");
  });

  it("preserves deterministic failure envelope", async () => {
    setDollarError({ stderr: "commit failed", message: "spawn failed" });
    const execute = await loadToolExecute("../../git_commit.ts");

    const result = await execute({ summary: "test commit" });

    assertContains(String(result), "ERROR: Failed to execute 'adw git commit'");
    assertContains(String(result), "stderr: commit failed");
  });

});
