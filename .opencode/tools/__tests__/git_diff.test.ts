import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const selectedRoot = process.cwd();

describe("git_diff wrapper", () => {
  beforeEach(() => { installSubprocessMocks(); resetSubprocessMocks(); resetCapturedToolDefinition(); });
  afterEach(() => { restoreSubprocessMocks(); resetCapturedToolDefinition(); });

  it("invokes only the local Python diagnostics adapter", async () => {
    setSpawnResponse({ stdout: JSON.stringify({ ok: true, data: { stdout: "", status: "clean" } }) });
    const execute = await loadToolExecute("../../git_diff.ts");
    expect(await execute({ command: "status", worktree_path: selectedRoot })).toContain("Git Command: status (clean)");
    const argv = getInvocations().at(-1)?.args ?? [];
    expect(argv[0]).toBe("python3");
    expect(argv[1]).toContain("read_only_git_diagnostics.py");
    expect(argv.join(" ")).not.toContain(" adw ");
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

    expect(await execute({ command: "status", worktree_path: "-untrusted" })).toBe(
      "ERROR: invalid or untrusted worktree_path",
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

  it("requires an explicit selected root without a cwd fallback", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "status" })).toBe("ERROR: invalid or untrusted worktree_path");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects relative and missing directories before adapter admission", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");

    expect(await execute({ command: "status", worktree_path: "." })).toBe(
      "ERROR: invalid or untrusted worktree_path",
    );
    expect(await execute({ command: "status", worktree_path: `${selectedRoot}/definitely-missing` })).toBe(
      "ERROR: invalid or untrusted worktree_path",
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
});
