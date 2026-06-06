import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { existsSync } from "node:fs";
import { readFile, rm } from "node:fs/promises";
import { join, resolve } from "node:path";

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
    expect(calls.at(-1)?.args.join(" ")).toContain("uv run adw git status --porcelain");
  });

  it("returns deterministic validation error for show without ref", async () => {
    const execute = await loadToolExecute("../../git_diff.ts");
    const result = await execute({ command: "show" });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("requires 'ref'");
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
