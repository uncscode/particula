import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync } from "node:fs";
import { join } from "node:path";
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
import {
  loadToolExecute,
  loadToolExecuteFromAbsolutePath,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

const normalizePathForAssertion = (value: string): string => value.replaceAll("\\", "/");

describe("run_pytest_basic wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds routine command for success path", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    const result = await execute({ testPath: "adw/core/tests", testFilter: "agent", failFast: true });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("run_pytest.py");
    expect(cmd).toContain("--fail-fast");
    expect(cmd).toContain("-k");
    expect(cmd).toContain("agent");
    expect(cmd).toContain("adw/core/tests");
  });

  it("rejects advanced option keys by presence", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const result = await execute({ coverage: false });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("does not accept advanced option 'coverage'");
  });

  it("rejects testPath beginning with dash", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const result = await execute({ testPath: "--maxfail=1" });

    expect(result).toContain("testPath must not start with '-'");
  });

  it("rejects non-positive numeric guards", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    expect(await execute({ minTests: 0 })).toContain("minTests must be a positive finite number");
    expect(await execute({ timeout: -1 })).toContain("timeout must be a positive finite number");
  });

  it("rejects nonexistent cwd path", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const missingCwd = join(tmpdir(), `run-pytest-missing-${Date.now()}`);

    expect(await execute({ cwd: missingCwd })).toContain(`cwd path does not exist: ${missingCwd}`);
  });

  it("rejects cwd that resolves to a file", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const fileCwd = import.meta.path;

    expect(await execute({ cwd: fileCwd })).toContain(`cwd path is not a directory: ${fileCwd}`);
  });

  it("rejects cwd resolving outside repository root", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const outsideCwd = tmpdir();

    const result = await execute({ cwd: outsideCwd });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain(`cwd path resolves outside repository root: ${outsideCwd}`);
  });

  it("prefers stdout over stderr for failure diagnostics", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));

    const result = await execute({});
    assertContains(String(result), "stdout diagnostic");
    expect(result).not.toContain("stderr shadow");
  });

  it("falls back to stderr then message when stdout absent", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "stderr diagnostic" }));
    expect(await execute({})).toContain("stderr diagnostic");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "", message: "fallback message" }));
    expect(await execute({})).toContain("fallback message");
  });

  it("symlinked .opencode/tools load resolves backing run_pytest.py path deterministically", async () => {
    setDollarText(buildSuccessOutput("ok"));

    const fixtureRoot = mkdtempSync(join(tmpdir(), "adw-tools-symlink-"));
    const fixtureOpencode = join(fixtureRoot, ".opencode");
    const symlinkTools = join(fixtureOpencode, "tools");
    mkdirSync(fixtureOpencode, { recursive: true });
    symlinkSync(join(import.meta.dir, ".."), symlinkTools, "dir");

    try {
      const execute = await loadToolExecuteFromAbsolutePath(join(symlinkTools, "run_pytest_basic.ts"));
      const result = await execute({ testPath: "adw/core/tests", testFilter: "agent", failFast: true });
      expect(result).toBe("ok");

      const invocation = getInvocations().at(-1);
      expect(invocation).toBeDefined();

      const args = invocation?.args ?? [];
      expect(args[0]).toBe("python3");
      expect(args).toContain("--fail-fast");
      expect(args).toContain("-k");
      expect(args).toContain("agent");
      expect(args).toContain("adw/core/tests");

      const scriptArg = args[1] ?? "";
      expect(normalizePathForAssertion(scriptArg)).toContain("/run_pytest.py");
      expect(normalizePathForAssertion(scriptArg)).toContain("/.opencode/tools/");

      setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
      const failedResult = await execute({});
      assertErrorPrefix(String(failedResult), "ERROR:");
      expect(failedResult).toContain("stdout diagnostic");
      expect(failedResult).not.toContain("stderr shadow");
    } finally {
      rmSync(fixtureRoot, { recursive: true, force: true });
    }
  });
});
