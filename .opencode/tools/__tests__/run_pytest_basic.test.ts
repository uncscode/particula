import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
import { COMPACT_SCHEMA_FIELD_FIXTURES, ERROR_PRECEDENCE_FIXTURES } from "./fixtures/wrapper_contract_fixtures";
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
  assertCountedAndExemptFields,
  assertPublicSchemaOmitsKeys,
  getCapturedToolDefinition,
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

    const result = await execute({
      testPath: "adw/core/tests",
      options: 'output=full test-filter="agent smoke" fail-fast',
    });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("run_pytest.py");
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--fail-fast");
    expect(cmd).toContain("-k");
    expect(cmd).toContain("agent smoke");
    expect(cmd).toContain("adw/core/tests");
  });

  it("rejects advanced option keys by presence", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const result = await execute({ coverage: false });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("does not accept advanced option 'coverage'");
  });

  it("keeps only compact routine fields in the public wrapper schema", async () => {
    await loadToolExecute("../../run_pytest_basic.ts");
    assertCountedAndExemptFields(getCapturedToolDefinition(), {
      counted: ["cwd", "minTests", "testPath", "timeout"],
      exempt: ["options"],
    });
    assertPublicSchemaOmitsKeys(
      getCapturedToolDefinition(),
      COMPACT_SCHEMA_FIELD_FIXTURES.runPytestBasicOmittedKeys,
    );
  });

  it("rejects removed legacy direct fields by presence", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    expect(await execute({ outputMode: "full" })).toContain(
      "does not accept direct field 'outputMode'",
    );
    expect(await execute({ failFast: true })).toContain(
      "does not accept direct field 'failFast'",
    );
    expect(await execute({ testFilter: "agent" })).toContain(
      "does not accept direct field 'testFilter'",
    );
  });

  it("rejects falsey and truthy advanced keys by presence", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    expect(await execute({ coverageThreshold: 0 })).toContain(
      "does not accept advanced option 'coverageThreshold'",
    );
    expect(await execute({ pytestArgs: [] })).toContain(
      "does not accept advanced option 'pytestArgs'",
    );
  });

  it("rejects testPath beginning with dash", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const result = await execute({ testPath: "--maxfail=1" });

    expect(result).toContain("testPath must not start with '-'");
  });

  it("rejects testPath resolving outside repository root", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const result = await execute({ testPath: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain(`testPath resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-positive numeric guards", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    expect(await execute({ minTests: 0 })).toContain("minTests must be a positive finite number");
    expect(await execute({ timeout: -1 })).toContain("timeout must be a positive finite number");
  });

  it("rejects timeout values above 3600 seconds before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    const result = await execute({ timeout: 120000 });

    expect(result).toBe(
      "ERROR: timeout must be a positive finite number in seconds and must not exceed 3600 seconds (1 hour).",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("accepts timeout=3600 at the boundary and forwards it to the helper", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    const result = await execute({ timeout: 3600, testPath: "adw/core/tests" });

    expect(result).toBe("ok");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--timeout=3600");
    expect(cmd).toContain("adw/core/tests");
  });

  it("rejects malformed or unsupported bounded options", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    expect(await execute({ options: "unknown-token" })).toContain("token is not supported");
    expect(await execute({ options: "output=xml" })).toContain("output must be one of summary, full, json");
    expect(await execute({ options: 'test-filter="' })).toContain("unterminated quoted value");
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
    setDollarError(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.failure);

    const result = await execute({});
    expect(result).toBe(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.preferred);
    expect(result).not.toContain(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.shadowed);
  });

  it("returns raw stdout unchanged for JSON failure payloads", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    const jsonStdout = ERROR_PRECEDENCE_FIXTURES.jsonStdout;
    setDollarError(buildDollarFailure({ stdout: jsonStdout, stderr: "ignored" }));

    expect(await execute({})).toBe(jsonStdout);
  });

  it("falls back to stderr then message when stdout absent", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");
    setDollarError(ERROR_PRECEDENCE_FIXTURES.stderrFallback);
    expect(await execute({})).toContain(ERROR_PRECEDENCE_FIXTURES.stderrFallback.stderr);

    setDollarError(ERROR_PRECEDENCE_FIXTURES.messageOnly);
    expect(await execute({})).toContain(ERROR_PRECEDENCE_FIXTURES.messageOnly.message);
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
      const result = await execute({ testPath: "adw/core/tests", options: 'test-filter="agent" fail-fast' });
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

      setDollarError(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.failure);
      const failedResult = await execute({});
      expect(failedResult).toBe(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.preferred);
      expect(failedResult).not.toContain(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.shadowed);
    } finally {
      rmSync(fixtureRoot, { recursive: true, force: true });
    }
  });
});
