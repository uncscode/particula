import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { tmpdir } from "node:os";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { COMPACT_SCHEMA_FIELD_FIXTURES } from "./fixtures/wrapper_contract_fixtures";
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
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

describe("run_bun_test wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds bun test command for success path", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_bun_test.ts");

    const result = await execute({
      testPath: "tools/__tests__/adw_spec_read.test.ts",
      timeout: 120,
      minTests: 1,
      cwd: process.cwd(),
      options: 'output=full test-filter="raw stdout" fail-fast',
    });

    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("run_bun_test.py");
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--test-path=tools/__tests__/adw_spec_read.test.ts");
    expect(cmd).toContain("--filter=raw stdout");
    expect(cmd).toContain("--timeout=120");
    expect(cmd).toContain("--min-tests=1");
    expect(cmd).toContain("--bail");
    expect(cmd).toContain(`--cwd=${process.cwd()}`);
  });

  it("keeps only compact routine fields in the public wrapper schema", async () => {
    await loadToolExecute("../../run_bun_test.ts");
    assertCountedAndExemptFields(getCapturedToolDefinition(), {
      counted: ["cwd", "minTests", "testPath", "timeout"],
      exempt: ["options"],
    });
    assertPublicSchemaOmitsKeys(
      getCapturedToolDefinition(),
      COMPACT_SCHEMA_FIELD_FIXTURES.runBunTestOmittedKeys,
    );
  });

  it("rejects removed legacy direct fields by presence", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");

    expect(await execute({ outputMode: "json" })).toContain(
      "does not accept direct field 'outputMode'",
    );
    expect(await execute({ failFast: true })).toContain(
      "does not accept direct field 'failFast'",
    );
    expect(await execute({ testFilter: "datetime" })).toContain(
      "does not accept direct field 'testFilter'",
    );
  });

  it("rejects blank testPath when provided", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    const result = await execute({ testPath: "   " });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("testPath must not be blank");
  });

  it("allows omitted testPath while still executing bun test", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_bun_test.ts");

    const result = await execute({ options: "output=json" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=json");
    expect(cmd).not.toContain("--test-path=");
  });

  it("rejects testPath beginning with dash", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    const result = await execute({ testPath: "--watch" });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("testPath must not start with '-'");
  });

  it("rejects testPath resolving outside repository root", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    const result = await execute({ testPath: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain(`testPath resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects blank or unsupported bounded option values", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");

    expect(await execute({ options: 'test-filter=""' })).toContain("non-empty '=value' suffix");
    expect(await execute({ options: "unknown-token" })).toContain("token is not supported");
  });

  it("rejects invalid numeric guards and blank cwd", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");

    expect(await execute({ timeout: 0 })).toContain("timeout must be a positive finite number");
    expect(await execute({ timeout: Number.NaN })).toContain("timeout must be a positive finite number");
    expect(await execute({ minTests: -1 })).toContain("minTests must be a positive finite number");
    expect(await execute({ minTests: Number.POSITIVE_INFINITY })).toContain("minTests must be a positive finite number");
    expect(await execute({ cwd: "   " })).toContain("cwd must not be blank");
  });

  it("rejects nonexistent cwd path", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    const missingCwd = `${tmpdir()}/run-bun-test-missing-${Date.now()}`;

    expect(await execute({ cwd: missingCwd })).toContain(`cwd path does not exist: ${missingCwd}`);
  });

  it("rejects cwd resolving outside repository root", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    const result = await execute({ cwd: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain(`cwd path resolves outside repository root: ${tmpdir()}`);
  });

  it("prefers stdout over stderr for failure diagnostics", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));

    const result = await execute({ testPath: "tools/__tests__/adw_spec_read.test.ts" });
    assertContains(String(result), "stdout diagnostic");
    expect(result).not.toContain("stderr shadow");
  });

  it("preserves ENOENT missing-script hints when stdout is absent", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing script" }));

    const result = await execute({ testPath: "tools/__tests__/adw_spec_read.test.ts" });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("Missing backing script .opencode/tools/run_bun_test.py.");
  });
});
