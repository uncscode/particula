import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { tmpdir } from "node:os";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
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

describe("run_pytest_advanced wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds advanced command from bounded options while preserving direct payload fields", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({
      options: 'output=json fail-fast test-filter="agent smoke" cov-report=term-missing,html:coverage_html durations=0 durations-min=0.25',
      testPath: "adw/core/tests/agent_test.py",
      coverage: false,
      overrideIni: ["addopts="],
      pytestArgs: ["-k", "agent"],
    });

    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--fail-fast");
    expect(cmd).toContain("-k");
    expect(cmd).toContain("agent smoke");
    expect(cmd).toContain("adw/core/tests/agent_test.py");
    expect(cmd).toContain("--no-coverage");
    expect(cmd).toContain("--durations=0");
    expect(cmd).toContain("--durations-min=0.25");
    expect(cmd).toContain("--override-ini=addopts=");
  });

  it("omits legacy direct compatibility fields from the wrapper schema", async () => {
    await loadToolExecute("../../run_pytest_advanced.ts");
    assertCountedAndExemptFields(getCapturedToolDefinition(), {
      counted: [
        "coverage",
        "coverageSource",
        "coverageThreshold",
        "cwd",
        "minTests",
        "overrideIni",
        "pytestArgs",
        "testPath",
        "timeout",
      ],
      exempt: ["options"],
    });
    assertPublicSchemaOmitsKeys(
      getCapturedToolDefinition(),
      COMPACT_SCHEMA_FIELD_FIXTURES.runPytestAdvancedOmittedKeys,
    );
  });

  it("rejects removed legacy direct fields by presence", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ outputMode: "json" })).toContain(
      "does not accept direct field 'outputMode'",
    );
    expect(await execute({ failFast: true })).toContain(
      "does not accept direct field 'failFast'",
    );
    expect(await execute({ testFilter: "agent" })).toContain(
      "does not accept direct field 'testFilter'",
    );
    expect(await execute({ covReport: ["term-missing"] })).toContain(
      "does not accept direct field 'covReport'",
    );
    expect(await execute({ durations: 0 })).toContain(
      "does not accept direct field 'durations'",
    );
    expect(await execute({ durationsMin: 0.5 })).toContain(
      "does not accept direct field 'durationsMin'",
    );
  });

  it("preserves durationsMin omission when durations is absent", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    await execute({ options: "durations-min=0.5" });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).not.toContain("--durations-min=0.5");
  });

  it("rejects malformed bounded options and invalid duration values", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ options: "unknown-token" })).toContain("token is not supported");
    expect(await execute({ options: "durations=abc" })).toContain("durations must be a finite number");
    expect(await execute({ options: "durations=-1" })).toContain("durations must be a non-negative finite number");
    expect(await execute({ coverageThreshold: -1 })).toContain("coverageThreshold must be a non-negative finite number");
    expect(await execute({ testPath: "--maxfail=1" })).toContain("testPath must not start with '-'");
  });

  it("rejects timeout values above 3600 seconds before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ timeout: 120000 });

    expect(result).toBe(
      "ERROR: timeout must be a positive finite number in seconds and must not exceed 3600 seconds (1 hour).",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("accepts timeout=3600 at the boundary and forwards it to the helper", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ timeout: 3600, testPath: "tests/run_pytest_default_test.py" });

    expect(result).toBe("ok");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--timeout=3600");
    expect(cmd).toContain("tests/run_pytest_default_test.py");
  });

  it("rejects testPath resolving outside repository root", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const result = await execute({ testPath: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain(`testPath resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves stdout/stderr/message failure precedence", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    setDollarError(buildDollarFailure({ stdout: '{"success":false,"error":"details"}', stderr: "stderr shadow" }));
    expect(await execute({ options: "output=json" })).toBe('{"success":false,"error":"details"}');

    setDollarError(buildDollarFailure({ stdout: "", stderr: "stderr diagnostic" }));
    expect(await execute({})).toContain("stderr diagnostic");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "", message: "fallback" }));
    assertErrorPrefix(String(await execute({})), "ERROR:");
  });

  it("returns raw stdout unchanged for JSON failure payloads", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const jsonStdout = '{"success":false,"error":"details"}';
    setDollarError(buildDollarFailure({ stdout: jsonStdout, stderr: "ignored" }));

    expect(await execute({ options: "output=json" })).toBe(jsonStdout);
  });

  it("returns raw stdout unchanged for alternate JSON failure payloads", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const jsonStdout = '{"ok":false,"error":"details"}';
    setDollarError(buildDollarFailure({ stdout: jsonStdout, stderr: "ignored" }));

    expect(await execute({ options: "output=json" })).toBe(jsonStdout);
  });

  it("returns an error envelope when thrown stdout is non-json or claims success", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    setDollarError(buildDollarFailure({ stdout: "plain stdout diagnostic", stderr: "ignored" }));
    expect(String(await execute({ options: "output=json" }))).toContain(
      "stdout did not report failure semantics",
    );

    setDollarError(buildDollarFailure({ stdout: '{"success":true}', stderr: "ignored" }));
    expect(String(await execute({ options: "output=json" }))).toContain(
      "stdout did not report failure semantics",
    );
  });

  it("returns deterministic validation failure stdout unchanged", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const validationStdout = [
      "============================================================",
      "VALIDATION: FAILED",
      "============================================================",
      "  - Coverage 77% is below threshold of 80%",
    ].join("\n");
    setDollarError(buildDollarFailure({ stdout: validationStdout, stderr: "ignored" }));

    expect(await execute({ coverage: true })).toBe(validationStdout);
  });

  it("rejects malformed coverageSource entries before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ coverageSource: "adw,,adw.utils" })).toContain(
      "empty comma-separated entries",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects absolute coverageSource paths before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ coverageSource: `${tmpdir()}/x.py` })).toContain(
      "must not contain absolute paths",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects coverageSource traversal before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ coverageSource: "../outside" })).toContain(
      "coverageSource must stay within the repository/worktree root",
    );
    expect(await execute({ coverageSource: "adw/../../outside" })).toContain(
      "coverageSource must stay within the repository/worktree root",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("passes through repo-relative file coverageSource requests", async () => {
    setDollarText(buildSuccessOutput('{"metrics":{"coverage_files":null},"success":true}'));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({
      options: "output=json",
      coverageSource: "adw/core/tests/agent_test.py",
    });

    expect(result).toContain('"coverage_files":null');
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--coverage-source=adw/core/tests/agent_test.py");
  });

  it("treats root-level .py coverageSource entries as file targets", async () => {
    setDollarText(buildSuccessOutput('{"metrics":{"coverage_files":null},"success":true}'));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({
      options: "output=json",
      coverageSource: "conftest.py",
    });

    expect(result).toContain('"coverage_files":null');
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--coverage-source=conftest.py");
  });

  it("treats coverageSource=all as default coverage without explicit sources", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ coverageSource: "all" });

    expect(result).toBe("ok");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--coverage");
    expect(cmd).not.toContain("--coverage-source=");
  });

  it("rejects raw coverage pytestArgs when coverage is disabled", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ coverage: false, pytestArgs: ["--cov=adw"] });

    expect(String(result)).toContain(
      "coverage-related pytest arguments are not allowed when coverage is disabled",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("passes through unusable coverage failure output from the helper", async () => {
    setDollarText([
      "============================================================",
      "VALIDATION: FAILED",
      "============================================================",
      "  - Coverage data is unusable: pytest-cov reported 'no data collected'. Review coverageSource/import targeting.",
    ].join("\n"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ testPath: "tests/run_pytest_default_test.py" });

    expect(result).toContain("Coverage data is unusable");
    expect(result).toContain("no data collected");
  });

  it("passes through same-worktree coverage lock failures from the helper", async () => {
    setDollarText(
      "ERROR: coverage-enabled pytest runs in the same worktree must be serialized; another coverage run is already active",
    );
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ testPath: "tests/run_pytest_default_test.py" });

    expect(result).toContain("must be serialized");
    expect(result).toContain("already active");
  });
});
