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
      options: 'output=json fail-fast test-filter="agent smoke" durations=0 durations-min=0.25',
      testPath: "adw/core/tests/agent_test.py",
      coverage: false,
      pytestArgs: ["-k", "agent"],
    });

    expect(result).toBe("ok");

    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      expect.stringContaining("/run_pytest.py"),
      "--output=json",
      "--min-tests=1",
      "--timeout=600",
      "--fail-fast",
      "--test-filter=agent smoke",
      "--test-path=adw/core/tests/agent_test.py",
      "--no-coverage",
      "--durations=0",
      "--durations-min=0.25",
      '--pytest-argv-json=["-k","agent"]',
    ]);
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
        "pytestArgs",
        "testPath",
        "testPaths",
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
    expect(await execute({ overrideIni: ["addopts="] })).toBe(
      "ERROR: run_pytest_advanced does not accept direct field 'overrideIni'. Use coverage: false for scoped assertion runs; caller ini overrides are prohibited.",
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

  it("rejects timeout values above 1200 seconds before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ timeout: 120000 });

    expect(result).toBe(
      "ERROR: timeout must be a positive finite number in seconds and must not exceed 1200 seconds (20 minutes).",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("accepts timeout=1200 at the boundary and forwards it to the helper", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ timeout: 1200, testPath: "tests/run_pytest_default_test.py" });

    expect(result).toBe("ok");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--timeout=1200");
    expect(cmd).toContain("tests/run_pytest_default_test.py");
  });

  it("rejects testPath resolving outside repository root", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const result = await execute({ testPath: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain(`testPath resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("transports ordered plural targets once as compact JSON", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ coverage: false, testPaths: ["adw/core/tests/agent_test.py", "adw/utils/tests"] })).toBe("ok");
    expect(getInvocations().at(-1)?.args).toContain(
      '--test-paths-json=["adw/core/tests/agent_test.py","adw/utils/tests"]',
    );
  });

  it("preserves pytest node-id suffixes in plural target transport", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(
      await execute({
        coverage: false,
        testPaths: ["adw/core/tests/agent_test.py::test_agent_resolution"],
      }),
    ).toBe("ok");
    expect(getInvocations().at(-1)?.args).toContain(
      '--test-paths-json=["adw/core/tests/agent_test.py::test_agent_resolution"]',
    );
  });

  it("preserves seven plural target order before the caller pytest suffix", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const testPaths = Array.from({ length: 7 }, (_, index) => `adw/tests/case_${index}_test.py`);

    expect(await execute({ coverage: false, testPaths, pytestArgs: ["-q"] })).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual(expect.arrayContaining([
      `--test-paths-json=${JSON.stringify(testPaths)}`,
      '--pytest-argv-json=["-q"]',
    ]));
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.indexOf(`--test-paths-json=${JSON.stringify(testPaths)}`)).toBeLessThan(
      args.indexOf('--pytest-argv-json=["-q"]'),
    );
  });

  it("rejects ambiguous or unsafe plural targets before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ testPath: "tests/x.py", testPaths: [] })).toContain("cannot be combined");
    expect(await execute({ testPaths: [] })).toContain("non-empty array");
    expect(await execute({ testPaths: ["../outside"] })).toContain("canonical relative POSIX");
    expect(await execute({ testPaths: Array.from({ length: 8 }, () => "adw/tests/x_test.py") })).toContain("at most 7");
    expect(await execute({ testPaths: ["adw\\tests\\x_test.py"] })).toContain("canonical relative POSIX");
    expect(getInvocations()).toHaveLength(0);
  });

  it("requires explicit disabled coverage for collect-only before spawning", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ pytestArgs: ["--collect-only"] })).toBe(
      "ERROR: --collect-only requires coverage: false.",
    );
    expect(getInvocations()).toHaveLength(0);

    setDollarText(buildSuccessOutput('{"success":true,"collection":{"collected_count":2,"executed_count":0}}'));
    expect(await execute({ coverage: false, pytestArgs: ["--collect-only"] })).toContain("collected_count");
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

  it("preserves valid runner JSON including its producer identity byte-for-byte", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");
    const jsonStdout = '{"success":false,"evidence_identity":{"contract":"e37-m2-validation-git","version":1},"error":"details"}';
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
    const malformed = JSON.parse(String(await execute({ options: "output=json" })));
    expect(malformed.outcome.classification).toBe("runner");
    expect(JSON.stringify(malformed)).not.toContain("plain stdout diagnostic");
    expect(malformed.evidence_identity).toBeUndefined();

    setDollarError(buildDollarFailure({ stdout: '{"success":true}', stderr: "ignored" }));
    const invalidSuccess = JSON.parse(String(await execute({ options: "output=json" })));
    expect(invalidSuccess.outcome.reason).toContain(
      "invalid failure envelope",
    );
    expect(invalidSuccess.evidence_identity).toBeUndefined();
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

    expect(await execute({ coverageSource: `${tmpdir()}/x.py` })).toContain("relative POSIX path");
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

  it("ignores repo-relative file coverageSource requests and uses repository coverage", async () => {
    setDollarText(buildSuccessOutput('{"metrics":{"coverage_files":null},"success":true}'));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({
      options: "output=json",
      coverageSource: "adw/core/tests/agent_test.py",
    });

    expect(result).toContain("INFO: coverageSource supports only 'all'");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--coverage");
    expect(cmd).not.toContain("--coverage-source=");
  });

  it("ignores root-level .py coverageSource entries", async () => {
    setDollarText(buildSuccessOutput('{"metrics":{"coverage_files":null},"success":true}'));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({
      options: "output=json",
      coverageSource: "conftest.py",
    });

    expect(result).toContain("INFO: coverageSource supports only 'all'");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).not.toContain("--coverage-source=");
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

  it("rejects coverage controls while disabled before invoking the helper", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    for (const args of [
      { coverage: false, coverageSource: "adw" },
      { coverage: false, coverageThreshold: 90 },
      { coverage: false, options: "cov-report=term" },
      { coverage: false, coverageSource: "module.txt" },
    ]) {
      expect(await execute(args)).toBe(
        "ERROR: coverage-specific controls are not allowed when coverage is disabled.",
      );
    }
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects combined coverage controls while disabled without spawning", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({
      coverage: false,
      coverageSource: "adw",
      coverageThreshold: 90,
      options: "cov-report=term-missing",
    })).toBe("ERROR: coverage-specific controls are not allowed when coverage is disabled.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves enabled coverage source order and accepts case-insensitive all", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ coverageSource: "adforge_core,adw" })).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual(expect.arrayContaining([
      "--coverage-source=adforge_core",
      "--coverage-source=adw",
    ]));

    resetSubprocessMocks();
    setDollarText(buildSuccessOutput("ok"));
    expect(await execute({ coverageSource: "ALL" })).toBe("ok");
    expect(getInvocations().at(-1)?.args).toContain("--coverage");
    expect(getInvocations().at(-1)?.args.join(" ")).not.toContain("--coverage-source=");
  });

  it("rejects mixed all and unsafe coverage sources before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(await execute({ coverageSource: "all,adw" })).toContain("must be the sole source");
    expect(await execute({ coverageSource: "pkg\\module" })).toContain("relative POSIX path");
    expect(getInvocations()).toHaveLength(0);
  });

  it("ignores dotted and missing coverage sources with informational fallback", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    for (const coverageSource of ["adw.core", "bad-name", "module.txt"]) {
      const result = await execute({ coverageSource });
      expect(result).toContain("INFO: coverageSource supports only 'all'");
      expect(getInvocations().at(-1)?.args.join(" ")).not.toContain("--coverage-source=");
    }
  });

  it("rejects raw coverage pytestArgs when coverage is disabled", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ coverage: false, pytestArgs: ["--cov=adw"] });

    expect(String(result)).toContain("pytestArgs token '--cov=adw' is not permitted.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("directs raw --no-cov callers to the dedicated coverage field", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    const result = await execute({ pytestArgs: ["--no-cov"] });

    expect(String(result)).toContain("set the wrapper field coverage: false instead");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects caller plugin and ini controls before spawn", async () => {
    const execute = await loadToolExecute("../../run_pytest_advanced.ts");

    expect(String(await execute({ pytestArgs: ["-p", "unsafe_plugin"] }))).toContain("not permitted");
    expect(String(await execute({ pytestArgs: ["--override-ini=pythonpath=/outside"] }))).toContain("not permitted");
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
