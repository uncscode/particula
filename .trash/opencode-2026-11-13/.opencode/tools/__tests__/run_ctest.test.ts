import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("run_ctest wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds default command when options are omitted", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");

    const result = await execute({ buildDir: "build" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--build-dir=build");
    expect(cmd).toContain("--output=summary");
    expect(cmd).toContain("--min-tests=1");
    expect(cmd).toContain("--timeout=300");
    expect(cmd).not.toContain(" -R ");
    expect(cmd).not.toContain(" -E ");
    expect(cmd).not.toContain(" -j ");
  });

  it("forwards bounded filter and parallel options", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");

    const result = await execute({
      buildDir: "build",
      minTests: 4,
      timeout: 90,
      options: "output=full test-filter=math exclude-filter=slow parallel=3",
    });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("-R math");
    expect(cmd).toContain("-E slow");
    expect(cmd).toContain("-j 3");
  });

  it("supports quoted filters containing spaces", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");

    const result = await execute({
      buildDir: "build",
      options: 'test-filter="math suite" exclude-filter="slow suite"',
    });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("-R math suite");
    expect(cmd).toContain("-E slow suite");
  });

  it("rejects unsupported or duplicate options before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");

    expect(await execute({ buildDir: "build", options: "unsupported=1" })).toContain("not supported");
    expect(await execute({ buildDir: "build", options: "parallel=2 parallel=4" })).toContain("duplicate token");
    expect(await execute({ buildDir: "build", options: 'test-filter="unterminated' })).toContain(
      "unterminated quoted value",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing buildDir and invalid numeric guards before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");

    expect(await execute({})).toContain("buildDir is required");
    expect(await execute({ buildDir: "build", timeout: Number.NaN })).toContain("Timeout must be positive");
    expect(await execute({ buildDir: "build", minTests: 0 })).toContain("minTests must be positive");
    expect(await execute({ buildDir: "build", options: "parallel=0" })).toContain("parallel must be positive");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects out-of-root buildDir before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");

    expect(await execute({ buildDir: "/tmp" })).toContain("buildDir path resolves outside repository root");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves deterministic failure precedence and ENOENT hinting", async () => {
    const execute = await loadToolExecute("../../run_ctest.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing script" }));

    const result = await execute({ buildDir: "build" });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "Missing backing script .opencode/tools/run_ctest.py");
  });
});
