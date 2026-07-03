import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { realpathSync } from "node:fs";
import { tmpdir } from "node:os";

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

describe("run_cpp_coverage_summary wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText(buildSuccessOutput("ok"));
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds routine coverage command from bounded options", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_summary.ts");
    const result = await execute({ buildDir: ".opencode", threshold: 80, timeout: 60, options: "output=full" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("run_cpp_coverage.py");
    expect(cmd).toContain(`--build-dir=${realpathSync(".opencode")}`);
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--threshold 80");
    expect(cmd).toContain("--timeout 60");
  });

  it("rejects legacy direct fields and advanced keys by presence including falsey payloads", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_summary.ts");
    expect(await execute({ buildDir: ".opencode", outputMode: "summary" })).toContain("does not accept direct field 'outputMode'");
    expect(await execute({ buildDir: ".opencode", tool: "" })).toContain("does not accept advanced option 'tool'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("fails closed for invalid options, threshold validation, and path validation", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_summary.ts");
    expect(await execute({ buildDir: ".opencode", options: "output=xml" })).toContain("output must be one of summary, full, json");
    expect(await execute({ buildDir: ".opencode", threshold: 101 })).toContain("threshold must be between 0 and 100");
    expect(await execute({ buildDir: tmpdir() })).toContain("buildDir path resolves outside repository root");
  });

  it("preserves missing-script hinting", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_summary.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT missing script" }));
    const result = await execute({ buildDir: ".opencode" });
    expect(result).toContain("Missing backing script .opencode/tools/run_cpp_coverage.py");
  });

  it("prefers stdout over stderr on subprocess failure", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_summary.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    expect(await execute({ buildDir: ".opencode" })).toBe("stdout diagnostic");
  });
});
