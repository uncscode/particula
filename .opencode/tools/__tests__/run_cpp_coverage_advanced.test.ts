import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, realpathSync } from "node:fs";
import path from "node:path";
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

describe("run_cpp_coverage_advanced wrapper", () => {
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

  it("builds advanced command with bounded tool and direct filter/html", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_advanced.ts");
    const result = await execute({ buildDir: "./.opencode", filter: "src/", html: "./.opencode", options: "output=json tool=llvm-cov" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--tool=llvm-cov");
    expect(cmd).toContain("--filter=src/");
    expect(cmd).toContain(`--build-dir=${realpathSync(".opencode")}`);
    expect(cmd).toContain(`--html=${realpathSync(".opencode")}`);
  });

  it("omits blank optional strings and rejects legacy direct fields or invalid option tokens", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_advanced.ts");
    await execute({ buildDir: ".opencode", filter: " ", html: " ", options: "output=summary" });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).not.toContain("--filter=");
    expect(cmd).not.toContain("--html=");

    resetSubprocessMocks();
    expect(await execute({ buildDir: ".opencode", outputMode: "summary" })).toContain("does not accept direct field 'outputMode'");
    expect(await execute({ buildDir: ".opencode", tool: "gcov" })).toContain("does not accept direct field 'tool'");
    expect(await execute({ buildDir: ".opencode", extraArgs: ["--flag"] })).toContain("does not accept direct field 'extraArgs'");
    expect(await execute({ buildDir: ".opencode", options: "tool=bad" })).toContain("tool must be one of gcov, llvm-cov");
    expect(getInvocations()).toHaveLength(0);
  });

  it("fails closed for html, threshold, and buildDir path validation", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_advanced.ts");
    expect(await execute({ buildDir: ".opencode", html: import.meta.path })).toContain("html path is not a directory");
    expect(await execute({ buildDir: ".opencode", threshold: -1 })).toContain("threshold must be between 0 and 100");
    expect(await execute({ buildDir: tmpdir() })).toContain("buildDir path resolves outside repository root");
  });

  it("canonicalizes and allows safe nested new html output directories", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_advanced.ts");
    const parentDir = path.join(".opencode", "tmp_cpp_wrapper_html_parent");
    mkdirSync(parentDir, { recursive: true });
    const nestedHtmlDir = path.join(parentDir, "nested", "coverage_html");

    const result = await execute({ buildDir: ".opencode", html: nestedHtmlDir, options: "output=summary" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain(`--html=${path.join(realpathSync(parentDir), "nested", "coverage_html")}`);
  });

  it("preserves failure diagnostics and missing-script hinting", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_advanced.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT missing script" }));
    const result = await execute({ buildDir: ".opencode" });
    expect(result).toContain("Missing backing script .opencode/tools/run_cpp_coverage.py");
  });

  it("prefers stdout over stderr on subprocess failure", async () => {
    const execute = await loadToolExecute("../../run_cpp_coverage_advanced.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    expect(await execute({ buildDir: ".opencode" })).toBe("stdout diagnostic");
  });
});
