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

describe("run_cpp_lint_fix wrapper", () => {
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

  it("always appends auto-fix while using bounded options", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_fix.ts");
    const result = await execute({ sourceDir: ".", buildDir: ".", options: "linters=clang-format output=summary" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--auto-fix");
    expect(cmd).toContain("--linters=clang-format");
    expect(cmd).toContain("--output=summary");
    expect(cmd).toContain(`--source-dir=${realpathSync(".")}`);
    expect(cmd).toContain(`--build-dir=${realpathSync(".")}`);
  });

  it("rejects legacy direct fields, invalid option tokens, and out-of-root buildDir before spawn", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_fix.ts");
    expect(await execute({ sourceDir: ".", outputMode: "summary" })).toContain("does not accept direct field 'outputMode'");
    expect(await execute({ sourceDir: ".", linters: ["clang-format"] })).toContain("does not accept direct field 'linters'");
    expect(await execute({ sourceDir: ".", options: "linters=" })).toContain("non-empty '=value'");
    expect(await execute({ sourceDir: ".", buildDir: tmpdir() })).toContain("buildDir path resolves outside repository root");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves missing-script hinting on stderr failures", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_fix.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT missing script" }));
    const result = await execute({ sourceDir: "." });
    expect(result).toContain("Missing backing script .opencode/tools/run_cpp_linters.py");
  });

  it("uses diagnostics fallback when subprocess returns only message", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_fix.ts");
    setDollarError(buildDollarFailure({ message: "mock failure" }));
    const result = await execute({ sourceDir: "." });
    expect(result).toContain("Failed to run C++ lint fix");
    expect(result).toContain("Diagnostics:");
  });

  it("prefers stdout over stderr on subprocess failure", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_fix.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    expect(await execute({ sourceDir: "." })).toBe("stdout diagnostic");
  });
});
