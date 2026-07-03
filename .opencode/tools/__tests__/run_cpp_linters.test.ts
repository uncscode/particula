import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { realpathSync } from "node:fs";

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

describe("run_cpp_linters wrapper", () => {
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

  it("preserves explicit autoFix bridge with bounded options", async () => {
    const execute = await loadToolExecute("../../run_cpp_linters.ts");
    const result = await execute({ sourceDir: "./.opencode", autoFix: true, options: "output=json linters=clang-tidy" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--auto-fix");
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--linters=clang-tidy");
    expect(cmd).toContain(`--source-dir=${realpathSync(".opencode")}`);
  });

  it("omits auto-fix when falsey and rejects legacy direct fields or invalid options", async () => {
    const execute = await loadToolExecute("../../run_cpp_linters.ts");
    await execute({ sourceDir: ".opencode", options: "linters=clang-format" });
    expect(getInvocations().at(-1)?.args.join(" ") ?? "").not.toContain("--auto-fix");

    resetSubprocessMocks();
    expect(await execute({ sourceDir: ".opencode", outputMode: "json" })).toContain("does not accept direct field 'outputMode'");
    expect(await execute({ sourceDir: ".opencode", linters: ["clang-tidy"] })).toContain("does not accept direct field 'linters'");
    expect(await execute({ sourceDir: ".opencode", options: "output=xml" })).toContain("output must be one of summary, full, json");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves missing-script hinting on stderr failures", async () => {
    const execute = await loadToolExecute("../../run_cpp_linters.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT missing script" }));
    const result = await execute({ sourceDir: ".opencode" });
    expect(result).toContain("Missing backing script .opencode/tools/run_cpp_linters.py");
  });

  it("prefers stdout over stderr on subprocess failure", async () => {
    const execute = await loadToolExecute("../../run_cpp_linters.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    const result = await execute({ sourceDir: ".opencode" });
    expect(result).toBe("stdout diagnostic");
  });
});
