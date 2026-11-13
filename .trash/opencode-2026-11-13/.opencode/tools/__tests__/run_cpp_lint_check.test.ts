import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { realpathSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
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

describe("run_cpp_lint_check wrapper", () => {
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

  it("builds bounded-options command without auto-fix", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_check.ts");
    const result = await execute({ sourceDir: ".", buildDir: ".", options: "output=full linters=clang-tidy" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("run_cpp_linters.py");
    expect(cmd).toContain(`--source-dir=${realpathSync(".")}`);
    expect(cmd).toContain(`--build-dir=${realpathSync(".")}`);
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--linters=clang-tidy");
    expect(cmd).not.toContain("--auto-fix");
  });

  it("rejects legacy direct fields, malformed tokens, or duplicate options before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_check.ts");
    expect(await execute({ sourceDir: ".", outputMode: "full" })).toContain("does not accept direct field 'outputMode'");
    expect(await execute({ sourceDir: ".", linters: ["clang-tidy"] })).toContain("does not accept direct field 'linters'");
    expect(await execute({ sourceDir: ".", options: "unknown=1" })).toContain("not supported");
    expect(await execute({ sourceDir: ".", options: "output=json output=full" })).toContain("duplicate token");
    expect(await execute({ sourceDir: ".", options: 'linters="unterminated' })).toContain("unterminated quoted value");
    expect(getInvocations()).toHaveLength(0);
  });

  it("fails closed for path validation and unsupported linter values", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_check.ts");
    expect(await execute({ sourceDir: join(tmpdir(), `missing-${Date.now()}`) })).toContain("sourceDir path does not exist");
    expect(await execute({ sourceDir: import.meta.path })).toContain("sourceDir path is not a directory");
    expect(await execute({ sourceDir: ".", options: "linters=bad-linter" })).toContain("unsupported linter");
    expect(await execute({ sourceDir: tmpdir() })).toContain("sourceDir path resolves outside repository root");
  });

  it("preserves failure envelope and missing-script hint", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_check.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT missing script" }));
    const result = await execute({ sourceDir: "." });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("Missing backing script .opencode/tools/run_cpp_linters.py");
  });

  it("prefers stdout over stderr on subprocess failure", async () => {
    const execute = await loadToolExecute("../../run_cpp_lint_check.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    expect(await execute({ sourceDir: "." })).toBe("stdout diagnostic");
  });
});
