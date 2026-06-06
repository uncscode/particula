import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { tmpdir } from "node:os";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
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
      testPath: "tools/__tests__/adw_spec.test.ts",
      testFilter: "raw stdout",
      timeout: 120,
      minTests: 1,
      failFast: true,
      cwd: process.cwd(),
    });

    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("run_bun_test.py");
    expect(cmd).toContain("--test-path=tools/__tests__/adw_spec.test.ts");
    expect(cmd).toContain("--filter=raw stdout");
    expect(cmd).toContain("--timeout=120");
    expect(cmd).toContain("--min-tests=1");
    expect(cmd).toContain("--bail");
    expect(cmd).toContain(`--cwd=${process.cwd()}`);
  });

  it("rejects blank testPath when provided", async () => {
    const execute = await loadToolExecute("../../run_bun_test.ts");
    const result = await execute({ testPath: "   " });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("testPath must not be blank");
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

    const result = await execute({ testPath: "tools/__tests__/adw_spec.test.ts" });
    assertContains(String(result), "stdout diagnostic");
    expect(result).not.toContain("stderr shadow");
  });
});
