import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("run_linters wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("LINTING SUMMARY\nRESULT: ALL LINTERS PASSED ✓");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds default auto-fix command without target-dir", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({});

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";

    expect(command).toContain("python3");
    expect(command).toContain("run_linters.py");
    expect(command).toContain("--output=summary");
    expect(command).toContain("--ruff-timeout=120");
    expect(command).toContain("--mypy-timeout=180");
    expect(command).toContain("--auto-fix");
    expect(command).toContain("--linters=ruff,mypy");
    expect(command).not.toContain("--target-dir=");
  });

  it("emits no-auto-fix and forwards target-dir when explicitly disabled", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({ autoFix: false, targetDir: "adw/core", outputMode: "full", linters: ["ruff"] });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";

    expect(command).toContain("python3");
    expect(command).toContain("run_linters.py");
    expect(command).toContain("--output=full");
    expect(command).toContain("--ruff-timeout=120");
    expect(command).toContain("--mypy-timeout=180");
    expect(command).toContain("--target-dir=adw/core");
    expect(command).toContain("--linters=ruff");
    expect(command).toContain("--no-auto-fix");
    expect(command).not.toContain("--auto-fix");
  });

  it("forwards custom timeout values and omits the linters flag when none are requested", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({ autoFix: false, linters: [], ruffTimeout: 33, mypyTimeout: 44 });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";

    expect(command).toContain("--ruff-timeout=33");
    expect(command).toContain("--mypy-timeout=44");
    expect(command).not.toContain("--linters=");
    expect(command).toContain("--no-auto-fix");
    expect(command).not.toContain("--auto-fix");
  });

  it("returns stdout from subprocess failures for deterministic diagnostics", async () => {
    setDollarError({ stdout: "lint failed details", stderr: "ignored stderr", message: "spawn failed" });
    const execute = await loadToolExecute("../../run_linters.ts");

    const result = await execute({ autoFix: false });

    expect(String(result)).toBe("lint failed details");
  });

  it("returns stderr when subprocess stdout is unavailable", async () => {
    setDollarError({ stderr: "lint failed stderr", message: "spawn failed" });
    const execute = await loadToolExecute("../../run_linters.ts");

    const result = await execute({ autoFix: false });

    expect(String(result)).toBe("lint failed stderr");
  });

  it("falls back to a deterministic error message when subprocess stdout is unavailable", async () => {
    setDollarError({ message: "spawn failed" });
    const execute = await loadToolExecute("../../run_linters.ts");

    const result = await execute({ autoFix: false });

    expect(String(result)).toBe("ERROR: Failed to run linters: spawn failed");
  });
});
