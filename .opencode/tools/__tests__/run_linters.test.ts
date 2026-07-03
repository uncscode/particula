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

    await execute({ autoFix: false, targetDir: "adw/core", options: "output=full linters=ruff" });

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

  it("forwards custom timeout values and selected linters from bounded options", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({ autoFix: false, ruffTimeout: 33, mypyTimeout: 44, options: "linters=mypy" });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";

    expect(command).toContain("--ruff-timeout=33");
    expect(command).toContain("--mypy-timeout=44");
    expect(command).toContain("--linters=mypy");
    expect(command).toContain("--no-auto-fix");
    expect(command).not.toContain("--auto-fix");
  });

  it("rejects unsupported, duplicate, and malformed bounded option tokens before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    expect(await execute({ options: "unsupported=1" })).toContain("not supported");
    expect(await execute({ options: "output=full output=json" })).toContain("duplicate token");
    expect(await execute({ options: "linters=eslint" })).toContain("unsupported linter");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects invalid timeout guards before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    expect(await execute({ ruffTimeout: 0 })).toContain("ruffTimeout must be positive");
    expect(await execute({ mypyTimeout: Number.NaN })).toContain("mypyTimeout must be positive");
    expect(getInvocations()).toHaveLength(0);
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
