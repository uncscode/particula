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

    await execute({ confirmed: true });

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

  it("forwards explicit Ruff mode and ordered targets without auto-fix", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({ mode: "format-check", targetPaths: ["adforge_core", ".opencode/tools"] });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(command).toContain("--mode=format-check");
    expect(command).toContain('--target-paths-json=["adforge_core",".opencode/tools"]');
    expect(command).toContain("--linters=ruff");
    expect(command).not.toContain("--auto-fix");
    expect(command).not.toContain("--no-auto-fix");
  });

  it("forwards a worktree cwd with explicit Ruff targets", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");
    const cwd = process.cwd();

    await execute({ cwd, mode: "format-check", targetPaths: ["adforge_core", ".opencode/tools"] });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(command).toContain(`--cwd=${cwd}`);
    expect(command).toContain('--target-paths-json=["adforge_core",".opencode/tools"]');
  });

  it("keeps every explicit mode Ruff-only and never transports legacy mutation selectors", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({ mode: "check" });
    await execute({ mode: "format-check" });
    await execute({ mode: "format", confirmed: true });

    const commands = getInvocations().map((invocation) => invocation.args.join(" "));
    expect(commands).toHaveLength(3);
    expect(commands[0]).toContain("--mode=check");
    expect(commands[1]).toContain("--mode=format-check");
    expect(commands[2]).toContain("--mode=format");
    for (const command of commands) {
      expect(command).toContain("--linters=ruff");
      expect(command).not.toContain("--auto-fix");
      expect(command).not.toContain("--no-auto-fix");
      expect(command).not.toContain("--target-dir=");
    }
  });

  it("rejects explicit-mode conflicts and unsafe target paths before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    expect(await execute({ mode: "check", autoFix: true })).toContain("mode conflicts with autoFix");
    expect(await execute({ targetPaths: ["adforge_core"] })).toContain("targetPaths requires mode");
    expect(await execute({ mode: "check", targetPaths: ["../outside"] })).toContain("invalid repository-relative path");
    expect(await execute({ cwd: "..", mode: "check" })).toContain("cwd path resolves outside repository root");
    expect(getInvocations()).toHaveLength(0);
  });

  it("requires confirmation for mutating format and legacy auto-fix requests", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    expect(await execute({ mode: "format" })).toContain("explicit confirmation");
    expect(await execute({ autoFix: true })).toContain("explicit confirmation");
    expect(getInvocations()).toHaveLength(0);

    await execute({ mode: "format", confirmed: true });
    await execute({ autoFix: true, confirmed: true });
    expect(getInvocations()).toHaveLength(2);
  });

  it("accepts an explicit mode with a materialized false auto-fix default", async () => {
    const execute = await loadToolExecute("../../run_linters.ts");

    await execute({ mode: "check", autoFix: false });

    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0].args.join(" ")).toContain("--mode=check");
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
