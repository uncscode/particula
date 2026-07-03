import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
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
import {
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

function makeBuildFixture(): { buildDir: string; executable: string; cleanup: () => void } {
  const repoRoot = path.resolve(import.meta.dir, "../../..");
  const buildDir = mkdtempSync(path.join(repoRoot, "sanitizers-build-"));
  const executable = path.join(buildDir, "demo_binary");
  writeFileSync(executable, "#!/bin/sh\nexit 0\n", "utf8");
  return {
    buildDir,
    executable,
    cleanup: () => rmSync(buildDir, { recursive: true, force: true }),
  };
}

describe("run_sanitizers_basic wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds routine command for valid payload", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "demo_binary",
        sanitizer: "asan",
        outputMode: "full",
        timeout: 45,
      });

      expect(result).toBe("ok");
      const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
      expect(cmd).toContain("python3");
      expect(cmd).toContain("run_sanitizers.py");
      expect(cmd).toContain(`--build-dir=${fixture.buildDir}`);
      expect(cmd).toContain(`--executable=${fixture.executable}`);
      expect(cmd).toContain("--sanitizer=asan");
      expect(cmd).toContain("--output-mode=full");
      expect(cmd).toContain("--timeout=45");
      expect(cmd).not.toContain("--options=");
      expect(cmd).not.toContain("--normal-duration=");
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects options key on basic wrapper even when empty string", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");

    expect(await execute({ options: "" })).toContain(
      "does not accept advanced option 'options'",
    );
  });

  it("rejects extraArgs key on basic wrapper even when empty array", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");

    expect(await execute({ extraArgs: [] })).toContain(
      "does not accept advanced option 'extraArgs'",
    );
  });

  it("does not advertise advanced-only keys in the basic schema", async () => {
    await loadToolExecute("../../run_sanitizers_basic.ts");
    const args = getCapturedToolDefinition()?.args ?? {};

    expect(args).not.toHaveProperty("suppressions");
    expect(args).not.toHaveProperty("options");
    expect(args).not.toHaveProperty("normalDuration");
    expect(args).not.toHaveProperty("extraArgs");
  });

  it("rejects suppressions key on basic wrapper even when non-empty", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");

    expect(await execute({ suppressions: "tsan.suppr" })).toContain(
      "does not accept advanced option 'suppressions'",
    );
  });

  it("rejects invalid sanitizer allowlist values", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      expect(
        await execute({ buildDir: fixture.buildDir, executable: "demo_binary", sanitizer: "msan" }),
      ).toContain("sanitizer must be one of asan, tsan, ubsan");
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects non-positive timeout before spawn", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "demo_binary",
        sanitizer: "asan",
        timeout: 0,
      });

      expect(result).toContain("Timeout must be a positive integer");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects float timeout before spawn", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "demo_binary",
        sanitizer: "asan",
        timeout: 1.5,
      });

      expect(result).toContain("Timeout must be a positive integer");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects buildDir outside repository root", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const outsideBuildDir = mkdtempSync(path.join(tmpdir(), "sanitizers-outside-"));

    try {
      const result = await execute({
        buildDir: outsideBuildDir,
        executable: "demo_binary",
        sanitizer: "asan",
      });

      assertErrorPrefix(String(result), "ERROR:");
      expect(result).toContain(`buildDir path resolves outside repository root: ${outsideBuildDir}`);
      expect(getInvocations()).toHaveLength(0);
    } finally {
      rmSync(outsideBuildDir, { recursive: true, force: true });
    }
  });

  it("rejects executable outside buildDir", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();
    const repoRoot = path.resolve(import.meta.dir, "../../..");
    const outsideExecutable = path.join(repoRoot, "README.md");

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: outsideExecutable,
        sanitizer: "asan",
      });

      expect(result).toContain(`executable path resolves outside buildDir: ${outsideExecutable}`);
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("wraps stdout-present failures in a deterministic error envelope", async () => {
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "demo_binary",
        sanitizer: "asan",
      });

      expect(result).toBe("ERROR: Sanitizer run failed\n\nstdout diagnostic");
      expect(result).not.toContain("stderr shadow");
    } finally {
      fixture.cleanup();
    }
  });

  it("falls back to stderr and preserves ENOENT hinting", async () => {
    setDollarError(buildDollarFailure({ stderr: "enoent backend missing", message: "spawn ENOENT" }));
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "demo_binary",
        sanitizer: "asan",
      });

      expect(result).toContain("ERROR: Sanitizer run failed");
      expect(result).toContain("enoent backend missing");
      expect(result).toContain("Missing backing script .opencode/tools/run_sanitizers.py");
    } finally {
      fixture.cleanup();
    }
  });

  it("uses a deterministic message-only ENOENT error envelope", async () => {
    setDollarError(buildDollarFailure({ message: "spawn ENOENT" }));
    const execute = await loadToolExecute("../../run_sanitizers_basic.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "demo_binary",
        sanitizer: "asan",
      });

      expect(result).toContain("ERROR: Sanitizer run failed");
      expect(result).toContain("Failed to run sanitizer: spawn ENOENT");
      expect(result).toContain("Missing backing script .opencode/tools/run_sanitizers.py");
    } finally {
      fixture.cleanup();
    }
  });

});
