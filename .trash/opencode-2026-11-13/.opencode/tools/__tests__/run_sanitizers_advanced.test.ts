import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

function makeBuildFixture(): { buildDir: string; executable: string; cleanup: () => void } {
  const repoRoot = path.resolve(import.meta.dir, "../../..");
  const buildDir = mkdtempSync(path.join(repoRoot, "sanitizers-advanced-build-"));
  const executable = path.join(buildDir, "advanced_binary");
  writeFileSync(executable, "#!/bin/sh\nexit 0\n", "utf8");
  return {
    buildDir,
    executable,
    cleanup: () => rmSync(buildDir, { recursive: true, force: true }),
  };
}

function makeSuppressionsFixture(): {
  suppressionsPath: string;
  relativePath: string;
  cleanup: () => void;
} {
  const repoRoot = path.resolve(import.meta.dir, "../../..");
  const suppressionsDir = mkdtempSync(path.join(repoRoot, "sanitizers-suppressions-"));
  const suppressionsPath = path.join(suppressionsDir, "tsan.suppr");
  writeFileSync(suppressionsPath, "race:SomeLibrary\n", "utf8");
  return {
    suppressionsPath,
    relativePath: path.relative(repoRoot, suppressionsPath),
    cleanup: () => rmSync(suppressionsDir, { recursive: true, force: true }),
  };
}

describe("run_sanitizers_advanced wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("forwards advanced controls for valid payload", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();
    const suppressionsFixture = makeSuppressionsFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "tsan",
        suppressions: suppressionsFixture.relativePath,
        options: "halt_on_error=1",
        normalDuration: 1.5,
        extraArgs: ["--gtest_filter=RaceSuite.*"],
      });

      expect(result).toBe("ok");
      const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
      expect(cmd).toContain(`--build-dir=${fixture.buildDir}`);
      expect(cmd).toContain(`--executable=${fixture.executable}`);
      expect(cmd).toContain("--sanitizer=tsan");
      expect(cmd).toContain(`--suppressions=${suppressionsFixture.suppressionsPath}`);
      expect(cmd).toContain("--options=halt_on_error=1");
      expect(cmd).toContain("--normal-duration=1.5");
      expect(cmd).toContain("-- --gtest_filter=RaceSuite.*");
    } finally {
      suppressionsFixture.cleanup();
      fixture.cleanup();
    }
  });

  it("rejects non-array extraArgs", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "asan",
        extraArgs: "--bad-shape",
      });

      expect(result).toContain("extraArgs must be an array of strings");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects non-finite normalDuration", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "asan",
        normalDuration: Number.NaN,
      });

      expect(result).toContain("normalDuration must be positive");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects non-finite timeout", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "asan",
        timeout: Number.POSITIVE_INFINITY,
      });

      expect(result).toContain("Timeout must be a positive integer");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects float timeout", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "asan",
        timeout: 2.5,
      });

      expect(result).toContain("Timeout must be a positive integer");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects missing suppressions files before spawn", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "tsan",
        suppressions: "missing.suppr",
      });

      expect(result).toContain("suppressions path does not exist: missing.suppr");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects out-of-scope absolute suppressions files before spawn", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();
    const outsideDir = mkdtempSync(path.join(tmpdir(), "sanitizers-suppressions-outside-"));
    const outsideSuppressions = path.join(outsideDir, "tsan.suppr");
    writeFileSync(outsideSuppressions, "race:Outside\n", "utf8");

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "tsan",
        suppressions: outsideSuppressions,
      });

      expect(result).toContain(`suppressions path resolves outside repository root: ${outsideSuppressions}`);
      expect(getInvocations()).toHaveLength(0);
    } finally {
      rmSync(outsideDir, { recursive: true, force: true });
      fixture.cleanup();
    }
  });

  it("rejects unreadable suppressions files before spawn", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();
    const suppressionsFixture = makeSuppressionsFixture();

    try {
      chmodSync(suppressionsFixture.suppressionsPath, 0o000);
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "tsan",
        suppressions: suppressionsFixture.relativePath,
      });

      expect(result).toContain(`suppressions path is not readable: ${suppressionsFixture.relativePath}`);
      expect(getInvocations()).toHaveLength(0);
    } finally {
      chmodSync(suppressionsFixture.suppressionsPath, 0o644);
      suppressionsFixture.cleanup();
      fixture.cleanup();
    }
  });

  it("rejects invalid sanitizer names", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "msan",
      });

      expect(result).toContain("sanitizer must be one of asan, tsan, ubsan");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("rejects path violations before spawn", async () => {
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const outsideBuildDir = mkdtempSync(path.join(tmpdir(), "sanitizers-advanced-outside-"));

    try {
      const result = await execute({
        buildDir: outsideBuildDir,
        executable: "advanced_binary",
        sanitizer: "asan",
      });

      assertErrorPrefix(String(result), "ERROR:");
      expect(result).toContain(`buildDir path resolves outside repository root: ${outsideBuildDir}`);
      expect(getInvocations()).toHaveLength(0);
    } finally {
      rmSync(outsideBuildDir, { recursive: true, force: true });
    }
  });

  it("returns deterministic stderr failure envelope when stdout absent", async () => {
    setDollarError(buildDollarFailure({ stderr: "backend failed" }));
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "ubsan",
      });

      expect(result).toBe("ERROR: Sanitizer run failed\n\nbackend failed");
    } finally {
      fixture.cleanup();
    }
  });

  it("wraps stdout-present failures in a deterministic error envelope", async () => {
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
        sanitizer: "ubsan",
      });

      expect(result).toBe("ERROR: Sanitizer run failed\n\nstdout diagnostic");
      expect(result).not.toContain("stderr shadow");
    } finally {
      fixture.cleanup();
    }
  });

  it("uses a deterministic message-only ENOENT error envelope", async () => {
    setDollarError(buildDollarFailure({ message: "spawn ENOENT" }));
    const execute = await loadToolExecute("../../run_sanitizers_advanced.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "advanced_binary",
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
