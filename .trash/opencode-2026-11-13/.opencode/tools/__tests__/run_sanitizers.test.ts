import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";

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
  const buildDir = mkdtempSync(path.join(repoRoot, "sanitizers-compat-build-"));
  const executable = path.join(buildDir, "compat_binary");
  writeFileSync(executable, "#!/bin/sh\nexit 0\n", "utf8");
  return {
    buildDir,
    executable,
    cleanup: () => rmSync(buildDir, { recursive: true, force: true }),
  };
}

describe("run_sanitizers compatibility wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("routes to basic for routine payload", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "asan",
        timeout: 12,
      });

      expect(result).toBe("ok");
      const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
      expect(cmd).toContain("--timeout=12");
      expect(cmd).not.toContain("--options=");
      expect(cmd).not.toContain("--normal-duration=");
    } finally {
      fixture.cleanup();
    }
  });

  it("routes to advanced when options key is present but empty", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "tsan",
        options: "",
      });

      expect(result).toBe("ok");
      const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
      expect(cmd).toContain("--sanitizer=tsan");
      expect(cmd).not.toContain("does not accept advanced option");
    } finally {
      fixture.cleanup();
    }
  });

  it("routes to advanced on falsey advanced-key presence", async () => {
    setDollarText(buildSuccessOutput("ok"));
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "asan",
        extraArgs: [],
      });

      const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
      expect(cmd).toContain(`--build-dir=${fixture.buildDir}`);
    } finally {
      fixture.cleanup();
    }
  });

  it("preserves split validation errors in compatibility mode", async () => {
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "asan",
        normalDuration: 0,
      });

      expect(result).toContain("normalDuration must be positive");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("preserves split subprocess failure diagnostics in compatibility mode", async () => {
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "ubsan",
      });

      expect(result).toBe("ERROR: Sanitizer run failed\n\nstdout diagnostic");
      expect(result).not.toContain("stderr shadow");
    } finally {
      fixture.cleanup();
    }
  });

  it("preserves split suppressions validation in compatibility mode", async () => {
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "tsan",
        suppressions: "missing.suppr",
      });

      expect(result).toContain("suppressions path does not exist: missing.suppr");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });

  it("preserves integer timeout validation in compatibility mode", async () => {
    const execute = await loadToolExecute("../../run_sanitizers.ts");
    const fixture = makeBuildFixture();

    try {
      const result = await execute({
        buildDir: fixture.buildDir,
        executable: "compat_binary",
        sanitizer: "asan",
        timeout: 3.25,
      });

      expect(result).toContain("Timeout must be a positive integer");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fixture.cleanup();
    }
  });
});
