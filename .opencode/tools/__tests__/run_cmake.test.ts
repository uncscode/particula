import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("run_cmake wrapper family", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("preserves compatibility-wrapper defaults when options are omitted", async () => {
    const execute = await loadToolExecute("../../run_cmake.ts");

    const result = await execute({ preset: "debug" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=summary");
    expect(cmd).toContain("--timeout=300");
    expect(cmd).toContain("--preset=debug");
    expect(cmd).not.toContain("--build");
    expect(cmd).not.toContain("--jobs=");
    expect(cmd).not.toContain("--build-timeout=");
  });

  it("routes compatibility build-only options only in valid build mode", async () => {
    const execute = await loadToolExecute("../../run_cmake.ts");

    const result = await execute({
      preset: "debug",
      build: true,
      buildTimeout: 44,
      options: "output=full jobs=8",
    });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--build");
    expect(cmd).toContain("--jobs=8");
    expect(cmd).toContain("--build-timeout=44");
  });

  it("keeps wrapper-owned build flags ahead of passthrough cmakeArgs", async () => {
    const execute = await loadToolExecute("../../run_cmake.ts");

    await execute({
      sourceDir: ".",
      buildDir: "build/debug",
      build: true,
      buildTimeout: 44,
      options: "jobs=8",
      cmakeArgs: ["-DTEST=ON"],
    });

    const args = getInvocations().at(-1)?.args ?? [];
    const buildIndex = args.indexOf("--build");
    const jobsIndex = args.indexOf("--jobs=8");
    const timeoutIndex = args.indexOf("--build-timeout=44");
    const separatorIndex = args.indexOf("--");
    expect(buildIndex).toBeGreaterThan(-1);
    expect(jobsIndex).toBeGreaterThan(buildIndex);
    expect(timeoutIndex).toBeGreaterThan(jobsIndex);
    expect(separatorIndex).toBeGreaterThan(timeoutIndex);
  });

  it("rejects compatibility jobs option outside build mode before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_cmake.ts");

    const result = await execute({ preset: "debug", options: "jobs=2" });
    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain("'jobs' requires build: true");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed or duplicate compatibility options tokens", async () => {
    const execute = await loadToolExecute("../../run_cmake.ts");

    expect(await execute({ options: "unknown-token" })).toContain("non-empty '=value' suffix");
    expect(await execute({ options: "output=full output=json" })).toContain("duplicate token");
  });

  it("supports configure-only bounded options and manual configure args", async () => {
    const execute = await loadToolExecute("../../run_cmake_configure.ts");

    const result = await execute({
      sourceDir: "src",
      buildDir: "build/debug",
      options: "output=json ninja",
    });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--source-dir=src");
    expect(cmd).toContain("--build-dir=build/debug");
    expect(cmd).toContain("--ninja");
    expect(cmd).not.toContain(" --build ");
    expect(cmd).not.toContain("--jobs=");
  });

  it("skips manual path validation for ignored preset-mode source/build inputs", async () => {
    const compatibilityExecute = await loadToolExecute("../../run_cmake.ts");
    const configureExecute = await loadToolExecute("../../run_cmake_configure.ts");
    const buildExecute = await loadToolExecute("../../run_cmake_build.ts");

    expect(await compatibilityExecute({ preset: "debug", sourceDir: "/tmp", buildDir: "/tmp/build" })).toBe("ok");
    expect(await configureExecute({ preset: "debug", sourceDir: "/tmp", buildDir: "/tmp/build" })).toBe("ok");
    expect(await buildExecute({ preset: "debug", buildDir: "/tmp/build" })).toBe("ok");
  });

  it("rejects blank configure inputs before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_cmake_configure.ts");

    expect(await execute({ preset: "   " })).toContain("preset must not be blank");
    expect(await execute({ sourceDir: "   " })).toContain("sourceDir must not be blank");
    expect(await execute({ buildDir: "   " })).toContain("buildDir must not be blank");
    expect(await execute({ cmakeArgs: ["-DENABLE_TESTS=ON", "   "] })).toContain(
      "cmakeArgs must not contain blank values",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects out-of-root path inputs before subprocess execution", async () => {
    const compatibilityExecute = await loadToolExecute("../../run_cmake.ts");
    const configureExecute = await loadToolExecute("../../run_cmake_configure.ts");
    const buildExecute = await loadToolExecute("../../run_cmake_build.ts");

    expect(await compatibilityExecute({ sourceDir: "/tmp", buildDir: "build" })).toContain(
      "sourceDir path resolves outside repository root",
    );
    expect(await configureExecute({ sourceDir: "/tmp" })).toContain(
      "sourceDir path resolves outside repository root",
    );
    expect(await buildExecute({ buildDir: "/tmp" })).toContain(
      "buildDir path resolves outside repository root",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects unsupported or duplicate configure-wrapper options before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_cmake_configure.ts");

    expect(await execute({ options: "output=full output=json" })).toContain("duplicate token");
    expect(await execute({ options: "jobs=2" })).toContain("not supported");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves build-wrapper default output when options are omitted", async () => {
    const execute = await loadToolExecute("../../run_cmake_build.ts");

    const result = await execute({ buildDir: "build/debug" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=summary");
    expect(cmd).toContain("--build");
    expect(cmd).toContain("--build-dir=build/debug");
    expect(cmd).not.toContain("--jobs=");
  });

  it("rejects duplicate or unsupported build-wrapper options before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_cmake_build.ts");

    expect(await execute({ buildDir: "build", options: "jobs=2 jobs=4" })).toContain("duplicate token");
    expect(await execute({ buildDir: "build", options: "ninja" })).toContain("requires a non-empty '=value' suffix");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing build context and invalid build timeout guards before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_cmake_build.ts");

    expect(await execute({})).toContain("Build context required");
    expect(await execute({ buildDir: "build", buildTimeout: 0 })).toContain(
      "buildTimeout must be positive",
    );
    expect(await execute({ buildDir: "build", timeout: Number.NaN })).toContain(
      "timeout must be a finite number",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves ENOENT hint precedence for build-wrapper failures", async () => {
    const execute = await loadToolExecute("../../run_cmake_build.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing script" }));

    const result = await execute({ buildDir: "build", options: "output=full" });
    assertContains(String(result), "ERROR: CMake build failed");
    assertContains(String(result), "Missing backing script .opencode/tools/run_cmake.py");
  });
});
