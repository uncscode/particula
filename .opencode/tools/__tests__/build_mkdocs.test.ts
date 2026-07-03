import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";

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

const REPO_ROOT = path.resolve(import.meta.dir, "../../..");
const REPO_TMP_ROOT = path.join(REPO_ROOT, "adforge_local", "opencode", "tmp");

describe("build_mkdocs wrapper family", () => {
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

  it("preserves compatibility routing defaults when options are omitted", async () => {
    const execute = await loadToolExecute("../../build_mkdocs.ts");

    const result = await execute({ validateOnly: true });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=summary");
    expect(cmd).toContain("--timeout=120");
    expect(cmd).toContain("--validate-only");
    expect(cmd).not.toContain("--no-clean");
    expect(cmd).not.toContain("--strict");
  });

  it("routes bounded options for compatibility build wrapper", async () => {
    const execute = await loadToolExecute("../../build_mkdocs.ts");

    const result = await execute({ options: "output=json strict clean=false" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--strict");
    expect(cmd).toContain("--no-clean");
  });

  it("applies compatibility-wrapper cwd/configFile confinement parity", async () => {
    const execute = await loadToolExecute("../../build_mkdocs.ts");

    const cwdResult = await execute({ cwd: tmpdir() });
    assertErrorPrefix(String(cwdResult), "ERROR:");
    expect(String(cwdResult)).toContain(`cwd path resolves outside repository root: ${tmpdir()}`);

    const configResult = await execute({ configFile: tmpdir() });
    assertErrorPrefix(String(configResult), "ERROR:");
    expect(String(configResult)).toContain(`configFile path resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("always keeps validate wrapper in validate-only mode", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");

    const result = await execute({ options: "output=full strict" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--validate-only");
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--strict");
  });

  it("returns explicit timeout messaging for validate wrapper failures", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");
    setDollarError(buildDollarFailure({ stderr: "mkdocs build timed out after 120 seconds", message: "timeout" }));

    const result = await execute({});

    assertContains(String(result), "ERROR: MkDocs validation timed out");
    assertContains(String(result), "exceeded the wrapper timeout after 120 seconds");
    assertContains(String(result), "defaults to 120 seconds");
  });

  it("does not mislabel ordinary validate-wrapper success output as timeout guidance", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");
    setDollarText("MKDOCS BUILD SUMMARY\nStatus: PASSED");

    const result = await execute({ options: "strict" });

    expect(result).toBe("MKDOCS BUILD SUMMARY\nStatus: PASSED");
    expect(String(result)).not.toContain("timed out");
  });

  it("forwards strict mode while keeping its failure semantics explicit", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");
    setDollarError(buildDollarFailure({ stderr: "strict mode enabled: warnings are treated as failures" }));

    const result = await execute({ options: "strict" });

    assertContains(String(result), "ERROR: MkDocs build failed");
    assertContains(String(result), "strict mode enabled: warnings are treated as failures");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--strict");
  });

  it("build wrapper omits validate-only flag and rejects unsupported options", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_build.ts");

    const success = await execute({ options: "clean=false" });
    expect(success).toBe("ok");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).not.toContain("--validate-only");

    const invalidResult = await execute({ options: "strict=false" });
    assertErrorPrefix(String(invalidResult), "ERROR:");
    expect(String(invalidResult)).toContain("does not accept a value");
  });

  it("rejects duplicate or unsupported mkdocs options before subprocess execution", async () => {
    const validateExecute = await loadToolExecute("../../build_mkdocs_validate.ts");

    expect(await validateExecute({ options: "output=full output=json" })).toContain("duplicate token");
    expect(await validateExecute({ options: "unknown=1" })).toContain("not supported");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects invalid timeout and out-of-root cwd before subprocess execution", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");

    expect(await execute({ timeout: 0 })).toContain("finite positive number");
    const outsideResult = await execute({ cwd: tmpdir() });
    assertErrorPrefix(String(outsideResult), "ERROR:");
    expect(String(outsideResult)).toContain(`cwd path resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("extracts decimal timeout values from message-only failures", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");
    setDollarError(buildDollarFailure({ stderr: "", message: "mkdocs build timed out after 120.5 seconds" }));

    const result = await execute({});

    expect(result).toBe(
      "ERROR: MkDocs validation timed out\n\n" +
        "diagnostic: mkdocs validation exceeded the wrapper timeout after 120.5 seconds\n" +
        "hint: build_mkdocs_validate defaults to 120 seconds; pass a larger direct timeout for slow validations.",
    );
  });

  it("rejects cwd file paths before subprocess execution", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_build.ts");
    mkdirSync(REPO_TMP_ROOT, { recursive: true });
    const tempDir = mkdtempSync(path.join(REPO_TMP_ROOT, "build-mkdocs-test-"));
    const filePath = path.join(tempDir, "not-a-directory.txt");
    writeFileSync(filePath, "content", "utf8");

    const result = await execute({ cwd: filePath });
    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain(`cwd path is not a directory: ${filePath}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("validates configFile confinement and forwards in-repo config paths", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");
    const configFile = path.resolve(import.meta.dir, "build_mkdocs.test.ts");

    const success = await execute({ configFile, options: "output=json" });
    expect(success).toBe("ok");
    let cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain(`--config-file=${configFile}`);
    expect(cmd).toContain("--validate-only");

    resetSubprocessMocks();
    setDollarText("ok");

    const outsideResult = await execute({ configFile: tmpdir() });
    assertErrorPrefix(String(outsideResult), "ERROR:");
    expect(String(outsideResult)).toContain(`configFile path resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("resolves relative configFile paths against cwd before confinement checks", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_validate.ts");
    const repoRoot = path.resolve(import.meta.dir, "../../..");
    const cwd = path.join(repoRoot, ".opencode");

    const result = await execute({ cwd, configFile: "tools/build_mkdocs.py" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--cwd=");
    expect(cmd).toContain("--config-file=tools/build_mkdocs.py");
  });

  it("preserves sanitized ENOENT diagnostics", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_build.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing script" }));

    const result = await execute({ options: "output=full" });
    assertContains(String(result), "ERROR: MkDocs build failed");
    assertContains(String(result), "backing script .opencode/tools/build_mkdocs.py exists");
  });

  it("redacts secrets and applies deterministic truncation markers in split-wrapper diagnostics", async () => {
    const execute = await loadToolExecute("../../build_mkdocs_build.ts");
    const longSecret = `token=ghp_${"a".repeat(40)} ${"x".repeat(5000)}`;
    setDollarError(buildDollarFailure({ stdout: "", stderr: longSecret }));

    const result = await execute({});
    assertContains(String(result), "[REDACTED]");
    expect(String(result)).not.toContain("ghp_");
    assertContains(String(result), "... [truncated]");
  });
});
