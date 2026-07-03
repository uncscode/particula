import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { ERROR_PRECEDENCE_FIXTURES } from "./fixtures/wrapper_contract_fixtures";
import { buildDollarFailure, buildSuccessOutput } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnResponse,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const REPO_ROOT = path.resolve(import.meta.dir, "../../..");
const REPO_TMP_ROOT = path.join(REPO_ROOT, "adforge_local", "opencode", "tmp");
const BASELINE_FIXTURE_PATH = path.join(
  REPO_ROOT,
  ".opencode",
  "guides",
  "agent-reference-validation-baseline.json",
);

function withRepoBaselineFixture(): () => void {
  mkdirSync(path.dirname(BASELINE_FIXTURE_PATH), { recursive: true });
  writeFileSync(BASELINE_FIXTURE_PATH, '{"version":1,"errors":[]}\n', "utf8");
  return () => rmSync(BASELINE_FIXTURE_PATH, { force: true });
}

describe("run_validate_agent_references wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("returns validator stdout on success and assembles python3 plus validator root", async () => {
    setDollarText(buildSuccessOutput("Validated .opencode/agent references successfully."));
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({ cwd: REPO_ROOT });
    expect(result).toBe("Validated .opencode/agent references successfully.");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("validate_agent_references.py");
    expect(cmd).toContain(`--root=${REPO_ROOT}`);
  });

  it("forwards an explicit baseline path when provided", async () => {
    setDollarText(buildSuccessOutput("Validated with baseline."));
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({
      cwd: REPO_ROOT,
      baselinePath: ".opencode/guides/agent-reference-validation-baseline.json",
    });
    expect(result).toBe("Validated with baseline.");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--baseline-path=.opencode/guides/agent-reference-validation-baseline.json");
  });

  it("defaults validation root to the current repository root when cwd is omitted", async () => {
    setDollarText(buildSuccessOutput("Validated default root successfully."));
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({});
    expect(result).toBe("Validated default root successfully.");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain(`--root=${REPO_ROOT}`);
  });

  it("prefers validator stdout over stderr for failure diagnostics", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.failure);

    const result = await execute({ cwd: REPO_ROOT });
    assertContains(String(result), ERROR_PRECEDENCE_FIXTURES.stdoutFirst.preferred);
    expect(result).not.toContain(ERROR_PRECEDENCE_FIXTURES.stdoutFirst.shadowed);
  });

  it("returns deterministic missing-runtime or missing-script guidance", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(
      buildDollarFailure({
        stdout: "",
        stderr: "python3: can't open file '/x/validate_agent_references.py': [Errno 2] No such file or directory",
        message: "spawn python3 ENOENT",
      }),
    );

    const result = await execute({ cwd: REPO_ROOT });
    assertContains(String(result), "ERROR: Agent reference validation failed");
    assertContains(String(result), "Ensure python3 is installed and on your PATH");
    assertContains(String(result), "scripts/validate_agent_references.py exists");
  });

  it("does not append missing-runtime guidance for normal validator stderr failures", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(
      buildDollarFailure({
        stdout: "",
        stderr: "ERROR: wrapper 'adw' requires exception-approved historical context",
        message: "validator reported failures",
      }),
    );

    const result = await execute({ cwd: REPO_ROOT });
    assertContains(String(result), "ERROR: Agent reference validation failed");
    assertContains(String(result), "exception-approved historical context");
    expect(String(result)).not.toContain("Ensure python3 is installed and on your PATH");
  });

  it("returns stderr-only failure diagnostics when stdout is empty", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(ERROR_PRECEDENCE_FIXTURES.stderrFallback);

    const result = await execute({ cwd: REPO_ROOT });
    expect(result).toBe(
      `ERROR: Agent reference validation failed\n\n${ERROR_PRECEDENCE_FIXTURES.stderrFallback.stderr}`,
    );
  });

  it("returns message-only failure diagnostics when stdout and stderr are empty", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(ERROR_PRECEDENCE_FIXTURES.messageOnly);

    const result = await execute({ cwd: REPO_ROOT });
    expect(result).toBe(
      `ERROR: Failed to run agent reference validation: ${ERROR_PRECEDENCE_FIXTURES.messageOnly.message}`,
    );
  });

  it("keeps stdout precedence for baseline-aware validator failures", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(
      buildDollarFailure({
        stdout: "ERROR: Missing agent references detected:\n  new or unbaselined failures:\n    sample",
        stderr: "shadow stderr",
        message: "shadow message",
      }),
    );

    const result = await execute({
      cwd: REPO_ROOT,
      baselinePath: ".opencode/guides/agent-reference-validation-baseline.json",
    });
    assertContains(String(result), "new or unbaselined failures");
    expect(String(result)).not.toContain("shadow stderr");
  });

  it("rejects blank cwd before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const result = await execute({ cwd: "   " });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("cwd must not be blank");
    expect(getInvocations()).toHaveLength(0);
  });

  it("treats blank baselinePath as omitted before subprocess execution", async () => {
    setDollarText(buildSuccessOutput("Validated without baseline."));
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const result = await execute({ baselinePath: "   " });

    expect(result).toBe("Validated without baseline.");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain(`--root=${REPO_ROOT}`);
    expect(cmd).not.toContain("--baseline-path=");
  });

  it("rejects baselinePath values outside the canonical guides root", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const absoluteResult = await execute({ baselinePath: path.join(REPO_ROOT, "baseline.json") });
    assertErrorPrefix(String(absoluteResult), "ERROR:");
    expect(String(absoluteResult)).toContain("baselinePath must be repo-relative under .opencode/guides");

    const outsideGuidesResult = await execute({ baselinePath: ".opencode/tools/baseline.json" });
    assertErrorPrefix(String(outsideGuidesResult), "ERROR:");
    expect(String(outsideGuidesResult)).toContain("baselinePath must resolve under .opencode/guides");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects untracked baselinePath values before subprocess execution", async () => {
    const cleanup = withRepoBaselineFixture();
    setSpawnResponse({ stdout: `?? .opencode/guides/agent-reference-validation-baseline.json\n`, stderr: "", exitCode: 0 });
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({
      cwd: REPO_ROOT,
      baselinePath: ".opencode/guides/agent-reference-validation-baseline.json",
    });

    expect(String(result)).toContain("only accepts baselinePath values that point to committed clean files");
    expect(String(result)).toContain("Rejecting untracked baselinePath");
    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.kind).toBe("spawnSync");
    cleanup();
  });

  it("rejects modified baselinePath values before subprocess execution", async () => {
    const cleanup = withRepoBaselineFixture();
    setSpawnResponse({ stdout: ` M .opencode/guides/agent-reference-validation-baseline.json\n`, stderr: "", exitCode: 0 });
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({
      cwd: REPO_ROOT,
      baselinePath: ".opencode/guides/agent-reference-validation-baseline.json",
    });

    expect(String(result)).toContain("only accepts baselinePath values that point to committed clean files");
    expect(String(result)).toContain("Rejecting baselinePath with local git status  M");
    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.kind).toBe("spawnSync");
    cleanup();
  });

  it("rejects cwd outside repository root before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const result = await execute({ cwd: tmpdir() });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain(`cwd path resolves outside repository root: ${tmpdir()}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects nested repository subdirectories before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const nestedPath = path.join(REPO_ROOT, ".opencode");

    const result = await execute({ cwd: nestedPath });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("cwd must resolve to the current repository/worktree root");
    expect(result).toContain(`canonical: ${nestedPath}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing cwd paths before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const missingPath = path.join(REPO_ROOT, ".tmp-missing-run-validate-agent-references");

    const result = await execute({ cwd: missingPath });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain(`cwd path does not exist: ${missingPath}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects cwd paths that are files before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    mkdirSync(REPO_TMP_ROOT, { recursive: true });
    const tempDir = mkdtempSync(path.join(REPO_TMP_ROOT, "run-validate-agent-references-"));
    const filePath = path.join(tempDir, "not-a-directory.txt");
    writeFileSync(filePath, "content", "utf8");

    const result = await execute({ cwd: filePath });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain(`cwd path is not a directory: ${filePath}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects execution when the validator script has local modifications", async () => {
    setSpawnResponse({ stdout: " M scripts/validate_agent_references.py\n", stderr: "", exitCode: 0 });
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({ cwd: REPO_ROOT });

    expect(result).toBe(
      "ERROR: run_validate_agent_references only runs the committed validator script; revert local edits to scripts/validate_agent_references.py before retrying.",
    );
    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.kind).toBe("spawnSync");
  });

  it("returns deterministic diagnostics when validator trust verification fails", async () => {
    setSpawnResponse({ stdout: "", stderr: "fatal: not a git repository", exitCode: 128 });
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({ cwd: REPO_ROOT });

    expect(result).toBe(
      "ERROR: Failed to verify validator script trust state\n\nfatal: not a git repository",
    );
    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.kind).toBe("spawnSync");
  });
});
