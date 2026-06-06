import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import path from "node:path";
import { tmpdir } from "node:os";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
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

    const result = await execute({ cwd: process.cwd() });
    expect(result).toBe("Validated .opencode/agent references successfully.");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("validate_agent_references.py");
    expect(cmd).toContain(`--root=${process.cwd()}`);
  });

  it("defaults validation root to the current repository root when cwd is omitted", async () => {
    setDollarText(buildSuccessOutput("Validated default root successfully."));
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({});
    expect(result).toBe("Validated default root successfully.");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain(`--root=${process.cwd()}`);
  });

  it("prefers validator stdout over stderr for failure diagnostics", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(buildDollarFailure({ stdout: "validator stdout diagnostic", stderr: "stderr shadow" }));

    const result = await execute({ cwd: process.cwd() });
    assertContains(String(result), "validator stdout diagnostic");
    expect(result).not.toContain("stderr shadow");
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

    const result = await execute({ cwd: process.cwd() });
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

    const result = await execute({ cwd: process.cwd() });
    assertContains(String(result), "ERROR: Agent reference validation failed");
    assertContains(String(result), "exception-approved historical context");
    expect(String(result)).not.toContain("Ensure python3 is installed and on your PATH");
  });

  it("returns stderr-only failure diagnostics when stdout is empty", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "stderr-only diagnostic" }));

    const result = await execute({ cwd: process.cwd() });
    expect(result).toBe("ERROR: Agent reference validation failed\n\nstderr-only diagnostic");
  });

  it("returns message-only failure diagnostics when stdout and stderr are empty", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "", message: "message-only diagnostic" }));

    const result = await execute({ cwd: process.cwd() });
    expect(result).toBe(
      "ERROR: Failed to run agent reference validation: message-only diagnostic",
    );
  });

  it("rejects blank cwd before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const result = await execute({ cwd: "   " });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("cwd must not be blank");
    expect(getInvocations()).toHaveLength(0);
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
    const nestedPath = path.join(process.cwd(), ".opencode");

    const result = await execute({ cwd: nestedPath });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("cwd must resolve to the current repository/worktree root");
    expect(result).toContain(`canonical: ${nestedPath}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing cwd paths before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const missingPath = path.join(process.cwd(), ".tmp-missing-run-validate-agent-references");

    const result = await execute({ cwd: missingPath });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain(`cwd path does not exist: ${missingPath}`);
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects cwd paths that are files before subprocess execution", async () => {
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");
    const tempDir = mkdtempSync(path.join(tmpdir(), "run-validate-agent-references-"));
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

    const result = await execute({ cwd: process.cwd() });

    expect(result).toBe(
      "ERROR: run_validate_agent_references only runs the committed validator script; revert local edits to scripts/validate_agent_references.py before retrying.",
    );
    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.kind).toBe("spawnSync");
  });

  it("returns deterministic diagnostics when validator trust verification fails", async () => {
    setSpawnResponse({ stdout: "", stderr: "fatal: not a git repository", exitCode: 128 });
    const execute = await loadToolExecute("../../run_validate_agent_references.ts");

    const result = await execute({ cwd: process.cwd() });

    expect(result).toBe(
      "ERROR: Failed to verify validator script trust state\n\nfatal: not a git repository",
    );
    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.kind).toBe("spawnSync");
  });
});
