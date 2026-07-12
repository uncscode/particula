import { afterEach, beforeEach, describe, expect, it } from "bun:test";

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

describe("auto_mode_manifest wrapper", () => {
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

  it("omits blank segment_size", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "init-from-batch",
      adw_id: "A1B2C3D4",
      segment_size: "   ",
      options: "force",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "init-from-batch",
      "--adw-id",
      "a1b2c3d4",
      "--force",
    ]);
  });

  it("preserves ship strategy and branch filters", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "init",
      issues: "42,43",
      depends: "43:42",
      source_branch: "epic/e14-auto",
      target_branch: "develop",
      branch_type: "epic",
      ship_strategy: "accumulate",
      segment_size: 0,
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "init",
      "--issues",
      "42,43",
      "--depends",
      "43:42",
      "--source-branch",
      "epic/e14-auto",
      "--target-branch",
      "develop",
      "--branch-type",
      "epic",
      "--segment-size",
      "0",
      "--ship-strategy",
      "accumulate",
    ]);
  });

  it("routes init force token without changing direct payload fields", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "init",
      issues: "42,43",
      depends: "43:42",
      source_branch: "epic/e14-auto",
      target_branch: "develop",
      branch_type: "epic",
      ship_strategy: "pr",
      options: "force",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "init",
      "--issues",
      "42,43",
      "--depends",
      "43:42",
      "--source-branch",
      "epic/e14-auto",
      "--target-branch",
      "develop",
      "--branch-type",
      "epic",
      "--ship-strategy",
      "pr",
      "--force",
    ]);
  });

  it("routes status and reset toggles without behavior change", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const statusResult = await execute({
      command: "status",
      branch: "epic/e14-auto",
      options: "json",
    });
    const resetResult = await execute({
      command: "reset",
      issue: "42",
      branch: "epic/e14-auto",
      options: "resume force",
    });

    expect(statusResult).toBe("ok");
    expect(resetResult).toBe("ok");
    expect(getInvocations()[0]?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "status",
      "--branch",
      "epic/e14-auto",
      "--json",
    ]);
    expect(getInvocations()[1]?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "reset",
      "--issue",
      "42",
      "--branch",
      "epic/e14-auto",
      "--resume",
      "--force",
    ]);
  });

  it("routes complete with direct payload fields and bounded toggles", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "complete",
      issue: "42",
      adw_id: "A1B2C3D4",
      branch: "epic/e14-auto",
      completed_at: "2026-06-27T23:59:59Z",
      detail: "Issue completed (branch accumulation).",
      options: "force dry-run branch-merged",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "complete",
      "--issue",
      "42",
      "--adw-id",
      "a1b2c3d4",
      "--branch",
      "epic/e14-auto",
      "--completed-at",
      "2026-06-27T23:59:59Z",
      "--detail",
      "Issue completed (branch accumulation).",
      "--force",
      "--dry-run",
      "--branch-merged",
    ]);
  });

  it("routes complete with no-branch-merged and omits blank optional payload fields", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "complete",
      issue: "42",
      adw_id: "deadbeef",
      completed_at: "   ",
      detail: "   ",
      options: "no-branch-merged",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "complete",
      "--issue",
      "42",
      "--adw-id",
      "deadbeef",
      "--no-branch-merged",
    ]);
  });

  it("routes delete and prune with bounded non-interactive options", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const deleteResult = await execute({
      command: "delete",
      branch: "epic/e14-auto",
      options: "dry-run",
    });
    const pruneResult = await execute({
      command: "prune",
      completed: true,
      options: "force",
    });

    expect(deleteResult).toBe("ok");
    expect(pruneResult).toBe("ok");
    expect(getInvocations()[0]?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "delete",
      "--branch",
      "epic/e14-auto",
      "--dry-run",
    ]);
    expect(getInvocations()[1]?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "prune",
      "--completed",
      "--force",
    ]);
  });

  it("routes delete --force and prune --dry-run symmetrically", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const deleteResult = await execute({
      command: "delete",
      branch: "epic/e14-auto",
      options: "force",
    });
    const pruneResult = await execute({
      command: "prune",
      completed: true,
      options: "dry-run",
    });

    expect(deleteResult).toBe("ok");
    expect(pruneResult).toBe("ok");
    expect(getInvocations()[0]?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "delete",
      "--branch",
      "epic/e14-auto",
      "--force",
    ]);
    expect(getInvocations()[1]?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "prune",
      "--completed",
      "--dry-run",
    ]);
  });

  it("documents workflow-context requirements for manual completion", async () => {
    await loadToolExecute("../../auto_mode_manifest.ts");

    const definition = getCapturedToolDefinition();
    expect(definition?.description).toContain("completion requires workflow context");
    expect(definition?.description).toContain("adw_id must match the persisted issue record");
    expect(definition?.args).toHaveProperty("completed_at");
    expect(definition?.args).toHaveProperty("detail");
  });

  it("preserves dependency args for single and multi dependency inputs", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "init",
      issues: "42,43,44",
      depends: "43:42,44:43",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "auto-mode",
      "init",
      "--issues",
      "42,43,44",
      "--depends",
      "43:42",
      "--depends",
      "44:43",
    ]);
  });

  it("rejects duplicate options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({ command: "status", options: "json json" });

    expect(String(result)).toContain("duplicate 'json' token is not allowed");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects command-disallowed options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({ command: "validate", options: "json" });

    expect(String(result)).toContain("token is not allowed for command 'validate'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects mutually exclusive complete branch-merged toggles before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({
      command: "complete",
      issue: "42",
      adw_id: "deadbeef",
      options: "branch-merged no-branch-merged",
    });

    expect(String(result)).toContain("cannot be combined");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects complete when required fields are missing or invalid before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    expect(String(await execute({ command: "complete", adw_id: "deadbeef" }))).toContain(
      "'issue' is required for complete",
    );
    expect(String(await execute({ command: "complete", issue: "42" }))).toContain(
      "'adw_id' is required for complete",
    );
    expect(
      String(await execute({ command: "complete", issue: "0", adw_id: "deadbeef" })),
    ).toContain('Issue must be a positive integer');
    expect(
      String(await execute({ command: "complete", issue: "42", adw_id: "not-hex" })),
    ).toContain('Must be an 8-character hex string');
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects delete and prune when required selectors are missing or invalid before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    expect(String(await execute({ command: "delete", options: "force" }))).toContain(
      "'branch' is required for delete",
    );
    expect(
      String(await execute({ command: "delete", branch: "--bad", options: "force" })),
    ).toContain("Invalid branch name");
    expect(String(await execute({ command: "prune", options: "force" }))).toContain(
      "'completed: true' is required for prune",
    );
    expect(
      String(await execute({ command: "prune", completed: false, options: "dry-run" })),
    ).toContain("'completed: true' is required for prune");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects interactive delete and prune invocations without dry-run or force", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const deleteResult = await execute({ command: "delete", branch: "feature/a" });
    const pruneResult = await execute({ command: "prune", completed: true });

    expect(String(deleteResult)).toContain("Interactive 'delete' calls are not allowed via wrapper");
    expect(String(pruneResult)).toContain("Interactive 'prune' calls are not allowed via wrapper");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects command-disallowed tokens for delete and prune before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const deleteResult = await execute({
      command: "delete",
      branch: "feature/a",
      options: "resume",
    });
    const pruneResult = await execute({
      command: "prune",
      completed: true,
      options: "json",
    });

    expect(String(deleteResult)).toContain("token is not allowed for command 'delete'");
    expect(String(pruneResult)).toContain("token is not allowed for command 'prune'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects invalid options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({ command: "init", issues: "42", options: "force=true" });

    expect(String(result)).toContain("bare tokens only; '=value' is not supported");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-kebab-case options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({ command: "status", options: "Json" });

    expect(String(result)).toContain("token names must use lowercase-kebab-case");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-string options with deterministic error instead of throwing", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const result = await execute({ command: "status", options: 7 as unknown as string });

    expect(String(result)).toContain("ERROR: 'options' must be a string when provided.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects option-like branch inputs before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const statusResult = await execute({ command: "status", branch: "--json" });
    const initResult = await execute({
      command: "init",
      issues: "42",
      source_branch: "--force",
    });

    expect(String(statusResult)).toContain("ERROR: Invalid branch name: --json");
    expect(String(initResult)).toContain("ERROR: Invalid branch name: --force");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects whitespace-only branch values before spawn", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");

    const deleteResult = await execute({
      command: "delete",
      branch: "   ",
      options: "force",
    });
    const statusResult = await execute({ command: "status", branch: "\t" });

    expect(String(deleteResult)).toContain("Branch name cannot be blank or whitespace-only");
    expect(String(statusResult)).toContain("Branch name cannot be blank or whitespace-only");
    expect(getInvocations()).toHaveLength(0);
  });

  it("fails on malformed json dependency metadata via cli envelope passthrough", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");
    setDollarText("ERROR: Malformed JSON dependency metadata.");

    const result = await execute({ command: "init-from-batch", adw_id: "deadbeef" });

    expect(result).toContain("ERROR: adw auto-mode command failed.");
    expect(result).toContain("ERROR: Malformed JSON dependency metadata.");
  });

  it("fails on decoded non-list dependency metadata via cli envelope passthrough", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");
    setDollarText("ERROR: Decoded dependency metadata must be a list.");

    const result = await execute({ command: "init-from-batch", adw_id: "deadbeef" });

    expect(result).toContain("ERROR: adw auto-mode command failed.");
    expect(result).toContain("ERROR: Decoded dependency metadata must be a list.");
  });

  it("fails on ambiguous identifier dependency tokens via cli envelope precedence", async () => {
    const execute = await loadToolExecute("../../auto_mode_manifest.ts");
    setDollarError({ stdout: "ERROR: Ambiguous dependency identifier token.", stderr: "" });

    const result = await execute({ command: "init-from-batch", adw_id: "deadbeef" });

    expect(result).toContain("ERROR: adw auto-mode command failed.");
    expect(result).toContain("ERROR: Ambiguous dependency identifier token.");
  });
});
