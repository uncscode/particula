import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { resolve } from "node:path";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import {
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

const repoRoot = resolve(import.meta.dir, "../../..");

describe("adw_plans_mutate strict input contract", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("exposes required cwd and direct clear_issue_number boolean", async () => {
    await loadToolExecute("../../adw_plans_mutate.ts");
    const args = getCapturedToolDefinition()?.args ?? {};

    expect(args).toHaveProperty("cwd");
    expect(args).toHaveProperty("clear_issue_number");
    expect(args).not.toHaveProperty("status");
    expect(args).not.toHaveProperty("issue_number");
  });

  it("rejects missing or malformed direct cwd before spawning", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    for (const cwd of [undefined, null, "", " ", "-value", false]) {
      const result = await execute({ command: "create", plan_type: "feature", title: "x", cwd } as any);
      assertContains(String(result), "cwd");
    }
    expect(getInvocations()).toHaveLength(0);
  });

  it("forwards an admitted cwd as CLI cwd and subprocess cwd", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot });

    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.args).toEqual([
      "uv", "run", "--active", "adw", "plans", "create", "--type", "feature", "--title", "x", "--cwd", repoRoot,
    ]);
    expect(getInvocations()[0]?.cwd).toBe(repoRoot);
  });

  it("forwards true clear_issue_number and omits false", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const common = { command: "update-phase", plan_id: "M1", phase_id: "M1-P1", cwd: repoRoot };

    await execute({ ...common, clear_issue_number: true });
    expect(getInvocations().at(-1)?.args).toContain("--clear-issue-number");

    await execute({ ...common, clear_issue_number: false });
    expect(getInvocations().at(-1)?.args).not.toContain("--clear-issue-number");
  });

  it("rejects wrong-type, cross-command, and options-encoded booleans pre-spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const cases = [
      { command: "update-phase", plan_id: "M1", phase_id: "M1-P1", clear_issue_number: "true", cwd: repoRoot },
      { command: "create", plan_type: "feature", title: "x", clear_issue_number: false, cwd: repoRoot },
      { command: "update-phase", plan_id: "M1", phase_id: "M1-P1", options: "clear-issue-number", cwd: repoRoot },
    ];

    for (const args of cases) {
      const result = await execute(args as any);
      expect(String(result)).toMatch(/must be a boolean|not accepted|token is not allowed/);
    }
    expect(getInvocations()).toHaveLength(0);
  });

  it("accepts only command-allowed value options and direct patch exceptions", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    await execute({
      command: "update-phase", plan_id: "M1", phase_id: "M1-P1", patch: '{"owner":"team"}',
      options: "phase-status=In Progress size=M issue=42", cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual(expect.arrayContaining([
      "--status", "In Progress", "--size", "M", "--issue", "42", "--patch", '{"owner":"team"}',
    ]));

    const result = await execute({ command: "create", plan_type: "feature", title: "x", options: "issue=42", cwd: repoRoot });
    assertContains(String(result), "token is not allowed for this command");
  });
});
