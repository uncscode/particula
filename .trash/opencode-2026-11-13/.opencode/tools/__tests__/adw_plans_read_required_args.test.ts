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

describe("adw_plans_read strict input contract", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("exposes required cwd and direct command-specific booleans", async () => {
    await loadToolExecute("../../adw_plans_read.ts");
    const args = getCapturedToolDefinition()?.args ?? {};

    expect(args).toHaveProperty("cwd");
    expect(args).toHaveProperty("json");
    expect(args).toHaveProperty("populate");
    expect(args).toHaveProperty("check");
    expect(args).not.toHaveProperty("status");
  });

  it("requires a direct canonical cwd before spawning", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    for (const cwd of [undefined, null, "", "  ", "-value", 1]) {
      const result = await execute({ command: "list", cwd } as any);
      assertContains(String(result), "cwd");
    }
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects nested paths and another worktree before spawning", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const nested = resolve(repoRoot, ".opencode");
    const result = await execute({ command: "list", cwd: nested });

    assertContains(String(result), "not a repository/worktree root");
    expect(getInvocations()).toHaveLength(0);
  });

  it("forwards an admitted cwd as CLI cwd and subprocess cwd", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    await execute({ command: "list", cwd: repoRoot });

    expect(getInvocations()).toHaveLength(1);
    expect(getInvocations()[0]?.args).toEqual([
      "uv", "run", "--active", "adw", "plans", "list", "--cwd", repoRoot,
    ]);
    expect(getInvocations()[0]?.cwd).toBe(repoRoot);
  });

  it("forwards true direct booleans and omits false values", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    await execute({ command: "list", json: true, cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toContain("--json");

    await execute({ command: "show", plan_id: "M1", json: false, cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).not.toContain("--json");

    await execute({ command: "schema", check: true, cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toContain("--check");

    await execute({ command: "list-sections", plan_id: "M1", populate: true, cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toContain("--populate");
  });

  it("rejects boolean tokens, wrong boolean types, and cross-command booleans pre-spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const cases = [
      { command: "list", options: "json", cwd: repoRoot },
      { command: "list", json: "true", cwd: repoRoot },
      { command: "list", check: true, cwd: repoRoot },
      { command: "schema", json: false, cwd: repoRoot },
      { command: "show", plan_id: "M1", populate: false, cwd: repoRoot },
    ];

    for (const args of cases) {
      const result = await execute(args as any);
      expect(String(result)).toMatch(/must be a boolean|not accepted|token is not allowed/);
    }
    expect(getInvocations()).toHaveLength(0);
  });

  it("retains only command-allowed value options", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    await execute({ command: "list", options: "status=In Progress", cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toContain("In Progress");

    const result = await execute({ command: "show", plan_id: "M1", options: "status=Ready", cwd: repoRoot });
    assertContains(String(result), "token is not allowed for this command");
  });
});
