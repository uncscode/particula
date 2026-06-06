import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnError,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const repoRoot = resolve(import.meta.dir, "../../..");

function createOtherRepositoryRoot(): string {
  const tempRoot = mkdtempSync(`${tmpdir()}/adw-plans-compat-`);
  writeFileSync(resolve(tempRoot, ".git"), "gitdir: /tmp/fake\n");
  return tempRoot;
}

describe("adw_plans compatibility required-arg preflight", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("rejects whitespace-only required plan_id for show before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "show", plan_id: "   " });
    assertContains(String(result), "show command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing required plan_id for show before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "show" });
    assertContains(String(result), "show command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null required plan_id for show before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "show", plan_id: null as any });
    assertContains(String(result), "show command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects wrong-type title for create before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: { bad: true } as any });
    assertContains(String(result), "create command requires 'title'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing cwd for mutating commands before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "My Plan" });
    assertContains(String(result), "create command requires 'cwd'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("omits optional blank parent while still spawning valid list command", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "list", parent: "   ", json: true });
    const args = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(args).toContain("uv run adw plans list --json");
    expect(args).not.toContain("--parent");
  });

  it("redacts cwd path in preflight diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list", cwd: "/definitely/missing/path" });
    assertContains(String(result), "cwd path does not exist");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns deterministic cwd not-a-directory preflight error before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list", cwd: "/dev/null" });
    assertContains(String(result), "ERROR: cwd path is not a directory: <path>");
    expect(String(result)).not.toContain("/dev/null");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns deterministic non-repository cwd preflight error before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list", cwd: "/tmp" });
    assertContains(String(result), "ERROR: cwd path is not a repository/worktree root: <path>");
    assertContains(String(result), "missing .git metadata at <path>");
    expect(String(result)).not.toContain("/tmp");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects mutating cwd values that resolve to a different repository/worktree root", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const otherRepoRoot = createOtherRepositoryRoot();
    const result = await execute({
      command: "create",
      plan_type: "feature",
      title: "x",
      cwd: otherRepoRoot,
    });
    assertContains(String(result), "ERROR: cwd path resolves outside repository root: <path> (canonical: <path>)");
    expect(getInvocations()).toHaveLength(0);
    rmSync(otherRepoRoot, { recursive: true, force: true });
  });

  it("accepts current worktree root for mutating commands in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(
      await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }),
    );
    expect(result).toBe("ADW Plans Command: create\n\nok");
    expect(getInvocations().at(-1)?.args).toContain(repoRoot);
  });

  it("rejects numeric zero issue_number in update-phase args before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      issue_number: 0,
      cwd: ".",
    });
    assertContains(String(result), "'issue_number' must be a positive safe integer when provided.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("prefers stderr over stdout in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "stderr diagnostics", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list" });
    assertContains(String(result), "stderr diagnostics");
    expect(String(result)).not.toContain("stdout diagnostics");
  });

  it("uses stdout when stderr is empty in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list" });
    assertContains(String(result), "stdout diagnostics");
  });

  it("uses stdout when stderr is whitespace-only in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "   \n", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "stdout diagnostics");
    expect(result).not.toContain("   \n");
  });

  it("redacts absolute paths and adds runtime hint for spawned-command failures", async () => {
    setSpawnResponse({
      stderr: "python3: can't open file /abs/path/backend.py: No such file or directory",
      exitCode: 2,
    });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "<path>");
    expect(result).not.toContain("/abs/path/backend.py");
    assertContains(
      result,
      "hint: verify the required runtime/tooling is installed and the backend script exists in this repository.",
    );
  });

  it("redacts Windows and spaced absolute paths in spawned-command failures", async () => {
    setSpawnResponse({
      stderr: 'backend failed at "C:\\Users\\me\\Project Files\\backend.py" and /tmp/path with spaces/backend.ts: failed',
      exitCode: 2,
    });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).toContain("<path>");
    expect(result).toContain("<path>: failed");
    expect(result).not.toContain("C:\\Users\\me\\Project Files\\backend.py");
    expect(result).not.toContain("/tmp/path with spaces/backend.ts");
  });

  it("redacts token-like secrets in spawned-command failures", async () => {
    setSpawnResponse({
      stderr: 'Authorization: Bearer ghp_secretToken123 token=abc123 /tmp/private.txt',
      exitCode: 2,
    });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).toContain("Authorization: Bearer <redacted-secret>");
    expect(result).toContain("token=<redacted-secret>");
    expect(result).toContain("<path>");
    expect(result).not.toContain("ghp_secretToken123");
    expect(result).not.toContain("abc123");
    expect(result).not.toContain("/tmp/private.txt");
  });

  it("adds cwd hint only for true cwd/path execution failures", async () => {
    setSpawnResponse({ stderr: "ERROR: cwd path does not exist: /tmp/worktree", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "hint: verify --cwd points to an existing in-repository repository/worktree root.");
  });

  it("does not add cwd hint for unrelated worktree wording", async () => {
    setSpawnResponse({ stderr: "plan worktree metadata is stale", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).not.toContain("hint: verify --cwd points to an existing in-repository repository/worktree root.");
  });

  it("preserves bounded truncation marker for long diagnostics", async () => {
    setSpawnResponse({ stderr: `failure:${"x".repeat(5000)}`, exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "...[output truncated to 4000 characters; original length");
  });

  it("uses catch-path message fallback when stderr/stdout are empty", async () => {
    setSpawnError({ message: "ENOENT: python3 not found" });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "ENOENT: python3 not found");
    assertContains(
      result,
      "hint: verify the required runtime/tooling is installed and the backend script exists in this repository.",
    );
  });

  it("uses unknown execution error fallback when catch-path diagnostics are empty", async () => {
    setSpawnError({});
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "Unknown execution error");
  });

  it("uses stdout before message in catch-path fallback when stderr is whitespace-only", async () => {
    setSpawnError({ stderr: "   ", stdout: "stdout diagnostics", message: "message third" });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "stdout diagnostics");
    expect(result).not.toContain("message third");
  });

  it("preserves timeout fallback text when timeout has no output", async () => {
    setSpawnResponse({ timedOut: true, exitCode: 1 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "Command timed out after 60000ms");
  });

  it("preserves success-path output", async () => {
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).toBe("ADW Plans Command: list\n\nok");
  });

  it("accepts registry-driven plan_type strings without wrapper allowlist rejection", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "list", plan_type: "research" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw plans list --type research");
  });

  it("update-phase patch-only payload ignores plan-level parent/priority/status in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      parent: "E1",
      priority: "high",
      status: "Ready",
      patch: '{"status":"Shipped"}',
      cwd: repoRoot,
    } as any);

    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update-phase",
      "E1-F1",
      "E1-F1-P1",
      "--patch",
      '{"status":"Shipped"}',
      "--cwd",
      repoRoot,
    ]);
    expect(args).not.toContain("--parent");
    expect(args).not.toContain("--priority");
    expect(args).not.toContain("Ready");
  });

  it("update-phase forwards phase-scoped fields and phase_status only in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      status: "Ready",
      phase_status: "In Progress",
      title: "Phase Title",
      size: "M",
      issue_number: 123,
      clear_issue_number: true,
      patch: '{"owner":"team"}',
      cwd: repoRoot,
    } as any);

    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update-phase",
      "E1-F1",
      "E1-F1-P1",
      "--status",
      "In Progress",
      "--title",
      "Phase Title",
      "--size",
      "M",
      "--issue",
      "123",
      "--clear-issue-number",
      "--patch",
      '{"owner":"team"}',
      "--cwd",
      repoRoot,
    ]);
    expect(args).not.toContain("Ready");
    expect(args.filter((arg) => arg === "--status")).toHaveLength(1);
  });
});
