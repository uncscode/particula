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
import {
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

const repoRoot = resolve(import.meta.dir, "../../..");

function createOtherRepositoryRoot(): string {
  const tempRoot = mkdtempSync(`${tmpdir()}/adw-plans-mutate-`);
  writeFileSync(resolve(tempRoot, ".git"), "gitdir: /tmp/fake\n");
  return tempRoot;
}

describe("adw_plans_mutate required-arg preflight", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("exposes options in split-wrapper schema", async () => {
    await loadToolExecute("../../adw_plans_mutate.ts");
    const args = getCapturedToolDefinition()?.args ?? {};
    expect(args).toHaveProperty("options");
    expect(args).not.toHaveProperty("priority");
    expect(args).not.toHaveProperty("size");
    expect(args).not.toHaveProperty("after");
    expect(args).not.toHaveProperty("issue_number");
    expect(args).not.toHaveProperty("clear_issue_number");
  });

  it("keeps mutate-wrapper command schema source scoped to mutate commands", async () => {
    const source = await Bun.file(resolve(import.meta.dir, "../adw_plans_mutate.ts")).text();
    expect(source).toContain('tool.schema.enum(["create", "update", "add-phase", "update-phase", "scaffold-sections"])');
    expect(source).not.toContain('tool.schema.enum(["create", "update", "add-phase", "update-phase", "scaffold-sections", "list"');
  });

  it("rejects whitespace-only required phase_id for update-phase before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "update-phase", plan_id: "E1", phase_id: "  " });
    assertContains(String(result), "update-phase command requires 'phase_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing required plan_type for create before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", title: "My Plan" });
    assertContains(String(result), "create command requires 'plan_type'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects whitespace-only required title for create before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "   " });
    assertContains(String(result), "create command requires 'title'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects wrong-type required plan_type for scaffold-sections before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "scaffold-sections", plan_id: "E1", plan_type: { bad: true } as any });
    assertContains(String(result), "scaffold-sections command requires 'plan_type'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing required plan_id for update before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "update", status: "Ready" });
    assertContains(String(result), "update command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing cwd for mutating commands before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "My Plan" });
    assertContains(String(result), "create command requires 'cwd'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null required title for add-phase before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "add-phase", plan_id: "E1", title: null as any });
    assertContains(String(result), "add-phase command requires 'title'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects read-only commands via mutate gate before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "list" } as any);
    assertContains(String(result), "Unsupported command for adw_plans_mutate: list");
    expect(getInvocations()).toHaveLength(0);
  });

  it("redacts cwd path in preflight diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "x", cwd: "/definitely/missing/path" });
    assertContains(String(result), "cwd path does not exist");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns deterministic cwd not-a-directory preflight error before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "x", cwd: "/dev/null" });
    assertContains(String(result), "ERROR: cwd path is not a directory: <path>");
    expect(String(result)).not.toContain("/dev/null");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns deterministic non-repository cwd preflight error before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "x", cwd: "/tmp" });
    assertContains(String(result), "ERROR: cwd path is not a repository/worktree root: <path>");
    assertContains(String(result), "missing .git metadata at <path>");
    expect(String(result)).not.toContain("/tmp");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects mutating cwd values that resolve to a different repository/worktree root", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
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

  it("accepts current worktree root for mutating commands", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(
      await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }),
    );
    expect(result).toBe("ADW Plans Command: create\n\nok");
    expect(getInvocations().at(-1)?.args).toContain(repoRoot);
  });

  it("parses bounded options for create/update/add-phase/update-phase in mutate wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");

    await execute({
      command: "create",
      plan_type: "feature",
      title: "x",
      options: "status=Ready priority=P1 size=L",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "create",
      "--type",
      "feature",
      "--title",
      "x",
      "--priority",
      "P1",
      "--size",
      "L",
      "--status",
      "Ready",
      "--cwd",
      repoRoot,
    ]);

    await execute({ command: "update", plan_id: "M37", options: "status=Ready priority=P1 size=L", cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update",
      "M37",
      "--status",
      "Ready",
      "--priority",
      "P1",
      "--size",
      "L",
      "--cwd",
      repoRoot,
    ]);

    await execute({ command: "add-phase", plan_id: "M37", title: "Core impl", options: "after=M37-P1 size=M", cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "add-phase",
      "M37",
      "--title",
      "Core impl",
      "--size",
      "M",
      "--after",
      "M37-P1",
      "--cwd",
      repoRoot,
    ]);

    await execute({
      command: "add-phase",
      plan_id: "M37",
      title: "Core impl",
      options: "phase-status=Blocked size=M after=M37-P1",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "add-phase",
      "M37",
      "--title",
      "Core impl",
      "--size",
      "M",
      "--status",
      "Blocked",
      "--after",
      "M37-P1",
      "--cwd",
      repoRoot,
    ]);

    await execute({ command: "update-phase", plan_id: "M37", phase_id: "M37-P2", options: "phase-status=In Progress size=M issue=42", cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update-phase",
      "M37",
      "M37-P2",
      "--status",
      "In Progress",
      "--size",
      "M",
      "--issue",
      "42",
      "--cwd",
      repoRoot,
    ]);
  });

  it("parses clear-issue-number and preserves direct patch while using options", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");

    await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      options: "clear-issue-number",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update-phase",
      "M37",
      "M37-P2",
      "--clear-issue-number",
      "--cwd",
      repoRoot,
    ]);

    await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      options: "phase-status=Blocked clear-issue-number",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update-phase",
      "M37",
      "M37-P2",
      "--status",
      "Blocked",
      "--clear-issue-number",
      "--cwd",
      repoRoot,
    ]);

    await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      options: "phase-status=Blocked size=M issue=42",
      patch: '{"actuals":"done"}',
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update-phase",
      "M37",
      "M37-P2",
      "--status",
      "Blocked",
      "--size",
      "M",
      "--issue",
      "42",
      "--patch",
      '{"actuals":"done"}',
      "--cwd",
      repoRoot,
    ]);
  });

  it("ignores whitespace-only options and merges identical direct values in mutate wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");

    await execute({ command: "create", plan_type: "feature", title: "x", options: "   ", cwd: repoRoot });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "create",
      "--type",
      "feature",
      "--title",
      "x",
      "--cwd",
      repoRoot,
    ]);

    await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      phase_status: "Blocked",
      options: "phase-status=Blocked size=M issue=42",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args.filter((arg) => arg === "--status")).toHaveLength(1);
  });

  it("rejects option conflicts and still requires cwd in mutate wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");

    const conflictResult = await execute({
      command: "update",
      plan_id: "M37",
      status: "Ready",
      options: "status=Blocked",
      cwd: repoRoot,
    });
    assertContains(String(conflictResult), "'status' cannot conflict between direct input and options string.");

    const issueConflictResult = await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      options: "issue=42 clear-issue-number",
      cwd: repoRoot,
    });
    assertContains(String(issueConflictResult), "'issue_number' and 'clear_issue_number' are mutually exclusive for update-phase.");

    const cwdResult = await execute({ command: "update", plan_id: "M37", options: "status=Ready priority=P1 size=L" });
    assertContains(String(cwdResult), "update command requires 'cwd'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves mutate command gate precedence when options are provided", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "list", options: "json" } as any);
    assertContains(String(result), "Unsupported command for adw_plans_mutate: list");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves multi-word plan status values as direct wrapper arguments", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    await execute({
      command: "create",
      plan_type: "feature",
      title: "x",
      status: "In Progress",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--status In Progress");
  });

  it("preserves multi-word phase status values as direct wrapper arguments", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      phase_status: "Not Started",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--status Not Started");
  });

  it("rejects contradictory update-phase issue-link options before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      options: "issue=42 clear-issue-number",
      cwd: repoRoot,
    });
    assertContains(
      String(result),
      "'issue_number' and 'clear_issue_number' are mutually exclusive for update-phase.",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("prefers stderr over stdout in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "stderr diagnostics", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot });
    assertContains(String(result), "stderr diagnostics");
    expect(String(result)).not.toContain("stdout diagnostics");
  });

  it("uses stdout when stderr is whitespace-only in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "   \n", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "stdout diagnostics");
  });

  it("redacts absolute paths and adds runtime hint for spawned-command failures", async () => {
    setSpawnResponse({
      stderr: "python3: can't open file /abs/path/backend.py: No such file or directory",
      exitCode: 2,
    });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
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
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
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
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(
      await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }),
    );
    expect(result).toContain("Authorization: Bearer <redacted-secret>");
    expect(result).toContain("token=<redacted-secret>");
    expect(result).toContain("<path>");
    expect(result).not.toContain("ghp_secretToken123");
    expect(result).not.toContain("abc123");
    expect(result).not.toContain("/tmp/private.txt");
  });

  it("adds cwd hint only for true cwd/path execution failures", async () => {
    setSpawnResponse({ stderr: "ERROR: cwd path does not exist: /tmp/worktree", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "hint: verify --cwd points to an existing in-repository repository/worktree root.");
  });

  it("does not add cwd hint for unrelated worktree wording", async () => {
    setSpawnResponse({ stderr: "plan worktree metadata is stale", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    expect(result).not.toContain("hint: verify --cwd points to an existing in-repository repository/worktree root.");
  });

  it("uses catch-path message fallback when stderr/stdout are empty", async () => {
    setSpawnError({ message: "ENOENT: python3 not found" });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "ENOENT: python3 not found");
  });

  it("uses stdout before message in catch-path fallback when stderr is whitespace-only", async () => {
    setSpawnError({ stderr: "  ", stdout: "stdout diagnostics", message: "message third" });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "stdout diagnostics");
    expect(result).not.toContain("message third");
  });

  it("uses unknown execution error fallback when catch-path diagnostics are empty", async () => {
    setSpawnError({});
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "Unknown execution error");
  });

  it("preserves bounded truncation marker for long diagnostics", async () => {
    setSpawnResponse({ stderr: `failure:${"x".repeat(5000)}`, exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "...[output truncated to 4000 characters; original length");
  });

  it("preserves timeout fallback text when timeout has no output", async () => {
    setSpawnResponse({ timedOut: true, exitCode: 1 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    assertContains(result, "Command timed out after 60000ms");
  });

  it("preserves success-path output", async () => {
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    expect(result).toBe("ADW Plans Command: create\n\nok");
  });

  it("accepts registry-driven plan_type strings without wrapper allowlist rejection", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    await execute({ command: "create", plan_type: "research", title: "x", cwd: repoRoot });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      `uv run adw plans create --type research --title x --cwd ${repoRoot}`,
    );
  });

  it("preserves path-like success output without split-wrapper redaction drift", async () => {
    setSpawnResponse({ stdout: "sections/root/path", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = String(await execute({ command: "create", plan_type: "feature", title: "x", cwd: repoRoot }));
    expect(result).toBe("ADW Plans Command: create\n\nsections/root/path");
  });

  it("rejects numeric zero issue option in update-phase args before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    const result = await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      options: "issue=0",
      cwd: ".",
    });
    assertContains(String(result), "issue values must be positive safe integers");
    expect(getInvocations()).toHaveLength(0);
  });

  it("update-phase patch-only payload ignores plan-level parent/priority/status in mutate wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
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

  it("update-phase forwards phase-scoped fields and phase_status only in mutate wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_mutate.ts");
    await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      status: "Ready",
      phase_status: "In Progress",
      title: "Phase Title",
      options: "size=M issue=123",
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
      "--patch",
      '{"owner":"team"}',
      "--cwd",
      repoRoot,
    ]);
    expect(args).not.toContain("Ready");
    expect(args.filter((arg) => arg === "--status")).toHaveLength(1);
  });
});
