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

  it("exposes options in schema and omits trimmed deprecated aliases", async () => {
    await loadToolExecute("../../adw_plans.ts");

    const args = getCapturedToolDefinition()?.args ?? {};
    expect(args).toHaveProperty("options");
    expect(args).toHaveProperty("status");
    expect(args).toHaveProperty("phase_status");
    expect(args).toHaveProperty("patch");
    expect(args).not.toHaveProperty("json");
    expect(args).not.toHaveProperty("priority");
    expect(args).not.toHaveProperty("size");
    expect(args).not.toHaveProperty("check");
    expect(args).not.toHaveProperty("populate");
    expect(args).not.toHaveProperty("after");
    expect(args).not.toHaveProperty("issue_number");
    expect(args).not.toHaveProperty("clear_issue_number");
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
    await execute({ command: "list", parent: "   ", options: "json" });
    const args = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(args).toContain("uv run adw plans list --json");
    expect(args).not.toContain("--parent");
  });

  it("parses list json option string in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "list", options: "json" });
    expect(getInvocations().at(-1)?.args).toEqual(["uv", "run", "adw", "plans", "list", "--json"]);
  });

  it("parses show json option string in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "show", plan_id: "E17-F1", options: "json" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "show",
      "E17-F1",
      "--json",
    ]);
  });

  it("parses schema check option string in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "schema", options: "check" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "schema",
      "--check",
    ]);
  });

  it("parses list-sections json and populate option strings in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "list-sections", plan_id: "M25", options: "json populate" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "list-sections",
      "M25",
      "--json",
      "--populate",
    ]);
  });

  it("parses add-phase after and size options in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "add-phase",
      plan_id: "M37",
      title: "Core impl",
      options: "after=M37-P1 size=M",
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
      "--after",
      "M37-P1",
      "--cwd",
      repoRoot,
    ]);
  });

  it("parses update priority and size options while preserving direct patch", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "update",
      plan_id: "M37",
      options: "priority=P1 size=L",
      patch: '{"status":"Ready"}',
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update",
      "M37",
      "--priority",
      "P1",
      "--size",
      "L",
      "--patch",
      '{"status":"Ready"}',
      "--cwd",
      repoRoot,
    ]);
  });

  it("dispatches create with bounded options string in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "create",
      plan_type: "feature",
      title: "Compat create",
      options: "priority=P1 size=L",
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
      "Compat create",
      "--priority",
      "P1",
      "--size",
      "L",
      "--cwd",
      repoRoot,
    ]);
  });

  it("parses status and phase-status options in compatibility wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "update",
      plan_id: "M37",
      options: "status=Ready priority=P1 size=L",
      cwd: repoRoot,
    });
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

    await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      options: "phase-status=In Progress size=M issue=42",
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
      "In Progress",
      "--size",
      "M",
      "--issue",
      "42",
      "--cwd",
      repoRoot,
    ]);
  });

  it("parses update-phase issue-link options while preserving direct phase status", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      phase_status: "In Progress",
      options: "size=M issue=42",
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
      "In Progress",
      "--size",
      "M",
      "--issue",
      "42",
      "--cwd",
      repoRoot,
    ]);
  });

  it("dispatches clear-issue-number through options for update-phase", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
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

  it("preserves multi-word plan status values in compatibility wrapper invocations", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "create",
      plan_type: "feature",
      title: "x",
      status: "In Progress",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--status In Progress");
  });

  it("preserves multi-word phase status values in compatibility wrapper invocations", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({
      command: "update-phase",
      plan_id: "E1-F1",
      phase_id: "E1-F1-P1",
      phase_status: "Not Started",
      cwd: repoRoot,
    });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--status Not Started");
  });

  it("rejects numeric zero issue option in update-phase args before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
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

  it("rejects contradictory update-phase issue link arguments before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
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

  it("rejects unknown options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list", options: "bogus" });
    assertContains(String(result), "Invalid options token 'bogus' for 'list': token is not allowed for this command");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing-value options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "add-phase", plan_id: "M37", title: "x", options: "after=", cwd: repoRoot });
    assertContains(String(result), "Invalid options token 'after=' for 'add-phase': token value must not be empty");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects extra-equals options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "schema", options: "check=yes=no" });
    assertContains(String(result), "tokens must contain at most one '=' separator");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects conflicting duplicate options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "update", plan_id: "M37", options: "size=M size=L", cwd: repoRoot });
    assertContains(String(result), "conflicting duplicate 'size=L'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects invalid issue values parsed from options before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "update-phase", plan_id: "M37", phase_id: "M37-P2", options: "issue=0", cwd: repoRoot });
    assertContains(String(result), "issue values must be positive safe integers");
    expect(getInvocations()).toHaveLength(0);
  });

  it("treats whitespace-only options as omitted without masking required-arg failures", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "show", plan_id: "   ", options: "   \n\t  " });
    assertContains(String(result), "show command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects direct and options conflicts for retained direct fields before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "update", plan_id: "M37", status: "Ready", options: "status=Blocked", cwd: repoRoot } as any);
    assertContains(String(result), "'status' cannot conflict between direct input and options string.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("accepts matching direct and options values for retained direct fields", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    await execute({ command: "update", plan_id: "M37", status: "Ready", options: "status=Ready", cwd: repoRoot } as any);
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "adw",
      "plans",
      "update",
      "M37",
      "--status",
      "Ready",
      "--cwd",
      repoRoot,
    ]);
  });

  it("ignores whitespace-only patch for commands that do not use patch normalization", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = String(
      await execute({ command: "create", plan_type: "feature", title: "x", patch: "   ", cwd: repoRoot } as any),
    );
    expect(result).toBe("ADW Plans Command: create\n\nok");
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
  });

  it("rejects oversized patch payloads before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const oversizedPatch = "x".repeat(65_537);
    const result = await execute({
      command: "update-phase",
      plan_id: "M37",
      phase_id: "M37-P2",
      patch: oversizedPatch,
      cwd: repoRoot,
    });
    assertContains(String(result), "'patch' exceeds maximum size (65536 bytes UTF-8)");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects cwd values that start with a dash before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({ command: "list", cwd: "-bad-path" });
    assertContains(String(result), "'cwd' must not start with '-' to avoid CLI option confusion.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects option tokens for scaffold-sections empty allowlist before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans.ts");
    const result = await execute({
      command: "scaffold-sections",
      plan_id: "M37",
      plan_type: "maintenance",
      options: "json",
      cwd: repoRoot,
    });
    assertContains(
      String(result),
      "Invalid options token 'json' for 'scaffold-sections': token is not allowed for this command",
    );
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
