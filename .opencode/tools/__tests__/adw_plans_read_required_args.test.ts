import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
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
  const tempRoot = mkdtempSync(`${tmpdir()}/adw-plans-read-`);
  writeFileSync(resolve(tempRoot, ".git"), "gitdir: /tmp/fake\n");
  return tempRoot;
}

describe("adw_plans_read required-arg preflight", () => {
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
    await loadToolExecute("../../adw_plans_read.ts");
    const args = getCapturedToolDefinition()?.args ?? {};
    expect(args).toHaveProperty("options");
    expect(args).not.toHaveProperty("status");
    expect(args).not.toHaveProperty("json");
    expect(args).not.toHaveProperty("check");
    expect(args).not.toHaveProperty("populate");
  });

  it("keeps read-wrapper command schema source scoped to read-only commands", async () => {
    const source = await Bun.file(resolve(import.meta.dir, "../adw_plans_read.ts")).text();
    expect(source).toContain('tool.schema.enum(["list", "show", "validate", "schema", "list-sections"])');
    expect(source).not.toContain('tool.schema.enum(["list", "show", "validate", "schema", "list-sections", "create"');
  });

  it("rejects whitespace-only required plan_id for list-sections before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list-sections", plan_id: "   " });
    assertContains(String(result), "list-sections command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects wrong-type required plan_id for show before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "show", plan_id: 123 as any });
    assertContains(String(result), "show command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing required plan_id for show before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "show" });
    assertContains(String(result), "show command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null required plan_id for list-sections before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list-sections", plan_id: null as any });
    assertContains(String(result), "list-sections command requires 'plan_id'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects mutating commands via read-only gate before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "create", plan_type: "feature", title: "x" } as any);
    assertContains(String(result), "Unsupported command for adw_plans_read: create");
    expect(getInvocations()).toHaveLength(0);
  });

  it("redacts cwd path in preflight diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list", cwd: "/definitely/missing/path" });
    assertContains(String(result), "cwd path does not exist");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns deterministic cwd not-a-directory preflight error before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list", cwd: "/dev/null" });
    assertContains(String(result), "ERROR: cwd path is not a directory: <path>");
    expect(String(result)).not.toContain("/dev/null");
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns deterministic non-repository cwd preflight error before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list", cwd: "/tmp" });
    assertContains(String(result), "ERROR: cwd path is not a repository/worktree root: <path>");
    assertContains(String(result), "missing .git metadata at <path>");
    expect(String(result)).not.toContain("/tmp");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects read cwd values that resolve to a different repository/worktree root", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const otherRepoRoot = createOtherRepositoryRoot();
    const result = await execute({ command: "list", cwd: otherRepoRoot });
    assertContains(String(result), "ERROR: cwd path resolves outside repository root: <path> (canonical: <path>)");
    expect(getInvocations()).toHaveLength(0);
    rmSync(otherRepoRoot, { recursive: true, force: true });
  });

  it("accepts current worktree root for read commands", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list", cwd: repoRoot }));
    expect(result).toBe("ADW Plans Command: list\n\nok");
    expect(getInvocations().at(-1)?.args).toContain(repoRoot);
  });

  it("forwards canonical cwd when read wrapper receives a symlink alias", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const tempRoot = resolve(repoRoot, "adforge_local/opencode/tmp");
    mkdirSync(tempRoot, { recursive: true });
    const aliasPath = resolve(tempRoot, "adw-plans-read-alias");
    rmSync(aliasPath, { recursive: true, force: true });
    symlinkSync(repoRoot, aliasPath, "dir");

    try {
      await execute({ command: "list", cwd: aliasPath });
      expect(getInvocations().at(-1)?.args).toContain(repoRoot);
      expect(getInvocations().at(-1)?.args).not.toContain(aliasPath);
    } finally {
      rmSync(aliasPath, { recursive: true, force: true });
    }
  });

  it("parses multi-word plan status values through options", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    await execute({ command: "list", options: "status=In Progress" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("--status In Progress");
  });

  it("rejects direct status on the split wrapper before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list", status: "Ready" } as any);
    assertContains(String(result), "'status' is not accepted as a direct field in adw_plans_read");
    expect(getInvocations()).toHaveLength(0);
  });

  it("treats inert direct status aliases as omitted in the split wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");

    await execute({ command: "list", status: "   " } as any);
    expect(getInvocations().at(-1)?.args).toEqual(["uv", "run", "--active", "adw", "plans", "list"]);

    await execute({ command: "list", status: false as any });
    expect(getInvocations().at(-1)?.args).toEqual(["uv", "run", "--active", "adw", "plans", "list"]);

    await execute({ command: "list", status: 0 as any });
    expect(getInvocations().at(-1)?.args).toEqual(["uv", "run", "--active", "adw", "plans", "list"]);
  });

  it("parses list/show/schema/list-sections option strings in read wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");

    await execute({ command: "list", options: "status=Ready json" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "plans",
      "list",
      "--status",
      "Ready",
      "--json",
    ]);

    await execute({ command: "show", plan_id: "M37", options: "json" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "plans",
      "show",
      "M37",
      "--json",
    ]);

    await execute({ command: "schema", options: "check" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "plans",
      "schema",
      "--check",
    ]);

    await execute({ command: "list-sections", plan_id: "M37", options: "json populate" });
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "plans",
      "list-sections",
      "M37",
      "--json",
      "--populate",
    ]);
  });

  it("ignores whitespace-only options in read wrapper", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");

    await execute({ command: "list", options: "   \n\t  " });
    expect(getInvocations().at(-1)?.args).toEqual(["uv", "run", "--active", "adw", "plans", "list"]);
  });

  it("rejects invalid option tokens in read wrapper before spawn", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list", options: "check" } as any);
    assertContains(String(result), "token is not allowed for this command");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves read-only command gate precedence when options are provided", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "update", options: "status=Ready" } as any);
    assertContains(String(result), "Unsupported command for adw_plans_read: update");
    expect(getInvocations()).toHaveLength(0);
  });

  it("prefers stderr over stdout in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "stderr diagnostics", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = await execute({ command: "list" });
    assertContains(String(result), "stderr diagnostics");
    expect(String(result)).not.toContain("stdout diagnostics");
  });

  it("uses stdout when stderr is whitespace-only in command failure envelope", async () => {
    setSpawnResponse({ stdout: "stdout diagnostics", stderr: "   \n", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "stdout diagnostics");
  });

  it("redacts absolute paths and adds runtime hint for spawned-command failures", async () => {
    setSpawnResponse({
      stderr: "python3: can't open file /abs/path/backend.py: No such file or directory",
      exitCode: 2,
    });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
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
    const execute = await loadToolExecute("../../adw_plans_read.ts");
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
    const execute = await loadToolExecute("../../adw_plans_read.ts");
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
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "hint: verify --cwd points to an existing in-repository repository/worktree root.");
  });

  it("does not add cwd hint for unrelated worktree wording", async () => {
    setSpawnResponse({ stderr: "plan worktree metadata is stale", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).not.toContain("hint: verify --cwd points to an existing in-repository repository/worktree root.");
  });

  it("uses catch-path message fallback when stderr/stdout are empty", async () => {
    setSpawnError({ message: "ENOENT: python3 not found" });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "ENOENT: python3 not found");
  });

  it("uses stdout before message in catch-path fallback when stderr is whitespace-only", async () => {
    setSpawnError({ stderr: "  ", stdout: "stdout diagnostics", message: "message third" });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "stdout diagnostics");
    expect(result).not.toContain("message third");
  });

  it("uses unknown execution error fallback when catch-path diagnostics are empty", async () => {
    setSpawnError({});
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "Unknown execution error");
  });

  it("preserves bounded truncation marker for long diagnostics", async () => {
    setSpawnResponse({ stderr: `failure:${"x".repeat(5000)}`, exitCode: 2 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "...[output truncated to 4000 characters; original length");
  });

  it("preserves timeout fallback text when timeout has no output", async () => {
    setSpawnResponse({ timedOut: true, exitCode: 1 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    assertContains(result, "Command timed out after 60000ms");
  });

  it("preserves success-path output", async () => {
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).toBe("ADW Plans Command: list\n\nok");
  });

  it("accepts registry-driven plan_type strings without wrapper allowlist rejection", async () => {
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    await execute({ command: "list", plan_type: "research" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw plans list --type research");
  });

  it("preserves path-like success output without split-wrapper redaction drift", async () => {
    setSpawnResponse({ stdout: "sections/root/path", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_plans_read.ts");
    const result = String(await execute({ command: "list" }));
    expect(result).toBe("ADW Plans Command: list\n\nsections/root/path");
  });
});
