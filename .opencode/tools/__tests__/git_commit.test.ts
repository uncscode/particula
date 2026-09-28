import { afterEach, beforeEach, describe, expect, it, spyOn } from "bun:test";
import * as fs from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import {
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

describe("git_commit wrapper", () => {
  const logDirectories = new Set<string>();
  const spies: Array<{ mockRestore: () => void }> = [];
  const logPath = (result: string): string => {
    const path = result.match(/^full_output_path: (.+)$/m)?.[1];
    expect(path).toBeDefined();
    logDirectories.add(dirname(path!));
    expect(isAbsolute(path!)).toBe(true);
    return path!;
  };
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(async () => {
    for (const spy of spies.splice(0)) spy.mockRestore();
    for (const directory of logDirectories) {
      await fs.rm(directory, { recursive: true, force: true });
    }
    logDirectories.clear();
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires non-empty summary", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "   " });
    assertContains(String(result), "requires non-empty 'summary'");
  });

  it("documents the narrow wrapper boundary distinctly from native protected git", async () => {
    await loadToolExecute("../../git_commit.ts");

    const description = getCapturedToolDefinition().description ?? "";
    expect(description).toContain("narrow OpenCode adw git commit path");
    expect(description).toContain("distinct from the native TUI/runtime protected-git");
    expect(description).toContain("does not imply arbitrary git verbs, shell execution, auto-push, or");
  });

  it("validates max_retries", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "msg", max_retries: 99 });
    assertContains(String(result), "between 0 and 10");
  });

  it("assembles commit with summary", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({ summary: "test commit" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git commit --summary test commit",
    );
  });

  it("assembles description adw_id worktree_path and stage_all with default retries", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({
      summary: "test commit",
      description: "body",
      adw_id: "dac13a15",
      worktree_path: "./trees/abc",
      stage_all: true,
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git commit --summary test commit --description body --adw-id dac13a15 --worktree-path ./trees/abc --stage-all --max-retries 3",
    );
  });

  it("passes through no_verify and zero retries", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({ summary: "test commit", no_verify: true, max_retries: 0 });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw git commit --summary test commit --no-verify --max-retries 0",
    );
  });

  it("rejects non-boolean no_verify values", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "msg", no_verify: "true" });

    assertContains(String(result), "'no_verify' must be a boolean");
  });

  it("rejects option-like worktree paths before spawning", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = await execute({ summary: "msg", worktree_path: "--repo=/tmp/x" });

    assertContains(String(result), "'worktree_path' cannot start with '-'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("does not emit flags for false optional booleans", async () => {
    const execute = await loadToolExecute("../../git_commit.ts");
    await execute({ summary: "test commit", no_verify: false, stage_all: false });

    const command = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(command).toContain("--max-retries 3");
    expect(command).not.toContain("--no-verify");
    expect(command).not.toContain("--stage-all");
  });

  it("passes through committed status and sha beneath wrapper prefix", async () => {
    setDollarText("✅ Success\nstatus=committed\nsha=abc123def456\ncreated");
    const execute = await loadToolExecute("../../git_commit.ts");

    const result = await execute({ summary: "test commit" });

    expect(String(result)).toContain(
      "Git Commit Command\n\n✅ Success\nstatus=committed\nsha=abc123def456\ncreated",
    );
  });

  it("passes through no-op status without sha", async () => {
    setDollarText("✅ Success\nstatus=no_op\nNo changes to commit");
    const execute = await loadToolExecute("../../git_commit.ts");

    const result = await execute({ summary: "test commit" });

    expect(String(result)).toContain("status=no_op");
    expect(String(result)).not.toContain("sha=");
  });

  it("preserves deterministic failure envelope", async () => {
    setDollarError({ stderr: "commit failed", message: "spawn failed" });
    const execute = await loadToolExecute("../../git_commit.ts");

    const result = await execute({ summary: "test commit" });

    assertContains(String(result), "ERROR: Failed to execute 'adw git commit'");
    assertContains(String(result), "stderr: commit failed");
  });

  it("keeps threshold-length responses unchanged without creating logs", async () => {
    const mkdir = spyOn(fs, "mkdir");
    spies.push(mkdir);
    const execute = await loadToolExecute("../../git_commit.ts");
    const output = "a".repeat(8000);
    setDollarText(output);
    expect(await execute({ summary: "msg" })).toBe(`Git Commit Command\n\n${output}`);
    setDollarError({ stderr: "e".repeat(500), exitCode: 1 });
    expect(await execute({ summary: "msg" })).toBe(
      "ERROR: Failed to execute 'adw git commit'\ncommand: commit\nexit_code: 1\n" +
      `stderr: ${"e".repeat(500)}\n` +
      "hint: Inspect stderr/stdout details and rerun with corrected inputs or repository state.",
    );
    expect(mkdir).not.toHaveBeenCalled();
  });

  it("preserves late codespell findings and raw streams in private unique logs", async () => {
    const stderr = "warning\n\tdeprecated\r\n".repeat(50) + "codespell: teh ==> the\n";
    const stdout = "pre-commit\n\tchecking files\r\n";
    const message = "hook failed\n\tdetails";
    setDollarError({ stderr, stdout, message, exitCode: 7 });
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = String(await execute({ summary: "msg", worktree_path: "./trees/abc" }));
    const path = logPath(result);
    expect(getInvocations()).toHaveLength(1);
    expect(result).toContain("exit_code: 7\nworktree_path: ./trees/abc");
    expect(result).toContain("Commit was rejected by git hooks");
    expect(result).not.toContain("codespell: teh");
    expect(await fs.readFile(path, "utf8")).toBe(
      `exit_code: 7\n\nstderr:\n${stderr}\n\nstdout:\n${stdout}\n\nmessage:\n${message}`,
    );
    expect((await fs.stat(path)).mode & 0o777).toBe(0o600);
    expect((await fs.stat(dirname(path))).mode & 0o777).toBe(0o700);
    const secondPath = logPath(String(await execute({ summary: "msg" })));
    expect(secondPath).not.toBe(path);
  });

  for (const stream of ["stderr", "stdout", "message"]) {
    it(`detects long raw ${stream} before whitespace normalization`, async () => {
      const raw = "\n\t".repeat(251);
      setDollarError({ stderr: "short", [stream]: raw, code: 2 });
      const execute = await loadToolExecute("../../git_commit.ts");
      const result = String(await execute({ summary: "msg" }));
      expect(await fs.readFile(logPath(result), "utf8")).toContain(`${stream}:\n${raw}`);
      expect(result).toContain("exit_code: 2");
      expect(getInvocations()).toHaveLength(1);
    });
  }

  it("saves long success exactly with a bounded head and tail preview", async () => {
    const output = "status=committed\n" + "details\n\t".repeat(1000) + "sha=abc123\n";
    setDollarText(output);
    const execute = await loadToolExecute("../../git_commit.ts");
    const result = String(await execute({ summary: "msg" }));
    expect(result.startsWith(`Git Commit Command\n\n${output.slice(0, 4000)}\n... [truncated]\n${output.slice(-2000)}\nfull_output_path:`)).toBe(true);
    expect(result.length).toBeLessThan(7000);
    expect(await fs.readFile(logPath(result), "utf8")).toBe(output);
    expect(getInvocations()).toHaveLength(1);
  });

  it("cleans only old direct matching directories before creating the new log", async () => {
    const root = resolve(import.meta.dir, "../../../adforge_local/opencode/tmp");
    await fs.mkdir(root, { recursive: true });
    const fixture = await fs.mkdtemp(join(root, "cleanup-test-"));
    logDirectories.add(fixture);
    const old = await fs.mkdtemp(join(root, "git-commit-old-test-"));
    const recent = await fs.mkdtemp(join(root, "git-commit-recent-test-"));
    logDirectories.add(old);
    logDirectories.add(recent);
    const nested = join(fixture, "git-commit-nested");
    await fs.mkdir(nested);
    const link = join(root, `git-commit-link-${fixture.split("/").at(-1)}`);
    const file = join(root, `git-commit-file-${fixture.split("/").at(-1)}`);
    logDirectories.add(link);
    logDirectories.add(file);
    await fs.symlink(fixture, link);
    await fs.writeFile(file, "keep");
    await fs.writeFile(join(old, "output.log"), "old output");
    const past = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000);
    for (const path of [old, nested, fixture, file]) await fs.utimes(path, past, past);
    // Limit cleanup enumeration to this test's entries, protecting existing logs.
    const entries = (await fs.readdir(root, { withFileTypes: true })).filter((entry) =>
      [old, recent, fixture, link, file].includes(join(root, entry.name)),
    );
    const readdir = spyOn(fs, "readdir").mockResolvedValue(entries as any);
    spies.push(readdir);
    const originalMkdtemp = fs.mkdtemp;
    const mkdtemp = spyOn(fs, "mkdtemp").mockImplementation(async (prefix: any, options: any) => {
      expect(await fs.stat(old).catch(() => null)).toBeNull();
      return originalMkdtemp(prefix, options);
    });
    spies.push(mkdtemp);
    setDollarText("s".repeat(8001));
    const execute = await loadToolExecute("../../git_commit.ts");
    logPath(String(await execute({ summary: "msg" })));
    expect(await fs.stat(old).catch(() => null)).toBeNull();
    expect((await fs.stat(recent)).isDirectory()).toBe(true);
    expect((await fs.stat(nested)).isDirectory()).toBe(true);
    expect((await fs.lstat(link)).isSymbolicLink()).toBe(true);
    expect(await fs.readFile(file, "utf8")).toBe("keep");
    expect(getInvocations()).toHaveLength(1);
  });

  for (const operation of ["readdir", "lstat", "rm"] as const) {
    it(`saves the new log despite cleanup ${operation} failure`, async () => {
      const root = resolve(import.meta.dir, "../../../adforge_local/opencode/tmp");
      await fs.mkdir(root, { recursive: true });
      const old = await fs.mkdtemp(join(root, "git-commit-cleanup-failure-"));
      logDirectories.add(old);
      const past = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000);
      await fs.utimes(old, past, past);
      const entries = (await fs.readdir(root, { withFileTypes: true })).filter((entry) => join(root, entry.name) === old);
      if (operation !== "readdir") spies.push(spyOn(fs, "readdir").mockResolvedValue(entries as any));
      spies.push(spyOn(fs, operation).mockRejectedValue(new Error("cleanup unavailable")));
      const output = "s".repeat(8001);
      setDollarText(output);
      const execute = await loadToolExecute("../../git_commit.ts");
      const result = String(await execute({ summary: "msg" }));
      expect(await fs.readFile(logPath(result), "utf8")).toBe(output);
      expect(result).not.toContain("output_log_error");
      expect(result.startsWith("Git Commit Command\n")).toBe(true);
      expect(getInvocations()).toHaveLength(1);
    });
  }

  for (const operation of ["mkdir", "mkdtemp", "writeFile"] as const) {
    for (const success of [true, false]) {
      it(`retains the result and full inline output when ${operation} fails (${success ? "success" : "failure"})`, async () => {
        const spy = spyOn(fs, operation).mockImplementation(async (...args: any[]) => {
          if (operation === "writeFile") logDirectories.add(dirname(String(args[0])));
          throw new Error("log unavailable");
        });
        spies.push(spy);
        const output = "raw\n\toutput\r\n".repeat(1000);
        if (success) setDollarText(output);
        else setDollarError({ stderr: output, stdout: "original\n\tstdout", message: "fallback", exitCode: 9 });
        const execute = await loadToolExecute("../../git_commit.ts");
        const result = String(await execute({ summary: "msg" }));
        expect(result).toContain("output_log_error: log unavailable");
        expect(result).not.toContain("full_output_path:");
        expect(result).toContain(output);
        expect(result.startsWith(success ? "Git Commit Command\n" : "ERROR: Failed to execute 'adw git commit'")).toBe(true);
        if (!success) {
          expect(result).toContain("exit_code: 9");
          expect(result).toContain("stdout:\noriginal\n\tstdout\n\nmessage:\nfallback");
        }
        expect(getInvocations()).toHaveLength(1);
      });
    }
  }

});
