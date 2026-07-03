import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import path from "node:path";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { getCapturedToolDefinition, loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const FIXTURE_DIR = path.join(import.meta.dir, "fixtures/search_scope");
const FIXTURE_FILE = path.join(FIXTURE_DIR, "alpha.ts");
const NESTED_DIR = path.join(FIXTURE_DIR, "nested");
const TRASH_DIR = path.join(FIXTURE_DIR, ".trash");

describe("ripgrep_advanced wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires contentPattern", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({});
    assertContains(String(result), "contentPattern");
  });

  it("keeps advanced controls on bounded options rather than direct schema fields", async () => {
    await loadToolExecute("../../ripgrep_advanced.ts");
    const definition = getCapturedToolDefinition();
    expect(definition?.args).not.toHaveProperty("beforeContext");
    expect(definition?.args).not.toHaveProperty("filesWithMatches");
    expect(definition?.args).not.toHaveProperty("unrestricted");
    expect(definition?.args).toHaveProperty("options");
  });

  it("validates beforeContext", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "x", options: "before-context=-1" });
    assertContains(String(result), "before-context must be a non-negative integer");
  });

  it("parses advanced options and preserves directional context precedence", async () => {
    setSpawnResponse({ stdout: "a:1:x\n", exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({
      contentPattern: "x",
      options:
        "pattern=**/*.ts file-type=ts context-lines=3 before-context=2 after-context=1 files-with-matches unrestricted=2 include-hidden",
    });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toContain("--glob");
    expect(args[args.indexOf("--glob") + 1]).toBe("**/*.ts");
    expect(args).toContain("-t");
    expect(args[args.indexOf("-t") + 1]).toBe("ts");
    expect(args).toContain("-l");
    expect(args).toContain("-B");
    expect(args[args.indexOf("-B") + 1]).toBe("2");
    expect(args).toContain("-A");
    expect(args[args.indexOf("-A") + 1]).toBe("1");
    expect(args).not.toContain("-C");
    expect(args).toContain("-uu");
    expect(args).not.toContain("--hidden");
  });

  it("rejects conflicting file-mode tokens", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({
      contentPattern: "x",
      options: "files-with-matches files-without-matches",
    });
    assertContains(String(result), "cannot both be true");
  });

  it("rejects unknown options tokens", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "x", options: "mystery=1" });
    assertContains(String(result), "Invalid options token 'mystery=1'");
  });

  it("rejects unrestricted values above the supported range", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "x", options: "unrestricted=4" });
    assertContains(String(result), "Invalid unrestricted value");
  });

  it("assembles content command", async () => {
    setSpawnResponse({ stdout: "a:1:x\n", exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({ contentPattern: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("rg -n -e x");
  });

  it("guards option-like contentPattern values from rg flag parsing", async () => {
    setSpawnResponse({ stdout: "a:1:--help\n", exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({ contentPattern: "--help" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.slice(0, 4)).toEqual(["rg", "-n", "-e", "--help"]);
    expect(args).toContain("--");
  });

  it("prefers stderr over stdout on non-zero exit", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ contentPattern: "x" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("routes file targets with advanced controls intact", async () => {
    setSpawnResponse({ stdout: `${FIXTURE_FILE}:1:scoped-search-needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({
      contentPattern: "scoped-search-needle",
      path: FIXTURE_FILE,
      options: "before-context=2 after-context=1 files-with-matches",
    });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(FIXTURE_FILE);
    expect(args).toContain("-B");
    expect(args).toContain("-A");
    expect(args).toContain("-l");
  });

  it("routes directory targets to the requested subtree only", async () => {
    setSpawnResponse({ stdout: `${path.join(NESTED_DIR, "beta.ts")}:1:nested-search-needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({ contentPattern: "nested-search-needle", path: NESTED_DIR });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(NESTED_DIR);
  });

  it("returns a deterministic error for missing scoped paths", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const missingPath = path.join(FIXTURE_DIR, "missing.ts");
    const result = await execute({ contentPattern: "x", path: missingPath });
    expect(String(result)).toBe(
      `ERROR: Search path does not exist: ${missingPath}\n\nHint: Verify the path is correct.`,
    );
  });

  it("rejects out-of-repo paths without widening scope", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "x", path: "/etc/passwd" });
    assertContains(String(result), "outside the repository");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves .trash subtree scope", async () => {
    setSpawnResponse({ stdout: `${path.join(TRASH_DIR, "trashed.ts")}:1:trash-search-needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    await execute({ contentPattern: "trash-search-needle", path: TRASH_DIR });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(TRASH_DIR);
  });

  it("keeps valid scoped no-match results non-error and scoped", async () => {
    setSpawnResponse({ stdout: "", exitCode: 1 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "absent-needle", path: NESTED_DIR });
    expect(String(result)).toBe(
      `No matches found for contentPattern 'absent-needle' in '${NESTED_DIR}'.`,
    );
  });

  it("supports compact-output for advanced content results", async () => {
    setSpawnResponse({ stdout: `${FIXTURE_FILE}:1:needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const result = await execute({ contentPattern: "needle", path: FIXTURE_FILE, options: "compact-output" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toContain("--with-filename");
    expect(String(result)).toBe("alpha.ts:1:needle");
  });

  it("rejects unsupported inode types without invoking rg", async () => {
    const execute = await loadToolExecute("../../ripgrep_advanced.ts");
    const fifoPath = path.join(FIXTURE_DIR, "fifo");
    const originalStat = fs.promises.stat;
    fs.promises.stat = (async (targetPath: fs.PathLike) => {
      if (targetPath === fifoPath) {
        return { isDirectory: () => false, isFile: () => false } as fs.Stats;
      }
      return originalStat(targetPath);
    }) as typeof fs.promises.stat;

    try {
      const result = await execute({ contentPattern: "needle", path: fifoPath });
      assertContains(String(result), "Unsupported search path type");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fs.promises.stat = originalStat;
    }
  });
});
