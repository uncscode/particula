import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import path from "node:path";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { getCapturedToolDefinition, loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

const FIXTURE_DIR = path.join(import.meta.dir, "fixtures/search_scope");
const FIXTURE_FILE = path.join(FIXTURE_DIR, "alpha.ts");
const NESTED_DIR = path.join(FIXTURE_DIR, "nested");
const TRASH_DIR = path.join(FIXTURE_DIR, ".trash");

describe("search_content wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires contentPattern", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({});
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "contentPattern");
  });

  it("omits advanced-only direct fields from the public schema", async () => {
    await loadToolExecute("../../search_content.ts");
    const definition = getCapturedToolDefinition();
    expect(definition?.args).not.toHaveProperty("beforeContext");
    expect(definition?.args).not.toHaveProperty("filesWithMatches");
    expect(definition?.args).not.toHaveProperty("unrestricted");
  });

  it("accepts bounded simple-search options via options", async () => {
    setSpawnResponse({ stdout: "a:1:x\n", exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({
      contentPattern: "x",
      options: "pattern=**/*.ts file-type=ts glob-case-insensitive max-matches-per-file=3",
    });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toContain("--glob");
    expect(args[args.indexOf("--glob") + 1]).toBe("**/*.ts");
    expect(args).toContain("-t");
    expect(args[args.indexOf("-t") + 1]).toBe("ts");
    expect(args).toContain("--glob-case-insensitive");
    expect(args).toContain("--max-count");
    expect(args[args.indexOf("--max-count") + 1]).toBe("3");
  });

  it("rejects advanced-only controls through options", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "x", options: "before-context=1" });
    assertContains(String(result), "Invalid options token 'before-context=1'");
  });

  it("rejects invalid max-matches token values", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "x", options: "max-matches-per-file=abc" });
    assertContains(String(result), "max-matches-per-file must be a non-negative integer");
  });

  it("rejects unknown options tokens", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "x", options: "unknown-token=1" });
    assertContains(String(result), "Invalid options token 'unknown-token=1'");
  });

  it("assembles rg content search", async () => {
    setSpawnResponse({ stdout: "a:1:x\n", exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({ contentPattern: "x", path: "." });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("rg -n -e x");
  });

  it("guards option-like contentPattern values from rg flag parsing", async () => {
    setSpawnResponse({ stdout: "a:1:--help\n", exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({ contentPattern: "--help" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.slice(0, 4)).toEqual(["rg", "-n", "-e", "--help"]);
    expect(args).toContain("--");
  });

  it("prefers stderr over stdout on non-zero exit", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ contentPattern: "x" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("routes file paths to a single-file ripgrep operand", async () => {
    setSpawnResponse({ stdout: `${FIXTURE_FILE}:1:scoped-search-needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({ contentPattern: "scoped-search-needle", path: FIXTURE_FILE });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(FIXTURE_FILE);
    expect(args).toContain("--");
  });

  it("routes directory paths to the requested subtree only", async () => {
    setSpawnResponse({ stdout: `${path.join(NESTED_DIR, "beta.ts")}:1:nested-search-needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({ contentPattern: "nested-search-needle", path: NESTED_DIR });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(NESTED_DIR);
  });

  it("returns a deterministic error for missing scoped paths", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const missingPath = path.join(FIXTURE_DIR, "missing.ts");
    const result = await execute({ contentPattern: "x", path: missingPath });
    expect(String(result)).toBe(
      `ERROR: Search path does not exist: ${missingPath}\n\nHint: Verify the path is correct.`,
    );
  });

  it("rejects out-of-repo paths without widening scope", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "x", path: "/etc/passwd" });
    assertContains(String(result), "outside the repository");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects nonexistent absolute paths outside the repo before existence checks", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "x", path: "/definitely-missing-outside-repo/nope.ts" });
    assertContains(String(result), "outside the repository");
    expect(String(result)).not.toContain("does not exist");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves .trash subtree scope", async () => {
    setSpawnResponse({ stdout: `${path.join(TRASH_DIR, "trashed.ts")}:1:trash-search-needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    await execute({ contentPattern: "trash-search-needle", path: TRASH_DIR });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(TRASH_DIR);
  });

  it("keeps valid scoped no-match results non-error and scoped", async () => {
    setSpawnResponse({ stdout: "", exitCode: 1 });
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "absent-needle", path: FIXTURE_FILE });
    expect(String(result)).toBe(
      `No matches found for contentPattern 'absent-needle' in '${FIXTURE_FILE}'.`,
    );
  });

  it("keeps regex parse guidance accurate for valid file-scoped paths", async () => {
    setSpawnResponse({ exitCode: 2, stderr: "regex parse error: bad regex" });
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "(", path: FIXTURE_FILE });
    assertContains(String(result), "Invalid contentPattern regex");
    expect(String(result)).not.toContain("Provide a directory path");
  });

  it("supports compact-output for content results", async () => {
    setSpawnResponse({ stdout: `${FIXTURE_FILE}:1:needle\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "needle", path: FIXTURE_FILE, options: "compact-output" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toContain("--with-filename");
    expect(String(result)).toBe("alpha.ts:1:needle");
  });

  it("clips large stdout during collection and returns a deterministic warning", async () => {
    setSpawnResponse({
      stdoutChunks: [
        `${FIXTURE_FILE}:1:first\n`,
        `${FIXTURE_FILE}:2:second\n`,
        `${FIXTURE_FILE}:3:third\n`,
      ],
      exitCode: 0,
    });
    const execute = await loadToolExecute("../../search_content.ts");
    const result = await execute({ contentPattern: "needle", options: "max-results=2" });
    expect(String(result)).toContain(`${FIXTURE_FILE}:1:first\n${FIXTURE_FILE}:2:second`);
    expect(String(result)).toContain("Results truncated to 2 lines (at least 3 total found)");
    expect(String(result)).toContain("Ripgrep stdout was clipped for safety");
  });

  it("rejects symlink escapes after stat succeeds without invoking rg", async () => {
    const execute = await loadToolExecute("../../search_content.ts");
    const escapePath = path.join(FIXTURE_DIR, "escape.ts");
    const originalStat = fs.promises.stat;
    const originalRealpath = fs.promises.realpath;
    fs.promises.stat = (async (targetPath: fs.PathLike) => {
      if (targetPath === escapePath) {
        return { isDirectory: () => false, isFile: () => true } as fs.Stats;
      }
      return originalStat(targetPath);
    }) as typeof fs.promises.stat;
    fs.promises.realpath = (async (targetPath: fs.PathLike) => {
      if (targetPath === escapePath) return "/tmp/outside-repo.ts" as Awaited<ReturnType<typeof originalRealpath>>;
      return originalRealpath(targetPath);
    }) as typeof fs.promises.realpath;

    try {
      const result = await execute({ contentPattern: "needle", path: escapePath });
      assertContains(String(result), "outside the repository");
      expect(getInvocations()).toHaveLength(0);
    } finally {
      fs.promises.stat = originalStat;
      fs.promises.realpath = originalRealpath;
    }
  });
});
