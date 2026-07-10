import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import path from "node:path";

import { assertContains } from "./helpers/assert-error-envelope";
import { COMPACT_SCHEMA_FIELD_FIXTURES } from "./fixtures/wrapper_contract_fixtures";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import {
  assertCountedAndExemptFields,
  assertPublicSchemaOmitsKeys,
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";
import { inspectWrapperSourceText } from "../tooling/wrapper_schema_inventory";

const TOOL_CWD = path.resolve(import.meta.dir, "../..");
const REAL_REPO_ROOT = path.resolve(import.meta.dir, "../../..");
const FIXTURE_DIR = path.join(import.meta.dir, "fixtures/search_scope");
const FIXTURE_FILE = path.join(FIXTURE_DIR, "alpha.ts");
const NESTED_DIR = path.join(FIXTURE_DIR, "nested");
const TRASH_DIR = path.join(FIXTURE_DIR, ".trash");

describe("find_files wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("omits unsupported direct fields from the public schema", async () => {
    await loadToolExecute("../../find_files.ts");
    const definition = getCapturedToolDefinition();
    const wrapperPath = path.join(REAL_REPO_ROOT, ".opencode/tools/find_files.ts");
    const row = inspectWrapperSourceText(
      wrapperPath,
      await Bun.file(wrapperPath).text(),
      {
        repoRoot: REAL_REPO_ROOT,
        docsByBasename: new Map([["find_files", ".opencode/tools/find_files.md"]]),
        compatibilityWrappers: new Set<string>(),
        metadataDiagnostics: [],
        testPathsByBasename: new Map(),
      },
    );

    expect(row.status).toBe("ok");
    assertPublicSchemaOmitsKeys(definition, COMPACT_SCHEMA_FIELD_FIXTURES.findFiles.omitted);
    assertCountedAndExemptFields(definition, {
      counted: COMPACT_SCHEMA_FIELD_FIXTURES.findFiles.counted,
      exempt: COMPACT_SCHEMA_FIELD_FIXTURES.findFiles.exempt,
      actualCounted: row.counted_fields,
      actualExempt: row.exempt_fields,
    });
  });

  it("accepts bounded discovery options via options", async () => {
    setSpawnResponse({ stdout: ".opencode/tools/find_files.ts\n", exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    await execute({
      pattern: "**/*.ts",
      options: "file-type=ts glob-case-insensitive compact-output",
    });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args).toContain("-t");
    expect(args[args.indexOf("-t") + 1]).toBe("ts");
    expect(args).toContain("--glob-case-insensitive");
  });

  it("rejects unknown options tokens", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", options: "bogus" });
    assertContains(String(result), "Invalid options token 'bogus'");
  });

  it("rejects invalid max-results token values", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", options: "max-results=abc" });
    assertContains(String(result), "max-results must be a non-negative integer");
  });

  it("rejects tokens with multiple separators", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", options: "file-type=ts=js" });
    assertContains(String(result), "at most one '=' separator");
  });

  it("assembles rg --files", async () => {
    setSpawnResponse({ stdout: "a.ts\n", exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    await execute({ pattern: "**/*.ts" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args[0]).toBe("rg");
    expect(args).toContain("--files");
    expect(args).toContain("--glob");
    expect(args[args.indexOf("--glob") + 1]).toBe("**/*.ts");
  });

  it("prefers stderr over stdout on non-zero exit", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ pattern: "**/*.ts" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("discovers a single file when path targets a file", async () => {
    setSpawnResponse({ stdout: `${FIXTURE_FILE}\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", path: FIXTURE_FILE });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(FIXTURE_FILE);
    expect(String(result)).toContain(path.relative(TOOL_CWD, FIXTURE_FILE));
  });

  it("keeps directory discovery scoped to the requested subtree", async () => {
    const nestedFile = path.join(NESTED_DIR, "beta.ts");
    setSpawnResponse({ stdout: `${nestedFile}\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", path: NESTED_DIR, options: "compact-output" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(NESTED_DIR);
    expect(String(result)).toBe("beta.ts");
  });

  it("returns basename-style compact output for file targets", async () => {
    setSpawnResponse({ stdout: `${FIXTURE_FILE}\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", path: FIXTURE_FILE, options: "compact-output" });
    expect(String(result)).toBe("alpha.ts");
  });

  it("returns a deterministic error for missing scoped paths", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const missingPath = path.join(FIXTURE_DIR, "missing.ts");
    const result = await execute({ pattern: "**/*.ts", path: missingPath });
    expect(String(result)).toBe(
      `ERROR: Search path does not exist: ${missingPath}\n\nHint: Verify the path is correct.`,
    );
  });

  it("rejects out-of-repo paths without widening scope", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", path: "/etc/passwd" });
    assertContains(String(result), "outside the repository");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves .trash subtree discovery scope", async () => {
    const trashedFile = path.join(TRASH_DIR, "trashed.ts");
    setSpawnResponse({ stdout: `${trashedFile}\n`, exitCode: 0 });
    const execute = await loadToolExecute("../../find_files.ts");
    const result = await execute({ pattern: "**/*.ts", path: TRASH_DIR, options: "compact-output" });
    const args = getInvocations().at(-1)?.args ?? [];
    expect(args.at(-1)).toBe(TRASH_DIR);
    expect(String(result)).toBe("trashed.ts");
  });

  it("keeps valid scoped empty results non-error for file and directory targets", async () => {
    setSpawnResponse({ stdout: "", exitCode: 1 });
    const execute = await loadToolExecute("../../find_files.ts");
    const fileResult = await execute({ pattern: "**/*.py", path: FIXTURE_FILE });
    expect(String(fileResult)).toBe(`No files found matching pattern '**/*.py' in '${FIXTURE_FILE}'.`);

    const dirResult = await execute({ pattern: "**/*.py", path: NESTED_DIR });
    expect(String(dirResult)).toBe(`No files found matching pattern '**/*.py' in '${NESTED_DIR}'.`);
  });

  it("sorts by mtime before applying max-results truncation", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const olderFile = path.join(FIXTURE_DIR, "older.ts");
    const newestFile = path.join(FIXTURE_DIR, "newest.ts");
    const middleFile = path.join(FIXTURE_DIR, "middle.ts");
    setSpawnResponse({ stdout: `${olderFile}\n${newestFile}\n${middleFile}\n`, exitCode: 0 });

    const originalStat = fs.promises.stat;
    fs.promises.stat = (async (targetPath: fs.PathLike) => {
      if (targetPath === olderFile) return { isDirectory: () => false, isFile: () => true, mtimeMs: 1 } as fs.Stats;
      if (targetPath === middleFile) return { isDirectory: () => false, isFile: () => true, mtimeMs: 5 } as fs.Stats;
      if (targetPath === newestFile) return { isDirectory: () => false, isFile: () => true, mtimeMs: 10 } as fs.Stats;
      return originalStat(targetPath);
    }) as typeof fs.promises.stat;

    try {
      const result = await execute({ pattern: "**/*.ts", options: "max-results=2 compact-output" });
      expect(String(result)).toBe(`.opencode/tools/__tests__/fixtures/search_scope/newest.ts\n.opencode/tools/__tests__/fixtures/search_scope/middle.ts\n\n${
        '[WARNING: Results truncated to 2 files (3 total found). Use options: "max-results=<n>" to increase limit.]'
      }`);
    } finally {
      fs.promises.stat = originalStat;
    }
  });

  it("forwards explicit max-results to discovery execution before truncation", async () => {
    const execute = await loadToolExecute("../../find_files.ts");
    const files = Array.from({ length: 5002 }, (_, index) =>
      path.join(FIXTURE_DIR, `bulk-${String(index).padStart(4, "0")}.ts`),
    );
    setSpawnResponse({ stdout: `${files.join("\n")}\n`, exitCode: 0 });

    const originalStat = fs.promises.stat;
    fs.promises.stat = (async (targetPath: fs.PathLike) => {
      const resolved = String(targetPath);
      const index = files.indexOf(resolved);
      if (index >= 0) {
        return {
          isDirectory: () => false,
          isFile: () => true,
          mtimeMs: index,
        } as fs.Stats;
      }
      return originalStat(targetPath);
    }) as typeof fs.promises.stat;

    try {
      const result = await execute({ pattern: "**/*.ts", options: "max-results=5001 compact-output" });
      expect(String(result)).toContain("bulk-5001.ts");
      expect(String(result)).toContain("bulk-0001.ts");
      expect(String(result)).not.toContain("bulk-0000.ts");
      expect(String(result)).toContain("Results truncated to 5001 files (5002 total found)");
    } finally {
      fs.promises.stat = originalStat;
    }
  });
});
