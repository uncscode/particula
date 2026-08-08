import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import path from "node:path";

import {
  getKillCount,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnResponse,
} from "./helpers/mock-subprocess";

const HELPER_MODULE_PATH = path.join(import.meta.dir, "../lib/ripgrep_shared.ts");
let importCounter = 0;

const loadHelper = async () => {
  importCounter += 1;
  return import(`${HELPER_MODULE_PATH}?test=${importCounter}`);
};

describe("ripgrep_shared helper", () => {
  beforeEach(() => {
    installSubprocessMocks();
  });

  afterEach(() => {
    restoreSubprocessMocks();
  });

  it("normalizes numeric params and validates non-negative integers", async () => {
    const { normalizeNumericParam, validateNonNegativeInt } = await loadHelper();

    expect(normalizeNumericParam(undefined)).toBeUndefined();
    expect(normalizeNumericParam(0)).toBeUndefined();
    expect(normalizeNumericParam(3)).toBe(3);
    expect(normalizeNumericParam(1.5)).toBeUndefined();

    expect(validateNonNegativeInt(undefined, "maxResults")).toBeUndefined();
    expect(validateNonNegativeInt("0", "maxResults")).toBeUndefined();
    expect(validateNonNegativeInt("7", "maxResults")).toBeUndefined();
    expect(validateNonNegativeInt("-1", "maxResults")).toBe(
      "ERROR: Invalid maxResults value. It must be a non-negative integer.",
    );
  });

  it("parses literal-safe simple and regex-opt-in search requests", async () => {
    const { parseRipgrepSearchRequest } = await loadHelper();

    expect(parseRipgrepSearchRequest("  a.b  ", "pattern=**/*.ts max-results=2", false)).toEqual({
      ok: true,
      request: {
        contentPattern: "a.b",
        matchMode: "literal",
        pattern: "**/*.ts",
        maxResults: 2,
      },
    });
    expect(parseRipgrepSearchRequest("a.b", "match-mode=regex", false)).toMatchObject({
      ok: true,
      request: { matchMode: "regex" },
    });
  });

  it("fails closed for duplicate, advanced-only, and incompatible request tokens", async () => {
    const { parseRipgrepSearchRequest } = await loadHelper();

    expect(parseRipgrepSearchRequest("x", "max-results=2 max-results=3", false)).toMatchObject({
      ok: false,
      error: expect.stringContaining("duplicate 'max-results'"),
    });
    expect(parseRipgrepSearchRequest("x", "before-context=1", false)).toMatchObject({
      ok: false,
      error: expect.stringContaining("not allowed"),
    });
    expect(parseRipgrepSearchRequest("x", "files-with-matches context-lines=1", true)).toMatchObject({
      ok: false,
      error: expect.stringContaining("files-only modes cannot be combined with context"),
    });
  });

  it("rejects malformed glob delimiters before path resolution", async () => {
    const { parseRipgrepSearchRequest } = await loadHelper();

    for (const token of ["pattern=src]", "pattern=src}", "pattern=[]", "pattern={}"]) {
      expect(parseRipgrepSearchRequest("x", token, false)).toMatchObject({
        ok: false,
        error: expect.stringContaining("invalid glob pattern"),
      });
    }
  });

  it("classifies file and directory targets", async () => {
    const { resolveValidatedSearchPath } = await loadHelper();
    const cwd = path.resolve(import.meta.dir, "../..");
    const filePath = path.join(import.meta.dir, "fixtures/search_scope/alpha.ts");
    const dirPath = path.join(import.meta.dir, "fixtures/search_scope/nested");

    const fileResult = await resolveValidatedSearchPath(filePath, cwd);
    expect(fileResult.error).toBeUndefined();
    expect(fileResult.targetKind).toBe("file");
    expect(fileResult.compactOutputBase).toBe(path.dirname(fileResult.canonicalPath!));

    const dirResult = await resolveValidatedSearchPath(dirPath, cwd);
    expect(dirResult.error).toBeUndefined();
    expect(dirResult.targetKind).toBe("directory");
    expect(dirResult.compactOutputBase).toBe(dirResult.canonicalPath);
  });

  it("fails closed when canonical resolution fails after stat succeeds", async () => {
    const { resolveValidatedSearchPath } = await loadHelper();
    const originalStat = fs.promises.stat;
    const originalRealpath = fs.promises.realpath;

    fs.promises.stat = (async () => ({ isDirectory: () => false, isFile: () => true })) as typeof fs.promises.stat;
    fs.promises.realpath = (async (target: string) => {
      if (target === "/repo" || target === "/repo/file.ts") {
        return Promise.reject(new Error("boom"));
      }
      return target;
    }) as typeof fs.promises.realpath;

    try {
      const result = await resolveValidatedSearchPath("/repo/file.ts", "/repo");
      expect(result).toEqual({
        error:
          "ERROR: Unable to resolve canonical search path: /repo/file.ts\n\nHint: Verify the path exists and is accessible.",
      });
    } finally {
      fs.promises.stat = originalStat;
      fs.promises.realpath = originalRealpath;
    }
  });

  it("formats truncation warnings with exact and approximate totals", async () => {
    const { buildTruncationWarning } = await loadHelper();

    expect(buildTruncationWarning(2, 3, "files")).toBe(
      '[WARNING: Results truncated to 2 files (3 total found). Use options: "max-results=<n>" to increase limit.]',
    );
    expect(buildTruncationWarning(10, 12, "lines", { approximateTotal: true })).toBe(
      '[WARNING: Results truncated to 10 lines (at least 12 total found). Use options: "max-results=<n>" to increase limit.]',
    );
  });

  it("kills a stalled ripgrep subprocess and returns a deterministic timeout", async () => {
    setSpawnResponse({ hangs: true });
    const { executeRipgrepSearch } = await loadHelper();

    const result = await executeRipgrepSearch({
      contentPattern: "needle",
      searchPath: ".",
      timeoutMs: 5,
    });

    expect(result).toEqual({
      files: [],
      exitCode: 2,
      errorMessage:
        "ERROR: Ripgrep search timed out after 5ms.\n\n" +
        "Hint: Narrow the search path or pattern and try again.",
    });
    expect(getKillCount()).toBe(1);
  });
});
