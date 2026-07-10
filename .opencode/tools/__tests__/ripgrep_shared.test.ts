import { describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import path from "node:path";

const HELPER_MODULE_PATH = path.join(import.meta.dir, "../lib/ripgrep_shared.ts");
let importCounter = 0;

const loadHelper = async () => {
  importCounter += 1;
  return import(`${HELPER_MODULE_PATH}?test=${importCounter}`);
};

describe("ripgrep_shared helper", () => {
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
});
