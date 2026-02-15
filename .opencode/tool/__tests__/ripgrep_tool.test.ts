import assert from "node:assert";
import { beforeEach, describe, it, mock } from "node:test";
import * as fs from "fs";
import * as path from "path";

const createSchema = (kind: string, extras: Record<string, unknown> = {}) => {
  return {
    kind,
    ...extras,
    optional() {
      return { ...this, optional: true };
    },
    describe(description: string) {
      return { ...this, description };
    },
  };
};

const schema = {
  string: () => createSchema("string"),
  number: () => createSchema("number"),
  boolean: () => createSchema("boolean"),
};

const toolFn = (definition: unknown) => definition;
(toolFn as any).schema = schema;

mock.module("@opencode-ai/plugin", () => ({
  tool: toolFn,
}));

const ripgrepTool = (await import("../ripgrep")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, any>) => Promise<string>;
};

type SpawnResult = {
  stdout?: string;
  stderr?: string;
  exitCode?: number;
};

let lastCmd: string[] = [];
let allCmds: string[][] = [];
let spawnResult: SpawnResult = {};
let spawnResults: SpawnResult[] = [];
let spawnCallCount = 0;

function setSpawnResult(result: SpawnResult) {
  spawnResult = result;
  spawnResults = [result];
}

function setSpawnResults(results: SpawnResult[]) {
  spawnResults = results;
  spawnResult = results[0] ?? {};
}

function createBunMock() {
  return {
    spawnSync: (args: string[]) => {
      lastCmd = args;
      allCmds.push([...args]);
      const result = spawnResults[spawnCallCount] ?? spawnResult;
      spawnCallCount++;
      return {
        stdout: Buffer.from(result.stdout ?? ""),
        stderr: Buffer.from(result.stderr ?? ""),
        exitCode: result.exitCode ?? 0,
      };
    },
  };
}

(globalThis as any).Bun = createBunMock();

beforeEach(() => {
  setSpawnResult({ stdout: "" });
  lastCmd = [];
  allCmds = [];
  spawnCallCount = 0;
  (globalThis as any).Bun = createBunMock();
  mock.restoreAll();
  mock.method(fs.promises, "stat", async () => ({
    isDirectory: () => true,
    mtimeMs: Date.now(),
  }) as any);
});

describe("ripgrep tool", () => {
  it("adds -t flag for fileType before search path", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    await ripgrepTool.execute({ pattern: "**/*", fileType: "ts" });

    const resolvedPath = path.resolve(process.cwd());
    const tIndex = lastCmd.indexOf("-t");
    assert.ok(tIndex > 0);
    assert.equal(lastCmd[tIndex + 1], "ts");
    assert.ok(lastCmd.indexOf(resolvedPath) > tIndex);
  });

  it("adds -T flag for excludeFileType after include when both provided", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    await ripgrepTool.execute({ pattern: "**/*", fileType: "ts", excludeFileType: "json" });

    const tIndex = lastCmd.indexOf("-t");
    const tExcludeIndex = lastCmd.indexOf("-T");
    assert.ok(tIndex > 0);
    assert.ok(tExcludeIndex > tIndex);
    assert.equal(lastCmd[tExcludeIndex + 1], "json");
  });

  it("adds globCaseInsensitive flag and combines with file types", async () => {
    setSpawnResult({ stdout: "/repo/file.md" });

    await ripgrepTool.execute({
      pattern: "**/readme.md",
      fileType: "md",
      excludeFileType: "json",
      globCaseInsensitive: true,
    });

    assert.ok(lastCmd.includes("--glob-case-insensitive"));
    const globIndex = lastCmd.indexOf("--glob-case-insensitive");
    const pathIndex = lastCmd.length - 1;
    assert.ok(globIndex < pathIndex);
  });

  it("prefers unrestricted over ignore flags", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    await ripgrepTool.execute({
      pattern: "**/*",
      ignoreGitignore: true,
      includeHidden: true,
      unrestricted: 2,
    });

    assert.ok(lastCmd.includes("-uu"));
    assert.ok(!lastCmd.includes("--no-ignore-vcs"));
    assert.ok(!lastCmd.includes("--hidden"));
  });

  it("keeps combined ordering with includeHidden and fileType flags", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    await ripgrepTool.execute({
      pattern: "**/*",
      includeHidden: true,
      fileType: "ts",
      globCaseInsensitive: true,
    });

    const hiddenIndex = lastCmd.indexOf("--hidden");
    const typeIndex = lastCmd.indexOf("-t");
    const globIndex = lastCmd.indexOf("--glob-case-insensitive");
    const pathIndex = lastCmd.length - 1;

    assert.ok(hiddenIndex > 0);
    assert.ok(typeIndex > hiddenIndex);
    assert.ok(globIndex > typeIndex);
    assert.ok(pathIndex > globIndex);
  });

  it("returns no matches message with search path context on exit code 1", async () => {
    setSpawnResult({ stdout: "", exitCode: 1 });

    const result = await ripgrepTool.execute({ pattern: "**/*.ts", path: "src" });

    assert.equal(result, "No files found matching pattern '**/*.ts' in 'src'.");
  });

  it("returns invalid pattern error on exit code 2 with stderr", async () => {
    setSpawnResult({ stdout: "", stderr: "invalid pattern", exitCode: 2 });

    const result = await ripgrepTool.execute({ pattern: "[" });

    assert.ok(result.startsWith("ERROR: Invalid glob pattern"));
    assert.ok(result.includes("["));
  });

  it("uses -n and contentPattern with optional --glob", async () => {
    setSpawnResult({ stdout: "src/app.ts:4:TODO", exitCode: 0 });

    await ripgrepTool.execute({ pattern: "**/*.ts", contentPattern: "TODO" });

    assert.equal(lastCmd[0], "rg");
    assert.ok(lastCmd.includes("-n"));
    assert.ok(lastCmd.includes("TODO"));
    const globIndex = lastCmd.indexOf("--glob");
    assert.ok(globIndex > -1);
    assert.equal(lastCmd[globIndex + 1], "**/*.ts");
  });

  it("adds -l for filesWithMatches and returns path-only output", async () => {
    setSpawnResult({ stdout: "src/app.ts\nsrc/lib.ts", exitCode: 0 });

    const result = await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithMatches: true,
    });

    assert.ok(lastCmd.includes("-l"));
    assert.equal(result, "src/app.ts\nsrc/lib.ts");
  });

  it("adds -L for filesWithoutMatches and returns path-only output", async () => {
    setSpawnResult({ stdout: "src/app.ts\nsrc/lib.ts", exitCode: 0 });

    const result = await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithoutMatches: true,
    });

    assert.ok(lastCmd.includes("-L"));
    assert.equal(result, "src/app.ts\nsrc/lib.ts");
  });

  it("rejects filesWithMatches and filesWithoutMatches together", async () => {
    const result = await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithMatches: true,
      filesWithoutMatches: true,
    });

    assert.equal(
      result,
      "ERROR: 'filesWithMatches' and 'filesWithoutMatches' cannot both be true."
    );
    assert.deepEqual(lastCmd, []);
  });

  it("errors when filesWithMatches is set without contentPattern", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    const result = await ripgrepTool.execute({ pattern: "**/*", filesWithMatches: true });

    assert.ok(result.includes("filesWithMatches/filesWithoutMatches are ignored"));
    assert.ok(result.includes("contentPattern"));
    assert.ok(!lastCmd.includes("-l"));
  });

  it("errors when filesWithoutMatches is set without contentPattern", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    const result = await ripgrepTool.execute({ pattern: "**/*", filesWithoutMatches: true });

    assert.ok(result.includes("filesWithMatches/filesWithoutMatches are ignored"));
    assert.ok(result.includes("contentPattern"));
    assert.ok(!lastCmd.includes("-L"));
  });

  it("formats paths relative to search path in filesWithMatches mode", async () => {
    const cwd = process.cwd();
    const searchPath = path.join(cwd, "src");
    const filePath = path.join(searchPath, "match.ts");
    setSpawnResult({ stdout: filePath, exitCode: 0 });

    const result = await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithMatches: true,
      path: searchPath,
      compactOutput: true,
    });

    assert.equal(result.trim(), "match.ts");
  });

  it("passes --max-count when maxResults set in content search", async () => {
    setSpawnResult({ stdout: "src/app.ts:1:TODO", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", maxResults: 5 });

    const countIndex = lastCmd.indexOf("--max-count");
    assert.ok(countIndex > -1);
    assert.equal(lastCmd[countIndex + 1], "5");
  });

  it("omits --max-count in filesWithMatches mode", async () => {
    setSpawnResult({ stdout: "src/app.ts", exitCode: 0 });

    await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithMatches: true,
      maxResults: 5,
    });

    assert.ok(!lastCmd.includes("--max-count"));
  });

  it("omits --max-count in filesWithoutMatches mode", async () => {
    setSpawnResult({ stdout: "src/app.ts", exitCode: 0 });

    await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithoutMatches: true,
      maxResults: 5,
    });

    assert.ok(!lastCmd.includes("--max-count"));
  });

  it("adds -C when contextLines set for content search", async () => {
    setSpawnResult({ stdout: "src/app.ts:1:TODO", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", contextLines: 2 });

    const contextIndex = lastCmd.indexOf("-C");
    assert.ok(contextIndex > -1);
    assert.equal(lastCmd[contextIndex + 1], "2");
  });

  it("adds -B/-A for directional context and skips -C", async () => {
    setSpawnResult({ stdout: "src/app.ts:1:TODO", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", beforeContext: 3, afterContext: 1 });

    const beforeIndex = lastCmd.indexOf("-B");
    const afterIndex = lastCmd.indexOf("-A");
    assert.ok(beforeIndex > -1);
    assert.ok(afterIndex > -1);
    assert.equal(lastCmd[beforeIndex + 1], "3");
    assert.equal(lastCmd[afterIndex + 1], "1");
    assert.ok(!lastCmd.includes("-C"));
  });

  it("skips context flags for non-positive values", async () => {
    setSpawnResult({ stdout: "src/app.ts:1:TODO", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", contextLines: 0, beforeContext: -2 });

    assert.ok(!lastCmd.includes("-C"));
    assert.ok(!lastCmd.includes("-B"));
    assert.ok(!lastCmd.includes("-A"));
  });

  it("rejects non-integer contextLines value", async () => {
    const result = await ripgrepTool.execute({ contentPattern: "TODO", contextLines: 2.5 });

    assert.ok(result.includes("ERROR: Invalid contextLines value"));
    assert.ok(result.includes("non-negative integer"));
    assert.deepEqual(lastCmd, []);
  });

  it("rejects negative beforeContext value", async () => {
    const result = await ripgrepTool.execute({ contentPattern: "TODO", beforeContext: -1 });

    assert.ok(result.includes("ERROR: Invalid beforeContext value"));
    assert.deepEqual(lastCmd, []);
  });

  it("rejects non-integer afterContext value", async () => {
    const result = await ripgrepTool.execute({ contentPattern: "TODO", afterContext: 1.7 });

    assert.ok(result.includes("ERROR: Invalid afterContext value"));
    assert.deepEqual(lastCmd, []);
  });

  it("does not pass --max-count when context flags are active", async () => {
    setSpawnResult({ stdout: "a.ts:1:TODO\na.ts:2:ctx\na.ts:3:ctx", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", maxResults: 5, contextLines: 2 });

    assert.ok(!lastCmd.includes("--max-count"), "--max-count should not be passed with context");
    assert.ok(lastCmd.includes("-C"), "-C should be present");
  });

  it("omits --max-count when maxMatchesPerFile set with context flags", async () => {
    setSpawnResult({ stdout: "a.ts:1:TODO\na.ts:2:ctx\na.ts:3:ctx", exitCode: 0 });

    await ripgrepTool.execute({
      contentPattern: "TODO",
      maxMatchesPerFile: 2,
      contextLines: 1,
    });

    assert.ok(!lastCmd.includes("--max-count"), "--max-count should not be passed with context");
    assert.ok(lastCmd.includes("-C"), "-C should be present");
  });

  it("still passes --max-count when no context flags are active", async () => {
    setSpawnResult({ stdout: "a.ts:1:TODO", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", maxResults: 5 });

    const countIndex = lastCmd.indexOf("--max-count");
    assert.ok(countIndex > -1);
    assert.equal(lastCmd[countIndex + 1], "5");
  });

  it("prefers maxMatchesPerFile over maxResults for --max-count", async () => {
    setSpawnResult({ stdout: "a.ts:1:TODO", exitCode: 0 });

    await ripgrepTool.execute({
      contentPattern: "TODO",
      maxResults: 5,
      maxMatchesPerFile: 2,
    });

    const countIndex = lastCmd.indexOf("--max-count");
    assert.ok(countIndex > -1);
    assert.equal(lastCmd[countIndex + 1], "2");
  });

  it("preserves context separators in output", async () => {
    setSpawnResult({ stdout: "a.ts:1:TODO\n--\na.ts:10:TODO", exitCode: 0 });

    const result = await ripgrepTool.execute({ contentPattern: "TODO", contextLines: 1 });

    assert.equal(result, "a.ts:1:TODO\n--\na.ts:10:TODO");
  });

  it("warns when context parameters are provided without contentPattern", async () => {
    setSpawnResult({ stdout: "/repo/file.ts" });

    const result = await ripgrepTool.execute({ pattern: "**/*", contextLines: 2 });

    assert.ok(result.includes("contextLines/beforeContext/afterContext are ignored"));
    assert.ok(!lastCmd.includes("-C"));
  });

  it("skips auto-retry for content search with no matches", async () => {
    setSpawnResult({ stdout: "", exitCode: 1 });

    const result = await ripgrepTool.execute({ contentPattern: "notfound" });

    assert.equal(allCmds.length, 1);
    assert.equal(result, "No matches found for contentPattern 'notfound'.");
  });

  it("returns contentPattern regex error message", async () => {
    setSpawnResult({ stdout: "", stderr: "regex parse error", exitCode: 2 });

    const result = await ripgrepTool.execute({ contentPattern: "[" });

    assert.ok(result.startsWith("ERROR: Invalid contentPattern regex"));
    assert.ok(result.includes("["));
  });

  it("truncates content search lines with warning", async () => {
    setSpawnResult({
      stdout: "a.ts:1:TODO\nb.ts:2:TODO\nc.ts:3:TODO",
      exitCode: 0,
    });

    const result = await ripgrepTool.execute({ contentPattern: "TODO", maxResults: 2 });

    const lines = result.split("\n");
    assert.equal(lines[0], "a.ts:1:TODO");
    assert.equal(lines[1], "b.ts:2:TODO");
    assert.ok(result.includes("Results truncated to 2 lines (3 total found)"));
  });

  it("rejects empty contentPattern", async () => {
    const result = await ripgrepTool.execute({ contentPattern: "  " });

    assert.equal(result, "ERROR: 'contentPattern' cannot be empty.");
    assert.deepEqual(lastCmd, []);
  });

  it("ignores whitespace-only pattern in content search mode", async () => {
    setSpawnResult({ stdout: "file.ts:1:TODO\nfile.js:2:TODO", exitCode: 0 });

    await ripgrepTool.execute({ contentPattern: "TODO", pattern: "  " });

    // Should not include --glob flag when pattern is only whitespace
    assert.ok(!lastCmd.includes("--glob"));
  });

  it("returns rg not installed guidance when spawn throws ENOENT", async () => {
    (globalThis as any).Bun.spawnSync = () => {
      const error = new Error("ENOENT: rg not found");
      throw error;
    };

    const result = await ripgrepTool.execute({ pattern: "**/*" });

    assert.ok(result.includes("rg) is not installed"));
    assert.ok(result.toLowerCase().includes("install"));
  });

  it("rejects search path outside repository", async () => {
    const result = await ripgrepTool.execute({ pattern: "**/*", path: "/tmp" });

    assert.ok(result.startsWith("ERROR: Search path is outside the repository"));
  });

  it("returns filesWithoutMatches empty-result message", async () => {
    setSpawnResult({ stdout: "", exitCode: 1 });

    const result = await ripgrepTool.execute({
      contentPattern: "TODO",
      filesWithoutMatches: true,
    });

    assert.equal(result, "No files found without matches for contentPattern 'TODO'.");
  });

  it("rejects search path that is not a directory", async () => {
    mock.method(fs.promises, "stat", async () => ({
      isDirectory: () => false,
    }) as any);

    const result = await ripgrepTool.execute({ pattern: "**/*", path: "file.txt" });

    assert.ok(result.startsWith("ERROR: Search path is not a directory"));
    assert.deepEqual(lastCmd, []);
  });

  it("rejects search path that does not exist", async () => {
    mock.method(fs.promises, "stat", async () => {
      throw new Error("not found");
    });

    const result = await ripgrepTool.execute({ pattern: "**/*", path: "missing" });

    assert.ok(result.startsWith("ERROR: Search path does not exist"));
    assert.deepEqual(lastCmd, []);
  });
});

describe("ripgrep tool auto-retry", () => {
  it("triggers auto-retry when initial search returns empty", async () => {
    const cwd = process.cwd();
    // First call returns empty, second call returns results
    setSpawnResults([
      { stdout: "", exitCode: 1 },
      { stdout: `${cwd}/node_modules/pkg/package.json`, exitCode: 0 },
    ]);

    const result = await ripgrepTool.execute({ pattern: "**/package.json" });

    // Should have made two calls
    assert.equal(allCmds.length, 2);
    // First call should NOT have --no-ignore-vcs
    assert.ok(!allCmds[0].includes("--no-ignore-vcs"));
    // Second call should have --no-ignore-vcs and --hidden
    assert.ok(allCmds[1].includes("--no-ignore-vcs"));
    assert.ok(allCmds[1].includes("--hidden"));
    // Output should contain auto-retry indicator
    assert.ok(result.includes("Auto-retry activated"));
    assert.ok(result.includes("ignoreGitignore=true"));
  });

  it("formats paths relative to cwd by default", async () => {
    const cwd = process.cwd();
    const filePath = path.join(cwd, "sub", "file.ts");
    setSpawnResult({ stdout: filePath });

    const result = await ripgrepTool.execute({ pattern: "**/*.ts", path: "sub" });

    assert.equal(result.trim(), path.relative(cwd, filePath));
  });

  it("formats paths relative to search path when compactOutput is true", async () => {
    const cwd = process.cwd();
    const searchPath = path.join(cwd, "src", "nested");
    const nestedFile = path.join(searchPath, "file.ts");
    setSpawnResult({ stdout: nestedFile });

    const result = await ripgrepTool.execute({
      pattern: "**/*.ts",
      path: searchPath,
      compactOutput: true,
    });

    assert.equal(result.trim(), "file.ts");
  });

  it("keeps cwd formatting when compactOutput is true and path is '.'", async () => {
    const cwd = process.cwd();
    const filePath = path.join(cwd, "file.ts");
    setSpawnResult({ stdout: filePath });

    const result = await ripgrepTool.execute({ pattern: "**/*.ts", path: ".", compactOutput: true });

    assert.equal(result.trim(), path.relative(cwd, filePath));
  });

  it("falls back to full path when relative path is empty", async () => {
    const cwd = process.cwd();
    const searchPath = path.join(cwd, "file.ts");
    setSpawnResult({ stdout: searchPath });

    const result = await ripgrepTool.execute({
      pattern: "**/*.ts",
      path: searchPath,
      compactOutput: true,
    });

    assert.equal(result.trim(), searchPath);
  });

  it("keeps mtime ordering and compact formatting with truncation warning", async () => {
    const cwd = process.cwd();
    const searchPath = path.join(cwd, "src");
    const files = [
      path.join(searchPath, "a.ts"),
      path.join(searchPath, "b.ts"),
      path.join(searchPath, "c.ts"),
    ];
    setSpawnResult({ stdout: files.join("\n") });

    let mtimeCall = 0;
    mock.method(fs.promises, "stat", async () => {
      const fakeTimes = [3, 2, 1];
      const value = fakeTimes[mtimeCall] ?? 0;
      mtimeCall++;
      return {
        isDirectory: () => true,
        mtimeMs: value,
      } as any;
    });

    const result = await ripgrepTool.execute({
      pattern: "**/*.ts",
      path: searchPath,
      compactOutput: true,
      maxResults: 2,
    });

    const lines = result.split("\n");
    assert.ok(lines[0].endsWith("a.ts"));
    assert.ok(lines[1].endsWith("b.ts"));
    assert.ok(lines[lines.length - 1].includes("WARNING: Results truncated"));
  });

  it("uses compact formatting in auto-retry output", async () => {
    const cwd = process.cwd();
    const searchPath = path.join(cwd, "src");
    setSpawnResults([
      { stdout: "", exitCode: 1 },
      { stdout: path.join(searchPath, "found.ts"), exitCode: 0 },
    ]);

    const result = await ripgrepTool.execute({
      pattern: "**/*.ts",
      path: "src",
      compactOutput: true,
    });

    assert.ok(result.includes("Auto-retry activated"));
    const lines = result.trim().split("\n");
    assert.ok(lines.some((line) => line.trim() === "found.ts"));
  });
});

  it("does NOT auto-retry when ignoreGitignore explicitly set", async () => {
    setSpawnResult({ stdout: "", exitCode: 1 });

    const result = await ripgrepTool.execute({
      pattern: "**/nonexistent.xyz",
      ignoreGitignore: true,
    });

    // Should only make one call
    assert.equal(allCmds.length, 1);
    // Output should NOT mention auto-retry
    assert.ok(!result.includes("Auto-retry"));
    assert.equal(result, "No files found matching pattern '**/nonexistent.xyz'.");
  });

  it("does NOT auto-retry when unrestricted mode used", async () => {
    setSpawnResult({ stdout: "", exitCode: 1 });

    const result = await ripgrepTool.execute({
      pattern: "**/nonexistent.xyz",
      unrestricted: 1,
    });

    // Should only make one call
    assert.equal(allCmds.length, 1);
    // First call should have -u flag
    assert.ok(allCmds[0].includes("-u"));
    // Output should NOT mention auto-retry
    assert.ok(!result.includes("Auto-retry"));
  });

  it("shows both searches empty message when retry also finds nothing", async () => {
    // Both calls return empty
    setSpawnResults([
      { stdout: "", exitCode: 1 },
      { stdout: "", exitCode: 1 },
    ]);

    const result = await ripgrepTool.execute({ pattern: "**/definitely-not-exists.xyz" });

    // Should have made two calls
    assert.equal(allCmds.length, 2);
    // Output should mention auto-retry was attempted
    assert.ok(result.includes("Auto-retry was attempted"));
    assert.ok(result.includes("ignoreGitignore=true"));
    assert.ok(result.includes("includeHidden=true"));
  });

  it("preserves original parameters except ignore flags during retry", async () => {
    const cwd = process.cwd();
    setSpawnResults([
      { stdout: "", exitCode: 1 },
      { stdout: `${cwd}/src/test.ts`, exitCode: 0 },
    ]);

    await ripgrepTool.execute({
      pattern: "**/test",
      fileType: "ts",
      globCaseInsensitive: true,
    });

    // Both calls should preserve fileType and globCaseInsensitive
    assert.ok(allCmds[0].includes("-t"));
    assert.ok(allCmds[0].includes("ts"));
    assert.ok(allCmds[0].includes("--glob-case-insensitive"));

    assert.ok(allCmds[1].includes("-t"));
    assert.ok(allCmds[1].includes("ts"));
    assert.ok(allCmds[1].includes("--glob-case-insensitive"));
    // Retry should add ignore flags
    assert.ok(allCmds[1].includes("--no-ignore-vcs"));
    assert.ok(allCmds[1].includes("--hidden"));
  });

  it("output format matches specification with emoji indicator", async () => {
    const cwd = process.cwd();
    setSpawnResults([
      { stdout: "", exitCode: 1 },
      { stdout: `${cwd}/found.ts`, exitCode: 0 },
    ]);

    const result = await ripgrepTool.execute({ pattern: "**/*" });

    // Check output format matches spec
    assert.ok(result.startsWith("⚠️ Auto-retry activated"));
    assert.ok(result.includes("Reason: Initial search returned no results"));
    assert.ok(result.includes("Original: ignoreGitignore=false"));
    assert.ok(result.includes("Retry:    ignoreGitignore=true, includeHidden=true"));
  });

  it("auto-retry still triggers when includeHidden is true but ignoreGitignore is not set", async () => {
    const cwd = process.cwd();
    setSpawnResults([
      { stdout: "", exitCode: 1 },
      { stdout: `${cwd}/found.ts`, exitCode: 0 },
    ]);

    const result = await ripgrepTool.execute({
      pattern: "**/*",
      includeHidden: true,
    });

    // Should have made two calls (gitignore is the common blocker)
    assert.equal(allCmds.length, 2);
    // First call should have --hidden but not --no-ignore-vcs
    assert.ok(allCmds[0].includes("--hidden"));
    assert.ok(!allCmds[0].includes("--no-ignore-vcs"));
    // Output should show original includeHidden was true
    assert.ok(result.includes("Original: ignoreGitignore=false, includeHidden=true"));
  });
});
