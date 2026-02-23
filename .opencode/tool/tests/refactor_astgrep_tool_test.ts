import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it, mock } from "node:test";

const moduleMock =
  typeof mock.module === "function"
    ? mock.module
    : (await import("bun:test")).mock.module;

const schemaChain = {
  optional: () => schemaChain,
  describe: () => schemaChain,
};

const schema = {
  string: () => schemaChain,
  boolean: () => schemaChain,
  enum: () => schemaChain,
};

const tool = (definition: unknown) => definition;
(tool as any).schema = schema;

moduleMock("@opencode-ai/plugin", () => ({ tool }));

const refactorAstgrepTool = (await import("../refactor_astgrep")).default;
const MISSING_HINT = "Install ast-grep-cli (pip install ast-grep-cli) and ensure ast-grep is on PATH.";

const calls: any[] = [];
let originalBun: any;
let originalBunDollar: any;

beforeEach(() => {
  originalBun = (globalThis as any).Bun;
  originalBunDollar = originalBun?.$;
});

afterEach(() => {
  if (originalBun === undefined) {
    delete (globalThis as any).Bun;
    return;
  }

  originalBun.$ = originalBunDollar;
});

function setBunStub({ output = "", error }: { output?: string; error?: any }) {
  calls.length = 0;
  const stub = (...args: any[]) => {
    calls.push(args);
    if (error) {
      throw error;
    }
    return {
      text: async () => output,
    };
  };

  const bunGlobal = (globalThis as any).Bun;
  if (bunGlobal) {
    bunGlobal.$ = stub;
    return;
  }

  (globalThis as any).Bun = { $: stub };
}

describe("refactor_astgrep tool wrapper", () => {
  it("exposes description with pattern syntax and examples", () => {
    assert.ok(refactorAstgrepTool);
    const description = String(refactorAstgrepTool.description);
    assert.ok(description.includes("$VAR"));
    assert.ok(description.includes("$$$ARGS"));
    assert.ok(description.includes("$_"));
    assert.ok(description.includes("Rename function (Python)"));
    assert.ok(description.includes("Rename function (TypeScript)"));
    assert.ok(description.includes("PATTERNS WILL NOT MATCH"));
  });

  it("defines expected argument schema", () => {
    assert.ok(refactorAstgrepTool.args.pattern);
    assert.ok(refactorAstgrepTool.args.rewrite);
    assert.ok(refactorAstgrepTool.args.lang);
    assert.ok(refactorAstgrepTool.args.path);
    assert.ok(refactorAstgrepTool.args.dryRun);
  });

  it("assembles command with defaults (dry-run)", async () => {
    setBunStub({ output: "diff" });
    const result = await refactorAstgrepTool.execute({
      pattern: "oldFunc($$$ARGS)",
      rewrite: "newFunc($$$ARGS)",
      lang: "typescript",
    });

    assert.equal(result, "diff");
    const cmd = calls[0][1] as string[];
    assert.equal(cmd[0], "ast-grep");
    assert.equal(cmd[1], "run");
    assert.ok(cmd.includes("-p"));
    assert.ok(cmd.includes("oldFunc($$$ARGS)"));
    assert.ok(cmd.includes("-r"));
    assert.ok(cmd.includes("newFunc($$$ARGS)"));
    assert.ok(cmd.includes("-l"));
    assert.ok(cmd.includes("typescript"));
    assert.ok(cmd.includes("."));
    assert.ok(!cmd.includes("--update-all"));
  });

  it("adds apply flag and custom path when dryRun is false", async () => {
    setBunStub({ output: "updated" });
    await refactorAstgrepTool.execute({
      pattern: "old_name($$$ARGS)",
      rewrite: "new_name($$$ARGS)",
      lang: "python",
      path: "src",
      dryRun: false,
    });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("src"));
    assert.ok(cmd.includes("--update-all"));
  });

  it("returns no-match message for dry-run with empty output", async () => {
    setBunStub({ output: "" });
    const result = await refactorAstgrepTool.execute({
      pattern: "oldFunc($$$ARGS)",
      rewrite: "newFunc($$$ARGS)",
      lang: "typescript",
    });

    assert.ok(result.includes("No matches found"));
  });

  it("returns no-files-modified message for apply with empty output", async () => {
    setBunStub({ output: "" });
    const result = await refactorAstgrepTool.execute({
      pattern: "oldFunc($$$ARGS)",
      rewrite: "newFunc($$$ARGS)",
      lang: "typescript",
      dryRun: false,
    });

    assert.equal(result, "No files modified (no matches).");
  });

  it("validates required arguments", async () => {
    const missingPattern = await refactorAstgrepTool.execute({});
    assert.ok(missingPattern.includes("pattern is required"));

    const missingRewrite = await refactorAstgrepTool.execute({ pattern: "$A", lang: "python" });
    assert.ok(missingRewrite.includes("rewrite is required"));

    const missingLang = await refactorAstgrepTool.execute({ pattern: "$A", rewrite: "$A" });
    assert.ok(missingLang.includes("lang is required"));
  });

  it("returns missing-binary hint for ENOENT", async () => {
    const error = Object.assign(new Error("ENOENT: ast-grep"), {
      stdout: "",
      stderr: "ENOENT: ast-grep not found",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });
    assert.ok(result.includes("ERROR [MISSING_BINARY]"));
    assert.ok(result.includes("ast-grep CLI not found"));
    assert.ok(result.includes("Stderr: ENOENT: ast-grep not found"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("returns invalid pattern message from stderr", async () => {
    const error = Object.assign(new Error("parse error"), {
      stdout: "",
      stderr: "Parse error: expected pattern",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });
    assert.ok(result.includes("ERROR [INVALID_PATTERN]"));
    assert.ok(result.includes("Stderr: Parse error: expected pattern"));
  });

  it("returns validation error for unsupported language", async () => {
    setBunStub({ output: "" });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "haskell",
    });

    assert.ok(result.includes("lang must be one of"));
    assert.equal(calls.length, 0);
  });

  it("returns stdout when command fails with stdout output", async () => {
    const error = Object.assign(new Error("command failed"), {
      stdout: "diff preview",
      stderr: "Some error",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });

    assert.equal(result, "diff preview");
  });

  it("returns generic stderr failure message", async () => {
    const error = Object.assign(new Error("execution failed"), {
      stdout: "",
      stderr: "unexpected failure",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });

    assert.ok(result.includes("ERROR [EXECUTION]"));
    assert.ok(result.includes("Stderr: unexpected failure"));
  });

  it("renders empty stderr for missing binary", async () => {
    const error = Object.assign(new Error("ENOENT: ast-grep"), {
      stdout: "",
      stderr: "",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });

    assert.ok(result.includes("ERROR [MISSING_BINARY]"));
    assert.ok(result.includes("Stderr: (empty)"));
  });

  it("renders empty stderr for invalid pattern", async () => {
    const error = Object.assign(new Error("parse error"), {
      stdout: "",
      stderr: "",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });

    assert.ok(result.includes("ERROR [INVALID_PATTERN]"));
    assert.ok(result.includes("Stderr: (empty)"));
  });

  it("renders empty stderr for execution failures", async () => {
    const error = Object.assign(new Error("execution failed"), {
      stdout: "",
      stderr: "",
    });
    setBunStub({ error });

    const result = await refactorAstgrepTool.execute({
      pattern: "$A",
      rewrite: "$A",
      lang: "python",
    });

    assert.ok(result.includes("ERROR [EXECUTION]"));
    assert.ok(result.includes("Stderr: (empty)"));
  });
});
