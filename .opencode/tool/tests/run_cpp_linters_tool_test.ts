import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const runCppLintersTool = (await import("../run_cpp_linters")).default;
const MISSING_HINT =
  "Missing backing script .opencode/tool/run_cpp_linters.py (dependency #1365).";

const calls: any[] = [];

function setBunStub({ output = "", error }: { output?: string; error?: any }) {
  calls.length = 0;
  (globalThis as any).Bun = {
    $: (...args: any[]) => {
      calls.push(args);
      if (error) {
        throw error;
      }
      return {
        text: async () => output,
      };
    },
  };
}

describe("run_cpp_linters tool wrapper", () => {
  it("exposes rich description with examples and linters", () => {
    assert.ok(runCppLintersTool);
    const description = String(runCppLintersTool.description);
    assert.ok(description.includes("clang-format"));
    assert.ok(description.includes("clang-tidy"));
    assert.ok(description.includes("cppcheck"));
    assert.ok(description.includes("EXAMPLES"));
    assert.ok(description.includes("IMPORTANT"));
  });

  it("defines expected argument schema", () => {
    assert.ok(runCppLintersTool.args.outputMode);
    assert.ok(runCppLintersTool.args.sourceDir);
    assert.ok(runCppLintersTool.args.buildDir);
    assert.ok(runCppLintersTool.args.linters);
    assert.ok(runCppLintersTool.args.autoFix);
    assert.ok(runCppLintersTool.args.timeout);
  });

  it("assembles command with defaults", async () => {
    setBunStub({ output: "ok" });
    const result = await runCppLintersTool.execute({ sourceDir: "src" });

    assert.equal(result, "ok");
    const cmd = calls[0][1] as string[];
    assert.equal(cmd[0], "python3");
    assert.ok(String(cmd[1]).endsWith("/run_cpp_linters.py"));
    assert.ok(cmd.includes("--source-dir=src"));
    assert.ok(cmd.includes("--output=summary"));
    assert.ok(cmd.includes("--timeout=300"));
    assert.ok(cmd.includes("--linters=clang-format,clang-tidy,cppcheck"));
    assert.ok(!cmd.includes("--build-dir"));
    assert.ok(!cmd.includes("--auto-fix"));
  });

  it("adds optional flags when provided", async () => {
    setBunStub({ output: "ok" });
    await runCppLintersTool.execute({
      sourceDir: "src",
      buildDir: "build",
      autoFix: true,
      linters: ["clang-tidy"],
      outputMode: "json",
      timeout: 120,
    });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("--build-dir=build"));
    assert.ok(cmd.includes("--auto-fix"));
    assert.ok(cmd.includes("--output=json"));
    assert.ok(cmd.includes("--timeout=120"));
    assert.ok(cmd.includes("--linters=clang-tidy"));
  });

  it("returns default message when output is empty", async () => {
    setBunStub({ output: "" });
    const result = await runCppLintersTool.execute({ sourceDir: "src" });

    assert.equal(result, "C++ linters completed but returned no output.");
  });

  it("validates required sourceDir", async () => {
    const result = await runCppLintersTool.execute({});
    assert.ok(result.includes("sourceDir is required"));
  });

  it("validates positive timeout", async () => {
    const result = await runCppLintersTool.execute({ sourceDir: "src", timeout: 0 });
    assert.ok(result.includes("Timeout must be positive"));
  });

  it("prefers stdout on errors", async () => {
    const error = Object.assign(new Error("boom"), {
      stdout: "stdout-first",
      stderr: "stderr-second",
    });
    setBunStub({ error });

    const result = await runCppLintersTool.execute({ sourceDir: "src" });
    assert.equal(result, "stdout-first");
  });

  it("returns stderr with hint for ENOENT", async () => {
    const error = Object.assign(new Error("ENOENT"), {
      stdout: "",
      stderr: "ENOENT: missing script",
    });
    setBunStub({ error });

    const result = await runCppLintersTool.execute({ sourceDir: "src" });
    assert.ok(result.startsWith("ERROR: C++ linters failed"));
    assert.ok(result.includes("ENOENT"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("returns stderr without hint when not ENOENT", async () => {
    const error = Object.assign(new Error("boom"), {
      stdout: "",
      stderr: "linter failure",
    });
    setBunStub({ error });

    const result = await runCppLintersTool.execute({ sourceDir: "src" });
    assert.ok(result.startsWith("ERROR: C++ linters failed"));
    assert.ok(result.includes("linter failure"));
    assert.ok(!result.includes(MISSING_HINT));
  });

  it("returns ENOENT hint when only message is available", async () => {
    const error = Object.assign(new Error("ENOENT: missing script"), {
      stdout: "",
      stderr: "",
    });
    setBunStub({ error });

    const result = await runCppLintersTool.execute({ sourceDir: "src" });
    assert.ok(result.startsWith("ERROR: Failed to run C++ linters"));
    assert.ok(result.includes("ENOENT"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("returns generic failure when no output is available", async () => {
    const error = new Error("unexpected failure");
    setBunStub({ error });

    const result = await runCppLintersTool.execute({ sourceDir: "src" });
    assert.equal(result, "ERROR: Failed to run C++ linters: unexpected failure");
  });
});
