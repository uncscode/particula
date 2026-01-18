import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const runSanitizersTool = (await import("../run_sanitizers")).default;
const MISSING_HINT = "Missing backing script .opencode/tool/run_sanitizers.py (dependency #1379).";

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

describe("run_sanitizers tool wrapper", () => {
  it("exposes rich description with sanitizer notes and examples", () => {
    assert.ok(runSanitizersTool);
    const description = String(runSanitizersTool.description);
    assert.ok(description.toLowerCase().includes("asan"));
    assert.ok(description.toLowerCase().includes("tsan"));
    assert.ok(description.toLowerCase().includes("ubsan"));
    assert.ok(description.includes("EXAMPLES"));
    assert.ok(description.includes("2–10x"));
    assert.ok(description.toLowerCase().includes("msan"));
  });

  it("defines expected argument schema", () => {
    assert.ok(runSanitizersTool.args.outputMode);
    assert.ok(runSanitizersTool.args.buildDir);
    assert.ok(runSanitizersTool.args.executable);
    assert.ok(runSanitizersTool.args.sanitizer);
    assert.ok(runSanitizersTool.args.timeout);
    assert.ok(runSanitizersTool.args.suppressions);
    assert.ok(runSanitizersTool.args.options);
    assert.ok(runSanitizersTool.args.normalDuration);
    assert.ok(runSanitizersTool.args.extraArgs);
  });

  it("assembles command with defaults and no stray separator", async () => {
    setBunStub({ output: "ok" });
    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });

    assert.equal(result, "ok");
    const cmd = calls[0][1] as string[];
    assert.equal(cmd[0], "python3");
    assert.ok(String(cmd[1]).endsWith("/run_sanitizers.py"));
    assert.ok(cmd.includes("--build-dir=build"));
    assert.ok(cmd.includes("--executable=./a.out"));
    assert.ok(cmd.includes("--sanitizer=asan"));
    assert.ok(cmd.includes("--output-mode=summary"));
    assert.ok(cmd.includes("--timeout=600"));
    assert.ok(!cmd.includes("--suppressions="));
    assert.ok(!cmd.includes("--options="));
    assert.ok(!cmd.includes("--normal-duration="));
    assert.ok(!cmd.includes("--"));
  });

  it("adds optional flags and extra args when provided", async () => {
    setBunStub({ output: "ok" });
    await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./race",
      sanitizer: "tsan",
      suppressions: "./supp.txt",
      options: "halt_on_error=1",
      normalDuration: 1.5,
      outputMode: "json",
      timeout: 42,
      extraArgs: ["--flag", "value"],
    });

    const cmd = calls[0][1] as string[];
    assert.ok(cmd.includes("--suppressions=./supp.txt"));
    assert.ok(cmd.includes("--options=halt_on_error=1"));
    assert.ok(cmd.includes("--normal-duration=1.5"));
    assert.ok(cmd.includes("--output-mode=json"));
    assert.ok(cmd.includes("--timeout=42"));
    const separatorIndex = cmd.indexOf("--");
    assert.ok(separatorIndex > 0);
    assert.equal(cmd[separatorIndex + 1], "--flag");
    assert.equal(cmd[separatorIndex + 2], "value");
  });

  it("returns default message when output is empty", async () => {
    setBunStub({ output: "" });
    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "ubsan",
    });

    assert.equal(result, "Sanitizer run completed but returned no output.");
  });

  it("validates required buildDir", async () => {
    const result = await runSanitizersTool.execute({ executable: "./a.out", sanitizer: "asan" });
    assert.ok(result.includes("buildDir is required"));
  });

  it("validates required executable", async () => {
    const result = await runSanitizersTool.execute({ buildDir: "build", sanitizer: "asan" });
    assert.ok(result.includes("executable is required"));
  });

  it("validates required sanitizer", async () => {
    const result = await runSanitizersTool.execute({ buildDir: "build", executable: "./a.out" });
    assert.ok(result.includes("sanitizer is required"));
  });

  it("validates positive timeout and normalDuration", async () => {
    const timeoutResult = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
      timeout: 0,
    });
    assert.ok(timeoutResult.includes("Timeout must be positive"));

    const durationResult = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
      normalDuration: 0,
    });
    assert.ok(durationResult.includes("normalDuration must be positive"));
  });

  it("validates sanitizer and outputMode enums", async () => {
    const sanitizerResult = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "msan" as any,
    });
    assert.ok(sanitizerResult.includes("sanitizer must be one of"));

    const outputResult = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
      outputMode: "detailed" as any,
    });
    assert.ok(outputResult.includes("outputMode must be one of"));
  });

  it("prefers stdout on errors", async () => {
    const error = Object.assign(new Error("boom"), {
      stdout: "stdout-first",
      stderr: "stderr-second",
    });
    setBunStub({ error });

    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });
    assert.equal(result, "stdout-first");
  });

  it("returns stderr with hint for ENOENT", async () => {
    const error = Object.assign(new Error("ENOENT"), {
      stdout: "",
      stderr: "ENOENT: missing script",
    });
    setBunStub({ error });

    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });
    assert.ok(result.startsWith("ERROR: Sanitizer run failed"));
    assert.ok(result.includes("ENOENT"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("returns stderr without hint when not ENOENT", async () => {
    const error = Object.assign(new Error("boom"), {
      stdout: "",
      stderr: "sanitizer failure",
    });
    setBunStub({ error });

    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });
    assert.ok(result.startsWith("ERROR: Sanitizer run failed"));
    assert.ok(result.includes("sanitizer failure"));
    assert.ok(!result.includes(MISSING_HINT));
  });

  it("returns ENOENT hint when only message is available", async () => {
    const error = Object.assign(new Error("ENOENT: missing script"), {
      stdout: "",
      stderr: "",
    });
    setBunStub({ error });

    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });
    assert.ok(result.startsWith("ERROR: Failed to run sanitizer"));
    assert.ok(result.includes("ENOENT"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("returns generic failure when no output is available", async () => {
    const error = new Error("unexpected failure");
    setBunStub({ error });

    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });
    assert.equal(result, "ERROR: Failed to run sanitizer: unexpected failure");
  });
});
