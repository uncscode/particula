import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const clearBuildTool = (await import("../clear_build")).default;
const MISSING_HINT =
  "Missing backing script .opencode/tool/clear_build.py (dependency #1354).";

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

describe("clear_build tool wrapper", () => {
  it("exposes registration and safety messaging", () => {
    assert.ok(clearBuildTool);
    assert.ok(clearBuildTool.description.includes("WARNING"));
    assert.ok(clearBuildTool.description.toLowerCase().includes("permanently deletes"));
    assert.ok(clearBuildTool.description.toLowerCase().includes("dryrun"));
    assert.ok(clearBuildTool.description.includes("SAFETY FEATURES"));
    assert.ok(clearBuildTool.description.includes("EXAMPLES"));
  });

  it("defines expected argument schema", () => {
    assert.ok(clearBuildTool.args.buildDir);
    assert.ok(clearBuildTool.args.dryRun);
    assert.ok(clearBuildTool.args.force);
  });

  it("wires default command without flags", async () => {
    setBunStub({ output: "ok" });
    const result = await clearBuildTool.execute({});

    assert.equal(result, "ok");
    const cmd = calls[0][1];
    assert.equal(cmd[0], "python3");
    assert.ok(String(cmd[1]).endsWith("/clear_build.py"));
    assert.ok(cmd.includes("--build-dir=build"));
    assert.ok(!cmd.includes("--dry-run"));
    assert.ok(!cmd.includes("--force"));
  });

  it("returns default message when output is empty", async () => {
    setBunStub({ output: "" });
    const result = await clearBuildTool.execute({});

    assert.equal(result, "Clear build completed but returned no output.");
  });

  it("adds only --dry-run when dryRun is true", async () => {
    setBunStub({ output: "dry-run" });
    await clearBuildTool.execute({ dryRun: true });

    const cmd = calls[0][1];
    assert.ok(cmd.includes("--build-dir=build"));
    assert.ok(cmd.includes("--dry-run"));
    assert.ok(!cmd.includes("--force"));
  });

  it("adds only --force when force is true", async () => {
    setBunStub({ output: "force" });
    await clearBuildTool.execute({ force: true });

    const cmd = calls[0][1];
    assert.ok(cmd.includes("--build-dir=build"));
    assert.ok(!cmd.includes("--dry-run"));
    assert.ok(cmd.includes("--force"));
  });

  it("adds both flags when dryRun and force are true", async () => {
    setBunStub({ output: "both" });
    await clearBuildTool.execute({ dryRun: true, force: true });

    const cmd = calls[0][1];
    assert.ok(cmd.includes("--dry-run"));
    assert.ok(cmd.includes("--force"));
  });

  it("passes through buildDir override", async () => {
    setBunStub({ output: "override" });
    await clearBuildTool.execute({ buildDir: "out" });

    const cmd = calls[0][1];
    assert.ok(cmd.includes("--build-dir=out"));
  });

  it("prefers stdout on errors", async () => {
    const error = Object.assign(new Error("boom"), {
      stdout: "stdout-first",
      stderr: "stderr-second",
    });
    setBunStub({ error });

    const result = await clearBuildTool.execute({});
    assert.equal(result, "stdout-first");
  });

  it("returns stderr message when error lacks ENOENT hint", async () => {
    const error = Object.assign(new Error("failed"), {
      stdout: "",
      stderr: "validation failed",
    });
    setBunStub({ error });

    const result = await clearBuildTool.execute({});
    assert.ok(result.startsWith("ERROR: Clear build failed"));
    assert.ok(result.includes("validation failed"));
    assert.ok(!result.includes(MISSING_HINT));
  });

  it("adds missing-script hint for ENOENT without stdout or stderr", async () => {
    const error = new Error("ENOENT: missing script");
    setBunStub({ error });

    const result = await clearBuildTool.execute({});
    assert.ok(result.includes("ENOENT"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("adds missing-script hint for ENOENT with stderr", async () => {
    const error = Object.assign(new Error("ENOENT"), {
      stdout: "",
      stderr: "ENOENT: cannot find file",
    });
    setBunStub({ error });

    const result = await clearBuildTool.execute({});
    assert.ok(result.includes("ENOENT"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("returns generic failure when no output is available", async () => {
    const error = new Error("unexpected failure");
    setBunStub({ error });

    const result = await clearBuildTool.execute({});
    assert.equal(result, "ERROR: Failed to run clear_build: unexpected failure");
  });
});
