import assert from "node:assert/strict";
import { beforeEach, describe, it, mock } from "node:test";

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
  enum: (values: string[]) => createSchema("enum", { values }),
  string: () => createSchema("string"),
  number: () => createSchema("number"),
  boolean: () => createSchema("boolean"),
  array: (item: unknown) => createSchema("array", { item }),
};

const toolFn = (definition: unknown) => definition;
(toolFn as any).schema = schema;

mock.module("@opencode-ai/plugin", () => ({
  tool: toolFn,
}));

let executionBehavior: { kind: "success"; output?: string } | { kind: "error"; error: any } = {
  kind: "success",
};
let lastCommand: (string | number)[] = [];

const setExecutionSuccess = (output?: string) => {
  executionBehavior = { kind: "success", output };
};

const setExecutionError = (error: any) => {
  executionBehavior = { kind: "error", error };
};

(globalThis as any).Bun = {
  $: (_strings: TemplateStringsArray, ...values: unknown[]) => {
    const parts = (values[0] || []) as (string | number)[];
    lastCommand = parts;
    return {
      text: async () => {
        if (executionBehavior.kind === "error") {
          throw executionBehavior.error;
        }
        if (executionBehavior.output !== undefined) {
          return executionBehavior.output;
        }
        return parts.join(" ");
      },
    };
  },
};

const runCmakeTool = (await import("../run_cmake")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("run_cmake tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
  });

  it("includes key description notes and examples", () => {
    assert.ok(runCmakeTool.description.includes("CMake"));
    assert.ok(runCmakeTool.description.includes("EXAMPLES"));
    assert.ok(runCmakeTool.description.includes("IMPORTANT"));
    assert.ok(runCmakeTool.description.includes("Ninja"));
    assert.ok(runCmakeTool.description.includes("Dependency"));
  });

  it("defines all expected arguments with enum values and timeout guidance", () => {
    const { args } = runCmakeTool;
    assert.ok(args.outputMode);
    assert.deepEqual(args.outputMode.values, ["summary", "full", "json"]);
    assert.ok(args.preset);
    assert.ok(args.sourceDir);
    assert.ok(args.buildDir);
    assert.ok(args.ninja);
    assert.ok(args.timeout);
    assert.ok(String(args.timeout.description || "").toLowerCase().includes("positive"));
    assert.ok(args.cmakeArgs);
  });

  it("builds manual command with defaults and cmakeArgs passthrough", async () => {
    const result = await runCmakeTool.execute({
      sourceDir: "src",
      buildDir: "build-dir",
      ninja: true,
      cmakeArgs: ["-DTEST=ON"],
      outputMode: "json",
      timeout: 42,
    });

    assert.ok(result.includes("python3"));
    assert.ok(lastCommand.includes("--output=json"));
    assert.ok(lastCommand.includes("--timeout=42"));
    assert.ok(lastCommand.includes("--source-dir=src"));
    assert.ok(lastCommand.includes("--build-dir=build-dir"));
    assert.ok(lastCommand.includes("--ninja"));
    assert.ok(lastCommand.includes("-DTEST=ON"));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--preset=")));
  });

  it("uses defaults when args omitted and omits ninja when false", async () => {
    await runCmakeTool.execute({});

    assert.ok(lastCommand.includes("--output=summary"));
    assert.ok(lastCommand.includes("--timeout=300"));
    assert.ok(lastCommand.includes("--source-dir=."));
    assert.ok(lastCommand.includes("--build-dir=build"));
    assert.ok(!lastCommand.includes("--ninja"));
  });

  it("prefers preset mode and drops manual flags even when provided", async () => {
    await runCmakeTool.execute({
      preset: "debug",
      sourceDir: "ignored",
      buildDir: "ignored",
      ninja: true,
    });

    assert.ok(lastCommand.includes("--preset=debug"));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--source-dir")));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--build-dir")));
    assert.ok(!lastCommand.includes("--ninja"));
  });

  it("returns fallback message when stdout is empty", async () => {
    setExecutionSuccess("");
    const result = await runCmakeTool.execute({ preset: "debug" });

    assert.equal(result, "CMake configuration completed but returned no output.");
  });

  it("prefers stdout over stderr and message on errors", async () => {
    setExecutionError({ stdout: "from-stdout", stderr: "from-stderr", message: "msg" });
    const result = await runCmakeTool.execute({});

    assert.equal(result, "from-stdout");
  });

  it("uses stderr when stdout is empty", async () => {
    setExecutionError({ stdout: "", stderr: "from-stderr", message: "msg" });
    const result = await runCmakeTool.execute({});

    assert.ok(result.startsWith("ERROR: CMake configuration failed"));
    assert.ok(result.includes("from-stderr"));
  });

  it("falls back to message when no stdout or stderr", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "only-message" });
    const result = await runCmakeTool.execute({});

    assert.ok(result.includes("only-message"));
  });

  it("includes missing-script hint on ENOENT in stderr", async () => {
    setExecutionError({ stdout: "", stderr: "ENOENT run_cmake.py missing", message: "ENOENT" });
    const result = await runCmakeTool.execute({});

    assert.ok(result.includes("Missing backing script"));
  });

  it("includes missing-script hint when message contains ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "ENOENT run_cmake.py missing" });
    const result = await runCmakeTool.execute({});

    assert.ok(result.includes("Failed to run CMake"));
    assert.ok(result.includes("Missing backing script"));
  });

  it("rejects non-positive timeout", async () => {
    const result = await runCmakeTool.execute({ timeout: 0 });

    assert.ok(result.includes("Timeout must be positive"));
    assert.equal(lastCommand.length, 0);
  });
});
