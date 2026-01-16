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
  array: (value: unknown) => createSchema("array", { value }),
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

const runSanitizersTool = (await import("../run_sanitizers")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

const MISSING_HINT = "Missing backing script .opencode/tool/run_sanitizers.py";

describe("run_sanitizers tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
  });

  it("includes key description notes and examples", () => {
    assert.ok(runSanitizersTool.description.toLowerCase().includes("sanitizer"));
    assert.ok(runSanitizersTool.description.includes("EXAMPLES"));
    assert.ok(runSanitizersTool.description.includes("2–10x"));
  });

  it("defines expected arguments with enums and required buildDir", () => {
    const { args } = runSanitizersTool;
    assert.ok(args.outputMode);
    assert.deepEqual(args.outputMode.values, ["summary", "full", "json"]);
    assert.ok(args.buildDir);
    assert.ok(args.executable);
    assert.ok(args.sanitizer);
    assert.deepEqual(args.sanitizer.values, ["asan", "tsan", "ubsan"]);
  });

  it("builds command with defaults when optional args omitted", async () => {
    const result = await runSanitizersTool.execute({
      buildDir: "build",
      executable: "./a.out",
      sanitizer: "asan",
    });

    assert.ok(result.length > 0);
    assert.ok(lastCommand.includes("python3"));
    assert.ok(lastCommand.includes("--build-dir=build"));
    assert.ok(lastCommand.includes("--executable=./a.out"));
    assert.ok(lastCommand.includes("--sanitizer=asan"));
    assert.ok(lastCommand.includes("--output-mode=summary"));
    assert.ok(lastCommand.includes("--timeout=600"));
  });

  it("rejects missing buildDir before executing", async () => {
    const result = await runSanitizersTool.execute({ executable: "./a.out", sanitizer: "asan" });

    assert.ok(result.toLowerCase().includes("builddir is required"));
    assert.equal(lastCommand.length, 0);
  });

  it("adds missing-script hint when stderr contains ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "ENOENT run_sanitizers.py missing", message: "ENOENT" });
    const result = await runSanitizersTool.execute({ buildDir: "b", executable: "./a.out", sanitizer: "asan" });

    assert.ok(result.includes(MISSING_HINT));
  });
});
