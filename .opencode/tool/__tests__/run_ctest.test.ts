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

const runCtestTool = (await import("../run_ctest")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

const MISSING_HINT = "Missing backing script .opencode/tool/run_ctest.py";

describe("run_ctest tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
  });

  it("includes key description notes and examples", () => {
    assert.ok(runCtestTool.description.includes("CTest"));
    assert.ok(runCtestTool.description.includes("EXAMPLES"));
    assert.ok(runCtestTool.description.includes("IMPORTANT"));
    assert.ok(runCtestTool.description.includes("CTestTestfile"));
  });

  it("defines all expected arguments with enum values and required buildDir", () => {
    const { args } = runCtestTool;
    assert.ok(args.outputMode);
    assert.deepEqual(args.outputMode.values, ["summary", "full", "json"]);
    assert.ok(args.buildDir);
    assert.ok(args.testFilter);
    assert.ok(args.excludeFilter);
    assert.ok(args.parallel);
    assert.ok(String(args.parallel.description || "").toLowerCase().includes("positive"));
    assert.ok(args.timeout);
    assert.ok(String(args.timeout.description || "").toLowerCase().includes("positive"));
    assert.ok(args.minTests);
    assert.ok(String(args.minTests.description || "").toLowerCase().includes("positive"));
  });

  it("builds command with defaults when optional args omitted", async () => {
    const result = await runCtestTool.execute({ buildDir: "build" });

    assert.ok(result.length > 0);
    assert.ok(lastCommand.includes("python3"));
    assert.ok(lastCommand.includes("--build-dir=build"));
    assert.ok(lastCommand.includes("--output=summary"));
    assert.ok(lastCommand.includes("--min-tests=1"));
    assert.ok(lastCommand.includes("--timeout=300"));
    assert.ok(!lastCommand.includes("-R"));
    assert.ok(!lastCommand.includes("-E"));
    assert.ok(!lastCommand.includes("-j"));
  });

  it("maps filters, parallelism, and overrides into the command", async () => {
    await runCtestTool.execute({
      buildDir: "b",
      testFilter: "include",
      excludeFilter: "skip",
      parallel: 4,
      outputMode: "json",
      minTests: 5,
      timeout: 42,
    });

    assert.ok(lastCommand.includes("--build-dir=b"));
    assert.ok(lastCommand.includes("--output=json"));
    assert.ok(lastCommand.includes("--min-tests=5"));
    assert.ok(lastCommand.includes("--timeout=42"));
    assert.ok(lastCommand.includes("-R"));
    assert.ok(lastCommand.includes("include"));
    assert.ok(lastCommand.includes("-E"));
    assert.ok(lastCommand.includes("skip"));
    assert.ok(lastCommand.includes("-j"));
    assert.ok(lastCommand.includes("4"));
  });

  it("returns fallback message when command output is empty", async () => {
    setExecutionSuccess("");
    const result = await runCtestTool.execute({ buildDir: "build" });

    assert.equal(result, "CTest completed but returned no output.");
    assert.ok(lastCommand.includes("--build-dir=build"));
  });

  it("rejects missing buildDir before executing", async () => {
    const result = await runCtestTool.execute({});

    assert.ok(result.toLowerCase().includes("builddir is required"));
    assert.equal(lastCommand.length, 0);
  });

  it("rejects non-positive timeout and minTests before executing", async () => {
    const timeoutResult = await runCtestTool.execute({ buildDir: "b", timeout: 0 });
    assert.ok(timeoutResult.includes("Timeout must be positive"));
    assert.equal(lastCommand.length, 0);

    lastCommand = [];
    const minTestsResult = await runCtestTool.execute({ buildDir: "b", minTests: 0 });
    assert.ok(minTestsResult.includes("minTests must be positive"));
    assert.equal(lastCommand.length, 0);

    lastCommand = [];
    const parallelZeroResult = await runCtestTool.execute({ buildDir: "b", parallel: 0 });
    assert.ok(parallelZeroResult.includes("parallel must be positive"));
    assert.equal(lastCommand.length, 0);

    lastCommand = [];
    const parallelNegativeResult = await runCtestTool.execute({ buildDir: "b", parallel: -1 });
    assert.ok(parallelNegativeResult.includes("parallel must be positive"));
    assert.equal(lastCommand.length, 0);
  });

  it("prefers stdout over stderr and message on errors", async () => {
    setExecutionError({ stdout: "from-stdout", stderr: "from-stderr", message: "msg" });
    const result = await runCtestTool.execute({ buildDir: "b" });

    assert.equal(result, "from-stdout");
  });

  it("uses stderr with prefix when stdout is empty", async () => {
    setExecutionError({ stdout: "", stderr: "from-stderr", message: "msg" });
    const result = await runCtestTool.execute({ buildDir: "b" });

    assert.ok(result.startsWith("ERROR: CTest failed"));
    assert.ok(result.includes("from-stderr"));
  });

  it("falls back to message and adds hint on ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "ENOENT run_ctest.py missing" });
    const result = await runCtestTool.execute({ buildDir: "b" });

    assert.ok(result.includes("Failed to run CTest"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("falls back to message when no stdout or stderr is provided", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "generic failure" });
    const result = await runCtestTool.execute({ buildDir: "b" });

    assert.ok(result.includes("ERROR: Failed to run CTest: generic failure"));
    assert.ok(!result.includes(MISSING_HINT));
  });

  it("adds missing-script hint when stderr contains ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "ENOENT run_ctest.py missing", message: "ENOENT" });
    const result = await runCtestTool.execute({ buildDir: "b" });

    assert.ok(result.includes(MISSING_HINT));
  });
});
