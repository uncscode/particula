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

const runBunTestTool = (await import("../run_bun_test")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

const MISSING_HINT = "Missing backing script .opencode/tool/run_bun_test.py";

describe("run_bun_test tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
  });

  it("includes key description notes and examples", () => {
    assert.ok(runBunTestTool.description.toLowerCase().includes("bun test"));
    assert.ok(runBunTestTool.description.includes("EXAMPLES"));
    assert.ok(runBunTestTool.description.includes("IMPORTANT"));
  });

  it("defines all expected arguments with enum values", () => {
    const { args } = runBunTestTool;
    assert.ok(args.outputMode);
    assert.deepEqual(args.outputMode.values, ["summary", "full", "json"]);
    assert.ok(args.testPath);
    assert.ok(args.testFilter);
    assert.ok(args.timeout);
    assert.ok(String(args.timeout.description || "").toLowerCase().includes("positive"));
    assert.ok(args.minTests);
    assert.ok(String(args.minTests.description || "").toLowerCase().includes("positive"));
    assert.ok(args.cwd);
    assert.ok(args.failFast);
  });

  it("builds command with defaults when optional args omitted", async () => {
    const result = await runBunTestTool.execute({});

    assert.ok(result.length > 0);
    assert.ok(lastCommand.includes("python3"));
    assert.ok(lastCommand.includes("--output=summary"));
    assert.ok(lastCommand.includes("--min-tests=1"));
    assert.ok(lastCommand.includes("--timeout=300"));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--test-path=")));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--filter=")));
    assert.ok(!lastCommand.includes("--bail"));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--cwd=")));
  });

  it("maps testPath, filter, failFast, cwd, and overrides into the command", async () => {
    await runBunTestTool.execute({
      testPath: "__tests__/",
      testFilter: "datetime",
      failFast: true,
      cwd: "/repo",
      outputMode: "json",
      minTests: 5,
      timeout: 42,
    });

    assert.ok(lastCommand.includes("--test-path=__tests__/"));
    assert.ok(lastCommand.includes("--filter=datetime"));
    assert.ok(lastCommand.includes("--bail"));
    assert.ok(lastCommand.includes("--cwd=/repo"));
    assert.ok(lastCommand.includes("--output=json"));
    assert.ok(lastCommand.includes("--min-tests=5"));
    assert.ok(lastCommand.includes("--timeout=42"));
  });

  it("normalizes failFast string values", async () => {
    await runBunTestTool.execute({ failFast: "true" });

    assert.ok(lastCommand.includes("--bail"));
  });

  it("returns fallback message when command output is empty", async () => {
    setExecutionSuccess("");
    const result = await runBunTestTool.execute({});

    assert.equal(result, "bun test completed but returned no output.");
  });

  it("rejects non-positive timeout and minTests before executing", async () => {
    const timeoutResult = await runBunTestTool.execute({ timeout: 0 });
    assert.ok(timeoutResult.includes("Timeout must be positive"));
    assert.equal(lastCommand.length, 0);

    lastCommand = [];
    const minTestsResult = await runBunTestTool.execute({ minTests: -1 });
    assert.ok(minTestsResult.includes("minTests must be positive"));
    assert.equal(lastCommand.length, 0);
  });

  it("prefers stdout over stderr and message on errors", async () => {
    setExecutionError({ stdout: "from-stdout", stderr: "from-stderr", message: "msg" });
    const result = await runBunTestTool.execute({});

    assert.equal(result, "from-stdout");
  });

  it("uses stderr with prefix when stdout is empty", async () => {
    setExecutionError({ stdout: "", stderr: "from-stderr", message: "msg" });
    const result = await runBunTestTool.execute({});

    assert.ok(result.startsWith("ERROR: Bun test failed"));
    assert.ok(result.includes("from-stderr"));
  });

  it("falls back to message and adds hint on ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "ENOENT run_bun_test.py missing" });
    const result = await runBunTestTool.execute({});

    assert.ok(result.includes("Failed to run bun test"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("falls back to message when no stdout or stderr is provided", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "generic failure" });
    const result = await runBunTestTool.execute({});

    assert.ok(result.includes("ERROR: Failed to run bun test: generic failure"));
    assert.ok(!result.includes(MISSING_HINT));
  });

  it("adds missing-script hint when stderr contains ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "ENOENT run_bun_test.py missing", message: "ENOENT" });
    const result = await runBunTestTool.execute({});

    assert.ok(result.includes(MISSING_HINT));
  });
});
