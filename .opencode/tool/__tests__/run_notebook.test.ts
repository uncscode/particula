import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it, mock } from "bun:test";

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
const bunGlobal = (globalThis as any).Bun as { $?: unknown } | undefined;
const originalDollar = bunGlobal?.$;

const setExecutionSuccess = (output?: string) => {
  executionBehavior = { kind: "success", output };
};

const mockBunDollar = () => {
  const bun = (globalThis as any).Bun as { $?: unknown } | undefined;
  const replacement = (_strings: TemplateStringsArray, ...values: unknown[]) => {
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
  };

  if (!bun) {
    (globalThis as any).Bun = { $: replacement };
    return;
  }

  try {
    bun.$ = replacement;
    return;
  } catch (error) {
    try {
      Object.defineProperty(bun, "$", {
        value: replacement,
        configurable: true,
        writable: true,
      });
      return;
    } catch (innerError) {
      throw error ?? innerError;
    }
  }
};

const restoreBunDollar = () => {
  const bun = (globalThis as any).Bun as { $?: unknown } | undefined;
  if (!bun) {
    return;
  }

  if (originalDollar === undefined) {
    return;
  }

  try {
    bun.$ = originalDollar;
  } catch {
    try {
      Object.defineProperty(bun, "$", {
        value: originalDollar,
        configurable: true,
        writable: true,
      });
    } catch {
      // Ignore restore failures; tests only rely on the mocked behavior.
    }
  }
};

const runNotebookTool = (await import("../run_notebook")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("run_notebook tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
    mockBunDollar();
  });

  afterEach(() => {
    restoreBunDollar();
  });

  it("includes script usage in description examples", () => {
    assert.ok(runNotebookTool.description.includes("EXAMPLES"));
    assert.ok(runNotebookTool.description.includes(".py"));
    assert.ok(runNotebookTool.description.includes("script"));
  });

  it("defines script argument with descriptive help text", () => {
    const { args } = runNotebookTool;
    assert.ok(args.script);
    assert.ok(String(args.script.description || "").includes("Single .py files"));
  });

  it("adds --script when script is true", async () => {
    await runNotebookTool.execute({ notebookPath: "examples/script.py", script: true });

    assert.ok(lastCommand.includes("examples/script.py"));
    assert.ok(lastCommand.includes("--script"));
  });

  it("omits --script when script is not set", async () => {
    await runNotebookTool.execute({ notebookPath: "examples/script.py" });

    assert.ok(lastCommand.includes("examples/script.py"));
    assert.ok(!lastCommand.includes("--script"));
  });
});
