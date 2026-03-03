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
};

const toolFn = (definition: unknown) => definition;
(toolFn as any).schema = schema;

mock.module("@opencode-ai/plugin", () => ({
  tool: toolFn,
}));

const bunGlobal = (globalThis as any).Bun as { $?: unknown } | undefined;
const originalDollar = bunGlobal?.$;

const mockBunDollar = () => {
  const bun = (globalThis as any).Bun as { $?: unknown } | undefined;
  const replacement = (_strings: TemplateStringsArray, ...values: unknown[]) => {
    const parts = (values[0] || []) as (string | number)[];
    return {
      text: async () => parts.join(" "),
    };
  };

  if (!bun) {
    (globalThis as any).Bun = { $: replacement };
    return;
  }

  try {
    bun.$ = replacement;
  } catch (error) {
    try {
      Object.defineProperty(bun, "$", {
        value: replacement,
        configurable: true,
        writable: true,
      });
    } catch (innerError) {
      throw error ?? innerError;
    }
  }
};

const restoreBunDollar = () => {
  const bun = (globalThis as any).Bun as { $?: unknown } | undefined;
  if (!bun || originalDollar === undefined) {
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

const platformOperationsTool = (await import("../platform_operations")).default as {
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("platform_operations rate-limit", () => {
  beforeEach(() => {
    mockBunDollar();
  });

  afterEach(() => {
    restoreBunDollar();
  });

  it("builds default rate-limit command without extra flags", async () => {
    const result = await platformOperationsTool.execute({ command: "rate-limit" });

    assert.equal(result, "uv run adw platform rate-limit");
  });

  it("includes output_format and prefer_scope when provided", async () => {
    const result = await platformOperationsTool.execute({
      command: "rate-limit",
      output_format: "json",
      prefer_scope: "upstream",
    });

    assert.equal(
      result,
      "uv run adw platform rate-limit --format json --prefer-scope upstream",
    );
  });
});
