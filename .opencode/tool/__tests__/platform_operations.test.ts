import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

// Provide a fake Bun.$ implementation to capture command assembly.
(globalThis as any).Bun = {
  $: (_strings: TemplateStringsArray, ...values: unknown[]) => {
    const parts = values[0] as (string | number)[];
    return {
      text: async () => parts.join(" "),
    };
  },
};

const platformOperationsTool = (await import("../platform_operations")).default as {
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("platform_operations rate-limit", () => {
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
