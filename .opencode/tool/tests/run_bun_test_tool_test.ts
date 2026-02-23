import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const runBunTestTool = (await import("../run_bun_test")).default;

describe("run_bun_test shim wiring", () => {
  it("exposes tool definition for test path compliance", () => {
    assert.ok(runBunTestTool);
    assert.ok(typeof runBunTestTool === "object" || typeof runBunTestTool === "function");
  });
});
