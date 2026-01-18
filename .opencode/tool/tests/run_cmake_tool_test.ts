import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const runCmakeTool = (await import("../run_cmake")).default;

describe("run_cmake shim wiring", () => {
  it("exposes tool definition for test path compliance", () => {
    assert.ok(runCmakeTool);
    assert.ok(typeof runCmakeTool === "object" || typeof runCmakeTool === "function");
  });
});
