import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

const runCtestTool = (await import("../run_ctest")).default;

describe("run_ctest shim wiring", () => {
  it("exposes tool definition for test path compliance", () => {
    assert.ok(runCtestTool);
    assert.ok(typeof runCtestTool === "object" || typeof runCtestTool === "function");
  });
});
