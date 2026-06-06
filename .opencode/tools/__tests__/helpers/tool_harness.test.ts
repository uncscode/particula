import { describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  loadToolExecute,
  loadToolExecuteFromAbsolutePath,
  resetCapturedToolDefinition,
} from "./tool_harness";

describe("tool_harness helper", () => {
  it("loads execute handler from wrapper module path", async () => {
    const execute = await loadToolExecute("../../run_pytest_basic.ts");

    expect(typeof execute).toBe("function");
  });

  it("loads execute handler from absolute wrapper module path", async () => {
    const execute = await loadToolExecuteFromAbsolutePath(
      join(import.meta.dir, "..", "..", "run_pytest_basic.ts"),
    );

    expect(typeof execute).toBe("function");
  });

  it("throws deterministic assertion error when module registers no tool handler", async () => {
    const fixtureRoot = mkdtempSync(join(tmpdir(), "tool-harness-fixture-"));
    const noToolModulePath = join(fixtureRoot, "no_tool_module.ts");
    writeFileSync(noToolModulePath, "export const value = 1;\n", "utf8");

    resetCapturedToolDefinition();

    try {
      await expect(loadToolExecuteFromAbsolutePath(noToolModulePath)).rejects.toThrow(
        "ASSERT: no tool.execute handler registered",
      );
    } finally {
      rmSync(fixtureRoot, { recursive: true, force: true });
    }
  });
});
