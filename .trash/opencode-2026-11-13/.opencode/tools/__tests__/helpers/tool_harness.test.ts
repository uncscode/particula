import { describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  assertCountedAndExemptFields,
  assertPublicSchemaIncludesKeys,
  assertPublicSchemaOmitsKeys,
  getCapturedToolDefinition,
  getPublicSchemaKeys,
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

  it("reads captured public schema keys and supports inclusion/omission assertions", async () => {
    await loadToolExecute("../../find_files.ts");
    const definition = getCapturedToolDefinition();

    expect(getPublicSchemaKeys(definition)).toEqual(["options", "path", "pattern"]);
    expect(() => assertPublicSchemaIncludesKeys(definition, ["pattern", "options"])).not.toThrow();
    expect(() =>
      assertPublicSchemaOmitsKeys(definition, ["contentPattern", "filesWithMatches"]),
    ).not.toThrow();
  });

  it("asserts counted and exempt field expectations against the captured schema", async () => {
    await loadToolExecute("../../run_validate_agent_references.ts");

    expect(() =>
      assertCountedAndExemptFields(getCapturedToolDefinition(), {
        counted: ["cwd"],
        exempt: ["baselinePath"],
        actualCounted: ["cwd"],
        actualExempt: ["baselinePath"],
      }),
    ).not.toThrow();
  });

  it("fails when counted and exempt category assignment shifts despite matching union", async () => {
    await loadToolExecute("../../find_files.ts");

    expect(() =>
      assertCountedAndExemptFields(getCapturedToolDefinition(), {
        counted: ["pattern", "path"],
        exempt: ["options"],
        actualCounted: ["options", "path", "pattern"],
        actualExempt: [],
      }),
    ).toThrow("ASSERT: counted field classification did not match expectation");
  });

  it("emits deterministic assertion text for schema expectation mismatches", async () => {
    await loadToolExecute("../../find_files.ts");
    const definition = getCapturedToolDefinition();

    expect(() => assertPublicSchemaIncludesKeys(definition, ["pattern", "contentPattern"])).toThrow(
      "ASSERT: public schema missing expected keys: contentPattern",
    );
    expect(() => assertCountedAndExemptFields(definition, { counted: ["pattern"], exempt: [] })).toThrow(
      "ASSERT: public schema keys did not match counted+exempt expectations",
    );
  });

  it("throws deterministic assertion text for malformed tool args schemas", () => {
    expect(() => getPublicSchemaKeys({ args: undefined })).toThrow(
      "ASSERT: tool.args schema must be an object-like record",
    );
  });
});
