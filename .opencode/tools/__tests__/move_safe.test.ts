import { afterEach, describe, expect, it, spyOn } from "bun:test";
import * as fs from "node:fs";
import { join } from "node:path";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("move_safe wrapper", () => {
  const originalCwd = process.cwd();

  afterEach(() => {
    process.chdir(originalCwd);
    resetCapturedToolDefinition();
  });

  it("rejects missing source/destination", async () => {
    const execute = await loadToolExecute("../../move_safe.ts");
    assertErrorPrefix(String(await execute({ source: "", destination: "x" })), "ERROR [INVALID_SOURCE]:");
    assertErrorPrefix(String(await execute({ source: "a", destination: "" })), "ERROR [INVALID_DESTINATION]:");
  });

  it("rejects same path deterministically", async () => {
    const execute = await loadToolExecute("../../move_safe.ts");
    expect(await execute({ source: "a.txt", destination: "a.txt" })).toContain("SAME_PATH");
  });

  it("returns SOURCE_NOT_FOUND when source does not exist", async () => {
    const execute = await loadToolExecute("../../move_safe.ts");
    expect(await execute({ source: "missing.txt", destination: "dst/missing.txt" })).toContain(
      "SOURCE_NOT_FOUND",
    );
  });

  it("rejects destination outside repository boundary", async () => {
    const execute = await loadToolExecute("../../move_safe.ts");
    const source = join(import.meta.dir, "move_safe.test.ts");

    const result = await execute({
      source,
      destination: "../../../outside.txt",
    });
    expect(result).toContain("DEST_OUTSIDE_REPO");
  });

  it("returns success marker for a valid in-repo move", async () => {
    const execute = await loadToolExecute("../../move_safe.ts");
    const source = join(import.meta.dir, "move_safe.test.ts");
    const mkdirSpy = spyOn(fs, "mkdirSync").mockImplementation(() => undefined as any);
    const renameSpy = spyOn(fs, "renameSync").mockImplementation(() => undefined as any);

    const result = await execute({
      source,
      destination: ".trash/move-safe-success-noop.txt",
    });

    expect(mkdirSpy).toHaveBeenCalled();
    expect(renameSpy).toHaveBeenCalled();
    expect(result).toContain("SUCCESS:");
    mkdirSpy.mockRestore();
    renameSpy.mockRestore();
  });
});
