import { afterEach, describe, expect, it, spyOn } from "bun:test";
import * as fs from "node:fs";
import { join } from "node:path";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
import {
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

describe("move compatibility wrapper", () => {
  const originalCwd = process.cwd();

  afterEach(() => {
    process.chdir(originalCwd);
    resetCapturedToolDefinition();
  });

  it("requires destination for default safe routing", async () => {
    const execute = await loadToolExecute("../../move.ts");
    const result = await execute({ source: "a.txt" });

    assertErrorPrefix(String(result), "ERROR [INVALID_DESTINATION]:");
  });

  it("requires destination for overwrite routing", async () => {
    const execute = await loadToolExecute("../../move.ts");
    const result = await execute({ source: "a.txt", overwrite: true });

    assertErrorPrefix(String(result), "ERROR [INVALID_DESTINATION]:");
  });

  it("keeps trash routing available through the public compatibility wrapper without destination", async () => {
    const execute = await loadToolExecute("../../move.ts");
    const definition = getCapturedToolDefinition();
    const source = join(import.meta.dir, "move.test.ts");
    const mkdirSpy = spyOn(fs, "mkdirSync").mockImplementation(() => undefined as any);
    const renameSpy = spyOn(fs, "renameSync").mockImplementation(() => undefined as any);

    expect(definition?.args).toHaveProperty("trash");
    expect(definition?.args).toHaveProperty("destination");

    const result = await execute({ source, trash: true });

    expect(result).toContain("SUCCESS: Moved file to trash");
    expect(renameSpy).toHaveBeenCalledTimes(1);
    mkdirSpy.mockRestore();
    renameSpy.mockRestore();
  });

  it("allows trash routing without destination", async () => {
    const execute = await loadToolExecute("../../move.ts");
    const source = join(import.meta.dir, "move.test.ts");
    const mkdirSpy = spyOn(fs, "mkdirSync").mockImplementation(() => undefined as any);
    const renameSpy = spyOn(fs, "renameSync").mockImplementation(() => undefined as any);

    const result = await execute({ source, trash: true });

    expect(result).toContain("SUCCESS: Moved file to trash");
    expect(mkdirSpy).toHaveBeenCalled();
    expect(renameSpy).toHaveBeenCalled();
    mkdirSpy.mockRestore();
    renameSpy.mockRestore();
  });

  it("gives trash precedence over overwrite", async () => {
    const execute = await loadToolExecute("../../move.ts");
    const source = join(import.meta.dir, "move.test.ts");
    const mkdirSpy = spyOn(fs, "mkdirSync").mockImplementation(() => undefined as any);
    const renameSpy = spyOn(fs, "renameSync").mockImplementation(() => undefined as any);

    const result = await execute({ source, overwrite: true, trash: true });

    expect(result).toContain("SUCCESS: Moved file to trash");
    expect(renameSpy).toHaveBeenCalledTimes(1);
    mkdirSpy.mockRestore();
    renameSpy.mockRestore();
  });
});
