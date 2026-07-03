import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure, buildSuccessOutput } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("sync_notebook_pair wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds sync argv with recursive flag", async () => {
    setDollarText(buildSuccessOutput("OK"));
    const execute = await loadToolExecute("../../sync_notebook_pair.ts");
    const result = await execute({ notebookPath: " docs/Examples/demo.ipynb ", recursive: true });

    expect(result).toBe("OK");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("docs/Examples/demo.ipynb");
    expect(cmd).toContain("--sync");
    expect(cmd).toContain("--recursive");
  });

  it("rejects blank notebookPath and outputDir misuse", async () => {
    const execute = await loadToolExecute("../../sync_notebook_pair.ts");
    expect(await execute({ notebookPath: "   " })).toContain("notebookPath is required and must be non-empty");

    const result = await execute({ notebookPath: "demo.ipynb", outputDir: "scripts" });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("does not support outputDir");
  });

  it("prefers stdout, then stderr envelope, then ENOENT hint", async () => {
    const execute = await loadToolExecute("../../sync_notebook_pair.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout-only", stderr: "stderr shadow" }));
    expect(await execute({ notebookPath: "demo.ipynb" })).toBe("stdout-only");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "sync failed" }));
    assertContains(String(await execute({ notebookPath: "demo.ipynb" })), "ERROR: Notebook tool failed");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing backend", message: "spawn" }));
    assertContains(
      String(await execute({ notebookPath: "demo.ipynb" })),
      "Missing backing script .opencode/tools/validate_notebook.py",
    );
  });
});
