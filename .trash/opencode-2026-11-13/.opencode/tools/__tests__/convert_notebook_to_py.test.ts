import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
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

describe("convert_notebook_to_py wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds conversion argv and omits blank outputDir", async () => {
    setDollarText(buildSuccessOutput("OK"));
    const execute = await loadToolExecute("../../convert_notebook_to_py.ts");

    await execute({ notebookPath: " docs/Examples/demo.ipynb ", recursive: true, outputDir: "  " });
    let cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("docs/Examples/demo.ipynb");
    expect(cmd).toContain("--convert-to-py");
    expect(cmd).toContain("--recursive");
    expect(cmd).not.toContain("--output-dir");

    await execute({ notebookPath: "docs/Examples", outputDir: "scripts" });
    cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--output-dir scripts");
  });

  it("rejects blank notebookPath", async () => {
    const execute = await loadToolExecute("../../convert_notebook_to_py.ts");
    expect(await execute({ notebookPath: "   " })).toContain("notebookPath is required and must be non-empty");
  });

  it("prefers stdout, then stderr envelope, then ENOENT hint", async () => {
    const execute = await loadToolExecute("../../convert_notebook_to_py.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout-only", stderr: "stderr shadow" }));
    expect(await execute({ notebookPath: "demo.ipynb" })).toBe("stdout-only");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "conversion failed" }));
    assertContains(String(await execute({ notebookPath: "demo.ipynb" })), "ERROR: Notebook tool failed");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing backend", message: "spawn" }));
    assertContains(
      String(await execute({ notebookPath: "demo.ipynb" })),
      "Missing backing script .opencode/tools/validate_notebook.py",
    );
  });
});
