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

describe("validate_notebook_readonly wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("runs validation command in success path", async () => {
    setDollarText(buildSuccessOutput("NOTEBOOK_OK"));
    const execute = await loadToolExecute("../../validate_notebook_readonly.ts");
    const result = await execute({ notebookPath: "docs/Examples", recursive: true, outputMode: "json" });

    expect(result).toBe("NOTEBOOK_OK");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("validate_notebook.py");
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--recursive");
  });

  it("rejects mutating keys", async () => {
    const execute = await loadToolExecute("../../validate_notebook_readonly.ts");
    const result = await execute({ notebookPath: "n.ipynb", sync: true });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("does not support mutating options");
  });

  it("rejects empty notebookPath and conflicting validation options", async () => {
    const execute = await loadToolExecute("../../validate_notebook_readonly.ts");
    expect(await execute({ notebookPath: "  " })).toContain("'notebookPath' is required");
    expect(await execute({ notebookPath: "n.ipynb", fast: true, full: true })).toContain("cannot both be true");
    expect(await execute({ notebookPath: "n.ipynb", validationMode: "fast", full: true })).toContain(
      "either 'validationMode' or 'fast/full'",
    );
    expect(await execute({ notebookPath: "n.ipynb", checkSync: true, outputMode: "summary" })).toContain(
      "'checkSync' cannot be combined",
    );
  });

  it("prefers stdout in subprocess failure", async () => {
    const execute = await loadToolExecute("../../validate_notebook_readonly.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout-only", stderr: "stderr shadow" }));
    expect(await execute({ notebookPath: "n.ipynb" })).toContain("stdout-only");
  });

  it("uses stderr envelope and ENOENT hint when stdout missing", async () => {
    const execute = await loadToolExecute("../../validate_notebook_readonly.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT notebook backend missing", message: "spawn" }));
    const result = await execute({ notebookPath: "n.ipynb" });
    assertContains(String(result), "ERROR: Notebook tool failed");
    assertContains(String(result), "Missing backing script .opencode/tools/validate_notebook.py");
  });
});
