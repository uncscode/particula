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

describe("run_notebook wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("builds execution argv with bounded output option and safety flags", async () => {
    setDollarText(buildSuccessOutput("OK"));
    const execute = await loadToolExecute("../../run_notebook.ts");
    const result = await execute({
      notebookPath: " docs/Examples/demo.ipynb ",
      options: "output=full",
      recursive: true,
      script: true,
      timeout: 300,
      cwd: "  ",
      writeExecuted: " ",
      expectOutput: ["done"],
      noOverwrite: true,
      noBackup: true,
      skipValidation: true,
    });

    expect(result).toBe("OK");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("run_notebook.py");
    expect(cmd).toContain("docs/Examples/demo.ipynb");
    expect(cmd).toContain("--output=full");
    expect(cmd).toContain("--timeout=300");
    expect(cmd).toContain("--recursive");
    expect(cmd).toContain("--script");
    expect(cmd).toContain("--expect-output done");
    expect(cmd).toContain("--no-overwrite");
    expect(cmd).toContain("--no-backup");
    expect(cmd).toContain("--skip-validation");
    expect(cmd).not.toContain("--cwd=");
    expect(cmd).not.toContain("--write-executed=");
  });

  it("rejects malformed, duplicate, and unsupported bounded options", async () => {
    const execute = await loadToolExecute("../../run_notebook.ts");

    expect(await execute({ notebookPath: "demo.ipynb", options: "unknown-token" })).toContain(
      "requires a non-empty '=value' suffix",
    );
    expect(await execute({ notebookPath: "demo.ipynb", options: "format=json" })).toContain(
      "token is not supported",
    );
    expect(await execute({ notebookPath: "demo.ipynb", options: "output=full output=json" })).toContain(
      "duplicate token",
    );
    expect(await execute({ notebookPath: "demo.ipynb", options: "output=xml" })).toContain(
      "output must be one of summary, full, json",
    );
    expect(await execute({ notebookPath: "demo.ipynb", options: "output=full=json" })).toContain(
      "at most one '=' separator",
    );
  });

  it("rejects blank notebookPath and invalid timeout values", async () => {
    const execute = await loadToolExecute("../../run_notebook.ts");

    expect(await execute({ notebookPath: "   " })).toContain("notebookPath is required and must be non-empty");
    expect(await execute({ notebookPath: "demo.ipynb", timeout: 0 })).toContain(
      "timeout must be a positive finite number",
    );
    expect(await execute({ notebookPath: "demo.ipynb", timeout: Number.POSITIVE_INFINITY })).toContain(
      "timeout must be a positive finite number",
    );
    expect(await execute({ notebookPath: "demo.ipynb", timeout: Number.NaN })).toContain(
      "timeout must be a positive finite number",
    );
  });

  it("rejects dash-prefixed expectOutput values and allows normal ones", async () => {
    const execute = await loadToolExecute("../../run_notebook.ts");
    expect(await execute({ notebookPath: "demo.ipynb", expectOutput: ["--json"] })).toContain(
      "expectOutput entries must not start",
    );

    setDollarText(buildSuccessOutput("OK"));
    const result = await execute({ notebookPath: "demo.ipynb", expectOutput: ["result", "plot"] });
    expect(result).toBe("OK");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--expect-output result plot");
  });

  it("prefers stdout in subprocess failure", async () => {
    const execute = await loadToolExecute("../../run_notebook.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));

    expect(await execute({ notebookPath: "demo.ipynb" })).toBe("stdout diagnostic");
  });

  it("uses stderr envelope and ENOENT hint when stdout missing", async () => {
    const execute = await loadToolExecute("../../run_notebook.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT execution backend missing", message: "spawn" }));

    const result = await execute({ notebookPath: "demo.ipynb" });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "ERROR: Notebook execution failed");
    assertContains(String(result), "Missing backing script .opencode/tools/run_notebook.py");
  });
});
