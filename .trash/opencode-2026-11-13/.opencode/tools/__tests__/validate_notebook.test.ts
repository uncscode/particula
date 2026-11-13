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

describe("validate_notebook wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("runs validation command with bounded options", async () => {
    setDollarText(buildSuccessOutput("NOTEBOOK_OK"));
    const execute = await loadToolExecute("../../validate_notebook.ts");
    const result = await execute({ notebookPath: "docs/Examples", recursive: true, options: "output=json skip-syntax" });

    expect(result).toBe("NOTEBOOK_OK");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("validate_notebook.py");
    expect(cmd).toContain("--output=json");
    expect(cmd).toContain("--skip-syntax");
    expect(cmd).toContain("--recursive");
  });

  it("supports validation-mode and alias bounded options on validation paths", async () => {
    setDollarText(buildSuccessOutput("NOTEBOOK_OK"));
    const execute = await loadToolExecute("../../validate_notebook.ts");

    await execute({ notebookPath: "docs/Examples/demo.ipynb", options: "validation-mode=fast" });
    let cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--validation-mode fast");
    expect(cmd).toContain("--output=summary");

    await execute({ notebookPath: "docs/Examples/demo.ipynb", options: "full" });
    cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--full");
  });

  it("keeps convert and check-sync bridge behavior explicit", async () => {
    setDollarText(buildSuccessOutput("NOTEBOOK_OK"));
    const execute = await loadToolExecute("../../validate_notebook.ts");

    await execute({ notebookPath: "docs/Examples/demo.ipynb", convertToPy: true, outputDir: "scripts" });
    let cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--convert-to-py");
    expect(cmd).toContain("--output-dir scripts");
    expect(cmd).not.toContain("--output=");

    await execute({ notebookPath: "docs/Examples/demo.py", convertToIpynb: true });
    cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--convert-to-ipynb");
    expect(cmd).not.toContain("--output=");

    await execute({ notebookPath: "docs/Examples/demo.ipynb", sync: true });
    cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--sync");
    expect(cmd).not.toContain("--output=");

    await execute({ notebookPath: "docs/Examples", checkSync: true, recursive: true });
    cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--check-sync");
    expect(cmd).toContain("--recursive");
    expect(cmd).not.toContain("--output=");
  });

  it("rejects malformed, duplicate, and unsupported options", async () => {
    const execute = await loadToolExecute("../../validate_notebook.ts");

    expect(await execute({ notebookPath: "n.ipynb", options: "unknown-token" })).toContain(
      "token is not supported",
    );
    expect(await execute({ notebookPath: "n.ipynb", options: "output=summary output=json" })).toContain(
      "duplicate token",
    );
    expect(await execute({ notebookPath: "n.ipynb", options: "output=xml" })).toContain(
      "output must be one of summary, full, json",
    );
    expect(await execute({ notebookPath: "n.ipynb", options: "validation-mode=slow" })).toContain(
      "validation-mode must be one of fast, full",
    );
    expect(await execute({ notebookPath: "n.ipynb", options: 'output="' })).toContain(
      "unterminated quoted value",
    );
  });

  it("rejects conflicting validation usage", async () => {
    const execute = await loadToolExecute("../../validate_notebook.ts");

    expect(await execute({ notebookPath: "n.ipynb", options: "fast full" })).toContain(
      "cannot both be true",
    );
    expect(await execute({ notebookPath: "n.ipynb", options: "validation-mode=fast full" })).toContain(
      "validation-mode' or 'fast/full",
    );
    expect(await execute({ notebookPath: "n.ipynb", convertToPy: true, options: "output=json" })).toContain(
      "can only be used with validation",
    );
    expect(await execute({ notebookPath: "n.ipynb", checkSync: true, options: "output=json" })).toContain(
      "can only be used with validation",
    );
    expect(await execute({ notebookPath: "n.ipynb", checkSync: true, options: "output=summary" })).toContain(
      "can only be used with validation",
    );
  });

  it("prefers stdout in subprocess failure", async () => {
    const execute = await loadToolExecute("../../validate_notebook.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout-only", stderr: "stderr shadow" }));
    expect(await execute({ notebookPath: "n.ipynb" })).toContain("stdout-only");
  });

  it("uses stderr envelope and ENOENT hint when stdout missing", async () => {
    const execute = await loadToolExecute("../../validate_notebook.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT notebook backend missing", message: "spawn" }));
    const result = await execute({ notebookPath: "n.ipynb" });
    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "ERROR: Notebook tool failed");
    assertContains(String(result), "Missing backing script .opencode/tools/validate_notebook.py");
  });
});
