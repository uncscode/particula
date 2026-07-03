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

describe("refactor_astgrep_preview wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles preview command without --update-all", async () => {
    setDollarText(buildSuccessOutput("preview diff"));
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    const result = await execute({ pattern: "old($$$ARGS)", rewrite: "new($$$ARGS)", lang: "python", path: "adw" });

    expect(result).toContain("preview diff");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("ast-grep run");
    expect(cmd).not.toContain("--update-all");
    expect(cmd).toContain("-- adw");
  });

  it("returns no-match output for empty preview results", async () => {
    setDollarText("");
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    expect(result).toBe("No matches found for pattern: a");
  });

  it("uses stderr before stdout/message in diagnostics", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal parse error", stdout: "shadow out", message: "shadow msg" }));
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    assertContains(String(result), "diagnostic: fatal parse error");
  });

  it("adds missing binary hint on ENOENT", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT ast-grep", message: "not found" }));
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    assertContains(String(result), "classification: missing_binary");
    assertContains(String(result), "Install ast-grep-cli");
  });

  it("separates parse-input failures from missing-binary failures", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    setDollarError(
      buildDollarFailure({
        stderr: "error: invalid pattern parse error near '$$$'",
        stdout: "tool output shadow",
        message: "rewrite parse failed",
      }),
    );
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    assertErrorPrefix(String(result), "ERROR:");
    assertContains(String(result), "classification: parse_input");
    assertContains(String(result), "Fix the ast-grep pattern/rewrite input and retry");
    expect(String(result)).not.toContain("Install ast-grep-cli");
  });
});
