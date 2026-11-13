import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import path from "node:path";

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
    const result = await execute({ pattern: "old($$$ARGS)", rewrite: "new($$$ARGS)", lang: "python", path: "." });

    expect(result).toEqual({ kind: "match", preview: "preview diff", previewLineCount: 1, truncated: false });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("ast-grep run");
    expect(cmd).not.toContain("--update-all");
    expect(cmd).toContain(`-- ${path.join(process.cwd(), ".")}`);
  });

  it("returns no-match output for empty preview results", async () => {
    setDollarText("");
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    expect(result).toEqual({ kind: "no_match" });
  });

  it("preserves preview output lines while bounding its structured result", async () => {
    setDollarText("first line\n\nsecond line\n");
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    expect(result).toEqual({
      kind: "match",
      preview: "first line\nsecond line",
      previewLineCount: 2,
      truncated: false,
    });
  });

  it("caps preview output at 200 non-empty lines and marks it truncated", async () => {
    setDollarText(Array.from({ length: 201 }, (_, index) => `match ${index + 1}`).join("\n"));
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    expect(result).toMatchObject({
      kind: "match",
      previewLineCount: 200,
      truncated: true,
    });
    expect(result).toMatchObject({ preview: expect.stringContaining("match 200") });
    expect(result).toMatchObject({ preview: expect.not.stringContaining("match 201") });
  });

  it("returns typed input failures without an ERROR envelope", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    const result = await execute({ rewrite: "b", lang: "python" });

    expect(result).toEqual({
      kind: "runtime_error",
      diagnostic: "pattern is required. Provide the AST pattern to match.",
    });
    expect(getInvocations()).toHaveLength(0);
  });

  it("uses stderr before stdout/message in diagnostics", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal parse error", stdout: "shadow out", message: "shadow msg" }));
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    expect(result).toEqual({
      kind: "parse_error",
      diagnostic: "fatal parse error",
      hint: "Fix the ast-grep pattern/rewrite input and retry.",
    });
  });

  it("adds missing binary hint on ENOENT", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_preview.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT ast-grep", message: "not found" }));
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });

    expect(result).toEqual({
      kind: "unavailable",
      diagnostic: "ENOENT ast-grep",
      hint: "Install ast-grep-cli (pip install ast-grep-cli) and ensure ast-grep is on PATH.",
    });
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

    expect(result).toEqual({
      kind: "parse_error",
      diagnostic: "invalid pattern parse error near '$$$'",
      hint: "Fix the ast-grep pattern/rewrite input and retry.",
    });
  });
});
