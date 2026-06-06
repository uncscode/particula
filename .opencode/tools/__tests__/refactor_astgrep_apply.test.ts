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

describe("refactor_astgrep_apply wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles apply command with --update-all", async () => {
    setDollarText(buildSuccessOutput("updated 2 files"));
    const execute = await loadToolExecute("../../refactor_astgrep_apply.ts");
    const result = await execute({ pattern: "old($$$ARGS)", rewrite: "new($$$ARGS)", lang: "python", path: "adw" });

    expect(result).toContain("updated 2 files");
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("ast-grep run");
    expect(cmd).toContain("--update-all");
    expect(cmd).toContain("-- adw");
  });

  it("rejects missing required fields", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_apply.ts");
    assertErrorPrefix(String(await execute({ rewrite: "x", lang: "python" })), "ERROR:");
    expect(await execute({ rewrite: "x", lang: "python" })).toContain("pattern is required");
    expect(await execute({ pattern: "a", lang: "python" })).toContain("rewrite is required");
    expect(await execute({ pattern: "a", rewrite: "b" })).toContain("lang is required");
  });

  it("rejects invalid lang value", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_apply.ts");
    const result = await execute({ pattern: "a", rewrite: "b", lang: "elixir" });
    expect(result).toContain("lang must be one of");
  });

  it("uses stderr before stdout/message in diagnostics", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_apply.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal parse error", stdout: "shadow out", message: "shadow msg" }));
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });
    assertContains(String(result), "diagnostic: fatal parse error");
  });

  it("adds missing binary hint on ENOENT", async () => {
    const execute = await loadToolExecute("../../refactor_astgrep_apply.ts");
    setDollarError(buildDollarFailure({ stderr: "ENOENT ast-grep", message: "not found" }));
    const result = await execute({ pattern: "a", rewrite: "b", lang: "python" });
    assertContains(String(result), "Install ast-grep-cli");
  });
});
