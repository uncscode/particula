import { afterEach, beforeEach, describe, expect, it } from "bun:test";

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

describe("get_version wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("runs script with optional file when provided", async () => {
    setDollarText(buildSuccessOutput("1.2.3"));
    const execute = await loadToolExecute("../../get_version.ts");

    const result = await execute({ file: "package.json" });
    expect(result).toBe("1.2.3");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("get_version.py");
    expect(cmd).toContain("package.json");
  });

  it("omits blank optional file argument", async () => {
    setDollarText(buildSuccessOutput("0.0.1"));
    const execute = await loadToolExecute("../../get_version.ts");

    await execute({ file: "   " });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).not.toContain("package.json");
  });

  it("prefers stdout in failure diagnostics", async () => {
    const execute = await loadToolExecute("../../get_version.ts");
    setDollarError(buildDollarFailure({ stdout: "stdout failure", stderr: "stderr shadow" }));
    expect(await execute({})).toContain("stdout failure");
  });

  it("falls back to stderr then message and includes ENOENT hint", async () => {
    const execute = await loadToolExecute("../../get_version.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "can't open file /x/get_version.py" }));
    expect(await execute({})).toContain("Backend script not found");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "", message: "Unknown crash" }));
    expect(await execute({})).toContain("Unknown crash");
  });
});
