import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("clear_build wrapper family", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("always forces dry-run preview semantics", async () => {
    const execute = await loadToolExecute("../../clear_build_preview.ts");

    const result = await execute({ buildDir: "build/debug" });
    expect(result).toBe("ok");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--build-dir=build/debug");
    expect(cmd).toContain("--dry-run");
    expect(cmd).not.toContain("--force");
  });

  it("requires force=true for delete wrapper and does not spawn on rejection", async () => {
    const execute = await loadToolExecute("../../clear_build_delete.ts");

    const result = await execute({});
    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain("requires force: true");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves compatibility routing defaults and conflict rejection", async () => {
    const execute = await loadToolExecute("../../clear_build.ts");

    const preview = await execute({ dryRun: true });
    expect(preview).toBe("ok");
    let cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--dry-run");
    expect(cmd).not.toContain("--force");

    resetSubprocessMocks();
    setDollarText("ok");

    const deletion = await execute({ force: true });
    expect(deletion).toBe("ok");
    cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("--force");
    expect(cmd).not.toContain("--dry-run");

    const conflict = await execute({ dryRun: true, force: true });
    assertErrorPrefix(String(conflict), "ERROR:");
    expect(String(conflict)).toContain("mode conflict");
  });

  it("rejects destructive compatibility call without force before subprocess execution", async () => {
    const execute = await loadToolExecute("../../clear_build.ts");

    const result = await execute({ dryRun: false });
    assertErrorPrefix(String(result), "ERROR:");
    expect(String(result)).toContain("requires force: true");
    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves ENOENT hinting for destructive wrapper failures", async () => {
    const execute = await loadToolExecute("../../clear_build_delete.ts");
    setDollarError(buildDollarFailure({ stdout: "", stderr: "ENOENT missing script" }));

    const result = await execute({ force: true });
    assertContains(String(result), "ERROR: Clear build delete failed");
    assertContains(String(result), "Missing backing script .opencode/tools/clear_build.py");
  });
});
