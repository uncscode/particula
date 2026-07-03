import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_issues_batch_init wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", stderr: "", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("initializes with total source and optional adw_id", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const result = await execute({ total: "5", source: "plans/input.md", adw_id: "A1B2C3D4" });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.kind).toBe("spawnSync");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "init",
      "--total",
      "5",
      "--source",
      "plans/input.md",
      "--adw-id",
      "a1b2c3d4",
    ]);
  });

  it("rejects invalid source with deterministic error", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const result = await execute({ total: "5", source: "../escape.md" });

    expect(result).toContain("ERROR: Invalid source path.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects blank required total or source before spawn", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const blankTotal = await execute({ total: "   ", source: "plan.md" });
    const blankSource = await execute({ total: "2", source: "   " });

    expect(blankTotal).toContain("ERROR: batch-init requires 'total' and 'source'.");
    expect(blankSource).toContain("ERROR: batch-init requires 'total' and 'source'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("treats blank optional adw_id as omitted", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const result = await execute({ total: "2", source: "plan.md", adw_id: "   " });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "init",
      "--total",
      "2",
      "--source",
      "plan.md",
    ]);
  });

  it("rejects invalid non-blank total before spawn", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const result = await execute({ total: "two", source: "plan.md" });

    expect(result).toContain('ERROR: Invalid total "two". Must be between 1 and 50.');
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-empty options for batch-init before spawn", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const result = await execute({ total: "2", source: "plan.md", options: "raw" });

    expect(result).toContain(
      'ERROR: Invalid options token "raw" for \'batch-init\': token is not allowed for this command.',
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("returns batch-init-specific spawn failure diagnostics", async () => {
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const execute = await loadToolExecute("../../adw_issues_batch_init.ts");

    const result = await execute({ total: "2", source: "plan.md" });

    expect(result).toContain("ERROR: Failed to execute 'adw spec batch batch-init'.");
    expect(result).toContain("stderr:\nfatal stderr");
    expect(result).not.toContain("shadow stdout");
  });
});
