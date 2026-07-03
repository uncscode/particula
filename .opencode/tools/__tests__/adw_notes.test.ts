import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnError,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_notes compatibility wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("parses show output through shared helper", async () => {
    setSpawnResponse({ stdout: '{"status":"ok"}', exitCode: 0 });
    const execute = await loadToolExecute("../../adw_notes.ts");
    const result = await execute({ command: "show", ref: "HEAD" });
    expect(result).toBe('{\n  "status": "ok"\n}');
  });

  it("uses shared field normalization for write operations", async () => {
    const execute = await loadToolExecute("../../adw_notes.ts");
    await execute({ command: "write", ref: "HEAD", fields: { " plan_summary ": "done" } } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw notes write --ref HEAD --field plan_summary done",
    );
  });

  it("preserves sparse omission when fields normalize to empty input", async () => {
    const execute = await loadToolExecute("../../adw_notes.ts");

    await execute({ command: "write", ref: "HEAD" } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw notes write --ref HEAD");

    await execute({ command: "write", ref: "HEAD", fields: null } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw notes write --ref HEAD");

    await execute({ command: "write", ref: "HEAD", fields: "   " } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw notes write --ref HEAD");
  });

  it("requires adw_id for write-from-state", async () => {
    const execute = await loadToolExecute("../../adw_notes.ts");
    const result = await execute({ command: "write-from-state", ref: "HEAD" });
    assertContains(String(result), "'adw_id' is required");
  });

  it("classifies malformed non-empty adw_id as a format error", async () => {
    const execute = await loadToolExecute("../../adw_notes.ts");
    const result = await execute({ command: "write-from-state", ref: "HEAD", adw_id: "bad" });
    assertContains(String(result), "8-character hex string");
  });

  it("decodes Uint8Array diagnostics on catch-path failures", async () => {
    setSpawnError({ stderr: Buffer.from("boom stderr") as unknown as string });
    const execute = await loadToolExecute("../../adw_notes.ts");
    const result = String(await execute({ command: "write", ref: "HEAD" }));
    expect(result).toContain("boom stderr");
  });
});
