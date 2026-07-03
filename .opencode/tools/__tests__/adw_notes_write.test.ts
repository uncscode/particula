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

describe("adw_notes_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires adw_id for write-from-state", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    const result = await execute({ command: "write-from-state", ref: "HEAD" });
    assertContains(String(result), "'adw_id' is required");
  });

  it("classifies malformed non-empty adw_id as a format error", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    const result = await execute({ command: "write-from-state", ref: "HEAD", adw_id: "bad" });
    assertContains(String(result), "8-character hex string");
  });

  it("accepts structured fields payloads and preserves deterministic ordering", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    await execute({
      command: "write",
      ref: "HEAD",
      fields: [[" first ", "one"], null, { key: "second", value: "two" }, { key: "third", value: "three" }],
    } as any);

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw notes write --ref HEAD --field first one --field second two --field third three",
    );
  });

  it("normalizes write-from-state adw_id and supports JSON-string fields", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    await execute({
      command: "write-from-state",
      ref: "HEAD",
      adw_id: "A1B2C3D4",
      fields: '{"plan_summary":"done"}',
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw notes write-from-state --ref HEAD --adw-id a1b2c3d4 --field plan_summary done",
    );
  });

  it("normalizes omitted, null, and blank-string fields to an empty field list", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");

    await execute({ command: "write", ref: "HEAD" } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw notes write --ref HEAD");

    await execute({ command: "write", ref: "HEAD", fields: null } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw notes write --ref HEAD");

    await execute({ command: "write", ref: "HEAD", fields: "   " } as any);
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw notes write --ref HEAD");
  });

  it("rejects malformed non-null structured field entries with exact index diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");

    const tupleResult = await execute({ command: "write", ref: "HEAD", fields: [["ok", "one", "extra"]] } as any);
    assertContains(String(tupleResult), "index 0");
    assertContains(String(tupleResult), "tuple must contain exactly [key, value]");

    const objectResult = await execute({ command: "write", ref: "HEAD", fields: [{ key: "ok" }] } as any);
    assertContains(String(objectResult), "index 0");
    assertContains(String(objectResult), "value is missing");
  });

  it("classifies null and wrong-typed object-style values separately", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");

    const nullResult = await execute({ command: "write", ref: "HEAD", fields: [{ key: "ok", value: null }] } as any);
    assertContains(String(nullResult), "index 0");
    assertContains(String(nullResult), "value is null");

    const wrongTypeResult = await execute({ command: "write", ref: "HEAD", fields: [{ key: "ok", value: 1 }] } as any);
    assertContains(String(wrongTypeResult), "index 0");
    assertContains(String(wrongTypeResult), "value has wrong type number");
  });

  it("reports offending plain-object keys for malformed values", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");

    const nullResult = await execute({ command: "write", ref: "HEAD", fields: { broken: null } } as any);
    assertContains(String(nullResult), 'object key "broken"');
    assertContains(String(nullResult), "value is null");

    const blankKeyResult = await execute({ command: "write", ref: "HEAD", fields: { "   ": "x" } } as any);
    assertContains(String(blankKeyResult), 'object key "   "');
    assertContains(String(blankKeyResult), "key is blank");
  });

  it("rejects malformed fields payload strings", async () => {
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    const result = await execute({ command: "write", ref: "HEAD", fields: '{bad json' });
    assertContains(String(result), "JSON string could not be parsed");
  });

  it("uses shared catch-path diagnostic precedence", async () => {
    setSpawnError({ stderr: "boom stderr", stdout: "shadow stdout", message: "ignored" });
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    const result = String(await execute({ command: "write", ref: "HEAD" }));
    expect(result).toContain("boom stderr");
    expect(result).not.toContain("shadow stdout");
  });

  it("decodes Uint8Array diagnostics on catch-path failures", async () => {
    setSpawnError({ stderr: Buffer.from("boom stderr") as unknown as string, stdout: "shadow stdout" });
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    const result = String(await execute({ command: "write", ref: "HEAD" }));
    expect(result).toContain("boom stderr");
    expect(result).not.toContain("shadow stdout");
  });

  it("preserves VIRTUAL_ENV for uv run --active", async () => {
    const previous = process.env.VIRTUAL_ENV;
    process.env.VIRTUAL_ENV = "/tmp/venv";
    try {
      const execute = await loadToolExecute("../../adw_notes_write.ts");
      await execute({ command: "write", ref: "HEAD" });
      expect(getInvocations().at(-1)?.env?.VIRTUAL_ENV).toBe("/tmp/venv");
    } finally {
      if (previous === undefined) {
        delete process.env.VIRTUAL_ENV;
      } else {
        process.env.VIRTUAL_ENV = previous;
      }
    }
  });

  it("redacts secrets and absolute paths in catch-path diagnostics", async () => {
    setSpawnError({
      stderr: "token=ghp_supersecret12345678 failed at /home/kyle/private.txt",
      stdout: "shadow stdout",
      message: "ignored",
    });
    const execute = await loadToolExecute("../../adw_notes_write.ts");
    const result = String(await execute({ command: "write", ref: "HEAD" }));
    expect(result).toContain("<redacted-secret>");
    expect(result).toContain("<path>");
    expect(result).not.toContain("ghp_supersecret12345678");
    expect(result).not.toContain("/home/kyle/private.txt");
    expect(result).not.toContain("shadow stdout");
  });
});
