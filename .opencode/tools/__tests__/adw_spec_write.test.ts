import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnError, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_spec_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires content or file for write", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "a1b2c3d4" });
    assertContains(String(result), "requires either 'content' or 'file'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing and whitespace-only adw_id before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");

    const missing = await execute({ command: "write", content: "x" } as any);
    assertContains(String(missing), "'adw_id' parameter is required");

    const whitespace = await execute({ command: "write", adw_id: "   ", content: "x" });
    assertContains(String(whitespace), "'adw_id' parameter is required");

    expect(getInvocations()).toHaveLength(0);
  });

  it("validates adw_id format before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "bad", content: "x" });
    assertContains(String(result), "8-character hex");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null and non-string adw_id before spawn with required-input validation", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");

    for (const invalidValue of [null, 1234, {}, []] as const) {
      const result = await execute({ command: "write", adw_id: invalidValue, content: "x" } as any);
      assertContains(String(result), "'adw_id' parameter is required");
    }

    expect(getInvocations()).toHaveLength(0);
  });

  it("requires field for delete", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "delete", adw_id: "a1b2c3d4" });
    assertContains(String(result), "requires 'field'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("assembles write with content", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    await execute({ command: "write", adw_id: "a1b2c3d4", content: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw spec write --adw-id a1b2c3d4 --content x");
  });

  it("forwards normalized field values and lowercase adw_id for write/delete", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");

    await execute({ command: "write", adw_id: "A1B2C3D4", field: " plan_file ", content: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec write --adw-id a1b2c3d4 --field plan_file --content x",
    );

    await execute({ command: "delete", adw_id: "A1B2C3D4", field: " stale_field " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec delete --adw-id a1b2c3d4 --field stale_field",
    );
  });

  it("rejects file path outside repository root", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "a1b2c3d4", file: "/etc/hosts" });
    assertContains(String(result), "resolves outside repository root");
    expect(String(result)).not.toContain("/etc/hosts");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing file path before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "a1b2c3d4", file: "does-not-exist.md" });
    assertContains(String(result), "path does not exist");
    expect(getInvocations()).toHaveLength(0);
  });

  it("redacts absolute paths in non-zero exit diagnostics", async () => {
    setSpawnResponse({ stderr: "failed at /home/kyle/private.txt", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "a1b2c3d4", content: "x" });
    expect(String(result)).toContain("<path>");
    expect(String(result)).not.toContain("/home/kyle/private.txt");
  });

  it("redacts absolute paths in catch-path diagnostics", async () => {
    setSpawnError({ stderr: "failed at /home/kyle/private.txt", stdout: "shadow stdout", message: "ignored" });
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = String(await execute({ command: "write", adw_id: "a1b2c3d4", content: "x" }));
    expect(result).toContain("<path>");
    expect(result).not.toContain("/home/kyle/private.txt");
    expect(result).not.toContain("shadow stdout");
  });

  it("prefers stderr over stdout for non-zero exit diagnostics", async () => {
    setSpawnResponse({ stderr: "fatal stderr", stdout: "shadow stdout", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "a1b2c3d4", content: "x" });
    const text = String(result);
    expect(text).toContain("fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("falls back to stdout when stderr is empty", async () => {
    setSpawnResponse({ stderr: "", stdout: "stdout diagnostic", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_write.ts");
    const result = await execute({ command: "write", adw_id: "a1b2c3d4", content: "x" });
    expect(String(result)).toContain("stdout diagnostic");
  });
});
