import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnError, setSpawnResponse } from "./helpers/mock-subprocess";
import { getCapturedToolDefinition, loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_spec_messages wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires agent for messages-write", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-write", adw_id: "a1b2c3d4", message: "done" });
    assertContains(String(result), "'agent' parameter is required");
  });

  it("rejects blank adw_id before spawn with required-input validation", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "   " });
    assertContains(String(result), "'adw_id' parameter is required");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null and non-string adw_id before spawn with required-input validation", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");

    for (const invalidValue of [null, 1234, {}, []] as const) {
      const result = await execute({ command: "messages-read", adw_id: invalidValue } as any);
      assertContains(String(result), "'adw_id' parameter is required");
    }

    expect(getInvocations()).toHaveLength(0);
  });

  it("validates last range", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4", options: "last=99" });
    assertContains(String(result), "between 0 and 50");
  });

  it("validates last is integer", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4", options: "last=1.25" });
    assertContains(String(result), "must be an integer");
    expect(getInvocations()).toHaveLength(0);
  });

  it("requires message for messages-write", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-write", adw_id: "a1b2c3d4", agent: "planner" });
    assertContains(String(result), "'message' parameter is required");
  });

  it("assembles messages-read", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    await execute({ command: "messages-read", adw_id: "a1b2c3d4", options: "last=3" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw spec messages read --adw-id a1b2c3d4 --last 3");
  });

  it("assembles raw messages-read and normalizes uppercase adw_id", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    await execute({ command: "messages-read", adw_id: "A1B2C3D4", options: "raw" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw spec messages read --adw-id a1b2c3d4 --raw",
    );
  });

  it("forwards normalized messages-write agent/message values", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    await execute({ command: "messages-write", adw_id: "A1B2C3D4", agent: " planner ", message: " done " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw spec messages write --adw-id a1b2c3d4 --agent planner --message done",
    );
  });

  it("omits --last when messages-read last is zero", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    await execute({ command: "messages-read", adw_id: "a1b2c3d4", options: "last=0" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw spec messages read --adw-id a1b2c3d4",
    );
  });

  it("keeps success envelope for raw messages-read", async () => {
    setSpawnResponse({ stdout: "line1\nline2", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4", options: "raw" });
    expect(String(result)).toBe("ADW Spec Command: messages-read\n\nline1\nline2");
  });

  it("keeps success envelope for last=0 raw messages-read", async () => {
    setSpawnResponse({ stdout: "all messages", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4", options: "last=0 raw" });
    expect(String(result)).toBe("ADW Spec Command: messages-read\n\nall messages");
  });

  it("prefers stderr over stdout for non-zero exit diagnostics", async () => {
    setSpawnResponse({ stderr: "fatal stderr", stdout: "shadow stdout", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4" });
    const text = String(result);
    expect(text).toContain("fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("falls back to stdout when stderr is empty", async () => {
    setSpawnResponse({ stderr: "", stdout: "stdout diagnostic", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4" });
    expect(String(result)).toContain("stdout diagnostic");
  });

  it("redacts absolute paths in diagnostics", async () => {
    setSpawnResponse({ stderr: "boom /var/tmp/secret.log", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4" });
    expect(String(result)).toContain("<path>");
    expect(String(result)).not.toContain("/var/tmp/secret.log");
  });

  it("redacts absolute paths in catch-path diagnostics", async () => {
    setSpawnError({ stderr: "boom /var/tmp/secret.log", stdout: "shadow stdout", message: "ignored" });
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = String(await execute({ command: "messages-read", adw_id: "a1b2c3d4" }));
    expect(result).toContain("<path>");
    expect(result).not.toContain("/var/tmp/secret.log");
    expect(result).not.toContain("shadow stdout");
  });

  it("rejects wrong-command options tokens before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec_messages.ts");
    const result = await execute({ command: "messages-write", adw_id: "a1b2c3d4", agent: "planner", message: "done", options: "raw" });
    assertContains(String(result), "token is not allowed for this command");
    expect(getInvocations()).toHaveLength(0);
  });

  it("exposes slim schema with options carrier", async () => {
    await loadToolExecute("../../adw_spec_messages.ts");
    const definition = getCapturedToolDefinition();
    expect(Object.keys(definition?.args ?? {})).toContain("options");
    expect(Object.keys(definition?.args ?? {})).not.toContain("last");
    expect(Object.keys(definition?.args ?? {})).not.toContain("raw");
  });
});
