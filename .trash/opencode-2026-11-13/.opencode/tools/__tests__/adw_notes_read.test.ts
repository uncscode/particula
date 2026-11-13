import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { installSubprocessMocks, restoreSubprocessMocks, setSpawnError, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_notes_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("parses successful show JSON output", async () => {
    setSpawnResponse({ stdout: '{"a":1}\n', exitCode: 0 });
    const execute = await loadToolExecute("../../adw_notes_read.ts");
    const result = await execute({ command: "show", ref: "HEAD" });
    expect(result).toBe('{\n  "a": 1\n}');
  });

  it("returns deterministic parse failure snippet for invalid show JSON", async () => {
    setSpawnResponse({ stdout: "not-json\n", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_notes_read.ts");
    const result = await execute({ command: "show", ref: "HEAD" });
    assertContains(String(result), "Failed to parse JSON output from adw notes show");
    assertContains(String(result), "not-json");
  });

  it("preserves stderr-first diagnostics", async () => {
    setSpawnResponse({ stderr: "fatal stderr", stdout: "shadow stdout", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_notes_read.ts");
    const result = String(await execute({ command: "show", ref: "HEAD" }));
    expect(result).toContain("fatal stderr");
    expect(result).not.toContain("shadow stdout");
  });

  it("uses catch-path diagnostic precedence", async () => {
    setSpawnError({ stderr: "boom stderr", stdout: "shadow stdout", message: "ignored" });
    const execute = await loadToolExecute("../../adw_notes_read.ts");
    const result = String(await execute({ command: "show", ref: "HEAD" }));
    expect(result).toContain("boom stderr");
    expect(result).not.toContain("shadow stdout");
  });

  it("redacts secrets and absolute paths in invalid show snippets", async () => {
    setSpawnResponse({ stdout: 'token=ghp_supersecret12345678 path="/home/kyle/private.txt"', exitCode: 0 });
    const execute = await loadToolExecute("../../adw_notes_read.ts");
    const result = String(await execute({ command: "show", ref: "HEAD" }));
    expect(result).toContain("<redacted-secret>");
    expect(result).toContain("<path>");
    expect(result).not.toContain("ghp_supersecret12345678");
    expect(result).not.toContain("/home/kyle/private.txt");
  });
});
