import { afterEach, beforeEach, describe, expect, it, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnError, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_spec_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "value", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("validates adw_id", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "bad" });
    assertContains(String(result), "8-character hex");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects missing adw_id before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read" } as any);
    assertContains(String(result), "'adw_id' parameter is required");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects whitespace-only adw_id before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "list", adw_id: "   " });
    assertContains(String(result), "'adw_id' parameter is required");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null and non-string adw_id before spawn with required-input validation", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");

    for (const invalidValue of [null, 1234, {}, []] as const) {
      const result = await execute({ command: "read", adw_id: invalidValue } as any);
      assertContains(String(result), "'adw_id' parameter is required");
    }

    expect(getInvocations()).toHaveLength(0);
  });

  it("assembles list with json", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    await execute({ command: "list", adw_id: "a1b2c3d4", json: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw spec list --adw-id a1b2c3d4 --json");
  });

  it("returns raw stdout for read command without envelope", async () => {
    setSpawnResponse({ stdout: "raw-content", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "a1b2c3d4" });
    expect(result).toBe("raw-content");
  });

  it("returns empty string for successful empty read", async () => {
    setSpawnResponse({ stdout: "", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "a1b2c3d4" });
    expect(result).toBe("");
  });

  it("adds --field and --raw when provided for read", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    await execute({ command: "read", adw_id: "a1b2c3d4", field: "spec_content", raw: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec read --adw-id a1b2c3d4 --field spec_content --raw",
    );
  });

  it("forwards normalized field values instead of whitespace-padded input", async () => {
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    await execute({ command: "read", adw_id: "A1B2C3D4", field: " spec_content ", raw: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec read --adw-id a1b2c3d4 --field spec_content --raw",
    );
  });

  it("prefers stderr over stdout for non-zero exit diagnostics", async () => {
    setSpawnResponse({ stderr: "fatal stderr", stdout: "shadow stdout", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "a1b2c3d4" });
    const text = String(result);
    expect(text).toContain("fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("falls back to stdout when stderr is empty", async () => {
    setSpawnResponse({ stderr: "", stdout: "stdout diagnostic", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "a1b2c3d4" });
    expect(String(result)).toContain("stdout diagnostic");
  });

  it("redacts absolute paths in diagnostics before truncation", async () => {
    setSpawnResponse({ stderr: `problem at /home/kyle/secret/file.txt ${"x".repeat(1400)}`, exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "a1b2c3d4" });
    const text = String(result);
    expect(text).toContain("<path>");
    expect(text).not.toContain("/home/kyle/secret/file.txt");
    expect(text).toContain("...");
  });

  it("uses catch-path stderr-first diagnostics with absolute-path redaction", async () => {
    setSpawnError({ stderr: "boom /var/tmp/private.log", stdout: "shadow stdout", message: "ignored" });
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = String(await execute({ command: "read", adw_id: "a1b2c3d4" }));
    expect(result).toContain("<path>");
    expect(result).not.toContain("/var/tmp/private.log");
    expect(result).not.toContain("shadow stdout");
  });

  it("validates findings artifact required fields and accepted mapping paths", () => {
    const artifactPath = path.resolve(
      import.meta.dir,
      "fixtures/feedback_findings/M34-P1-findings.md",
    );
    const content = fs.readFileSync(artifactPath, "utf8");

    expect(content).toContain("## M34-P1-F01");
    expect(content).toContain("## M34-P1-F02");
    expect(content).toContain("**Decision:** accepted");
    expect(content).toContain("**Drift class:** input validation drift");
    expect(content).toContain("**Drift class:** diagnostic precedence/envelope drift");
    expect(content).toContain(".opencode/tools/adw_spec_read.ts");
    expect(content).toContain(".opencode/tools/platform_operations.ts");
    expect(content).toContain("**Repro input payload:**");
    expect(content).toContain("**Observed behavior:**");
    expect(content).toContain("**Expected contract behavior");
    expect(content).toContain("**Wrapper-level behavior:**");
    expect(content).toContain("**Runtime/CLI behavior boundary:**");
  });

  test.skip("baseline repro (quarantined): returns deterministic validation envelope for invalid adw_id", async () => {
    // baseline repro retained for M34-P1 audit evidence; quarantined from default pass path
    const execute = await loadToolExecute("../../adw_spec_read.ts");
    const result = await execute({ command: "read", adw_id: "bad" });
    assertContains(String(result), "ERROR:");
    assertContains(String(result), "exactly 8-character lowercase hex");
  });
});
