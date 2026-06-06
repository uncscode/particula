import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import fs from "node:fs";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_spec compatibility wrapper diagnostics", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("keeps success marker for non-read commands", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "list", adw_id: "a1b2c3d4" });
    expect(String(result)).toContain("ADW Spec Command: list");
  });

  it("returns raw stdout for read commands without envelope", async () => {
    setSpawnResponse({ stdout: "read-value", exitCode: 0 });
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "read", adw_id: "a1b2c3d4" });
    expect(result).toBe("read-value");
  });

  it("rejects missing and blank adw_id before spawn with deterministic validation", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");

    const missing = await execute({ command: "list" } as any);
    assertContains(String(missing), "'adw_id' parameter is required");

    const blank = await execute({ command: "list", adw_id: "   " });
    assertContains(String(blank), "'adw_id' parameter is required");

    const malformed = await execute({ command: "list", adw_id: "bad" });
    assertContains(String(malformed), "8-character hex string");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects null and non-string adw_id before spawn with required-input validation", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");

    for (const invalidValue of [null, 1234, {}, []] as const) {
      const result = await execute({ command: "list", adw_id: invalidValue } as any);
      assertContains(String(result), "'adw_id' parameter is required");
    }

    expect(getInvocations()).toHaveLength(0);
  });

  it("prefers stderr over stdout for non-zero exit diagnostics", async () => {
    setSpawnResponse({ stderr: "fatal stderr", stdout: "shadow stdout", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "list", adw_id: "a1b2c3d4" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("falls back to stdout when stderr is empty", async () => {
    setSpawnResponse({ stderr: "", stdout: "stdout diagnostic", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "list", adw_id: "a1b2c3d4" });
    assertContains(String(result), "stdout diagnostic");
  });

  it("validates messages-read last is an integer before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4", last: 1.25 });

    assertContains(String(result), "must be an integer");
    expect(getInvocations()).toHaveLength(0);
  });

  it("validates messages-read last range before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "messages-read", adw_id: "a1b2c3d4", last: 99 });

    assertContains(String(result), "between 0 and 50");
    expect(getInvocations()).toHaveLength(0);
  });

  it("assembles messages-read raw mode and lowercases adw_id", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");
    await execute({ command: "messages-read", adw_id: "A1B2C3D4", last: 3, raw: true });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec messages read --adw-id a1b2c3d4 --last 3 --raw",
    );
  });

  it("omits optional blank field for read and write commands", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");

    await execute({ command: "read", adw_id: "a1b2c3d4", field: "   " });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run adw spec read --adw-id a1b2c3d4",
    );

    await execute({ command: "write", adw_id: "a1b2c3d4", field: "   ", content: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run adw spec write --adw-id a1b2c3d4 --content x",
    );
  });

  it("confines compatibility write --file paths to repository roots", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");
    const fixturePath = `${process.cwd()}/README.md`;

    await execute({ command: "write", adw_id: "a1b2c3d4", file: fixturePath });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(`--file ${fs.realpathSync(fixturePath)}`);

    const outside = await execute({ command: "write", adw_id: "a1b2c3d4", file: "/etc/hosts" });
    assertContains(String(outside), "resolves outside repository root");
  });

  it("forwards normalized strings for read, write, and messages-write", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");

    await execute({ command: "read", adw_id: "A1B2C3D4", field: " spec_content " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec read --adw-id a1b2c3d4 --field spec_content",
    );

    await execute({ command: "write", adw_id: "A1B2C3D4", field: " plan_file ", content: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec write --adw-id a1b2c3d4 --field plan_file --content x",
    );

    await execute({ command: "messages-write", adw_id: "A1B2C3D4", agent: " planner ", message: " done " });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec messages write --adw-id a1b2c3d4 --agent planner --message done",
    );
  });

  it("rejects whitespace-only required strings for delete and messages-write before spawn", async () => {
    const execute = await loadToolExecute("../../adw_spec.ts");

    const deleteResult = await execute({ command: "delete", adw_id: "a1b2c3d4", field: "   " });
    assertContains(String(deleteResult), "requires 'field'");

    const agentResult = await execute({
      command: "messages-write",
      adw_id: "a1b2c3d4",
      agent: "   ",
      message: "done",
    });
    assertContains(String(agentResult), "'agent' parameter is required");

    const messageResult = await execute({
      command: "messages-write",
      adw_id: "a1b2c3d4",
      agent: "planner",
      message: "   ",
    });
    assertContains(String(messageResult), "'message' parameter is required");

    expect(getInvocations()).toHaveLength(0);
  });

  it("redacts absolute paths in non-zero exit diagnostics", async () => {
    setSpawnResponse({ stderr: "fatal /var/tmp/secret.log", exitCode: 2 });
    const execute = await loadToolExecute("../../adw_spec.ts");
    const result = await execute({ command: "list", adw_id: "a1b2c3d4" });
    const text = String(result);

    expect(text).toContain("<path>");
    expect(text).not.toContain("/var/tmp/secret.log");
  });

  it("keeps stderr-first precedence in catch-path diagnostics", async () => {
    const bunRef = (globalThis as { Bun: typeof Bun }).Bun;
    const original = bunRef.spawnSync;
    bunRef.spawnSync = (() => {
      const err = new Error("ignored message") as Error & { stderr?: Uint8Array; stdout?: Uint8Array };
      err.stderr = Buffer.from("stderr catch");
      err.stdout = Buffer.from("stdout catch");
      throw err;
    }) as typeof bunRef.spawnSync;

    try {
      const execute = await loadToolExecute("../../adw_spec.ts");
      const result = await execute({ command: "list", adw_id: "a1b2c3d4" });
      const text = String(result);
      assertContains(text, "stderr catch");
      expect(text).not.toContain("stdout catch");
    } finally {
      bunRef.spawnSync = original;
    }
  });

  it("falls back to thrown message when stderr/stdout diagnostics are unavailable", async () => {
    const bunRef = (globalThis as { Bun: typeof Bun }).Bun;
    const original = bunRef.spawnSync;
    bunRef.spawnSync = (() => {
      throw new Error("message fallback diagnostic");
    }) as typeof bunRef.spawnSync;

    try {
      const execute = await loadToolExecute("../../adw_spec.ts");
      const result = await execute({ command: "list", adw_id: "a1b2c3d4" });
      assertContains(String(result), "message fallback diagnostic");
    } finally {
      bunRef.spawnSync = original;
    }
  });
});
