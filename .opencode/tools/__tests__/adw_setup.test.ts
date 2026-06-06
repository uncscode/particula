import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse, getInvocations } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_setup wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires command unless wizard/help", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const result = await execute({});
    assertContains(String(result), "requires 'command'");
  });

  it("enforces args command allowlist", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const result = await execute({ command: "env", args: ["--strict"] });
    assertContains(String(result), "'args' is only supported");
  });

  it("rejects wizard+command combination", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const result = await execute({ wizard: true, command: "env" });
    assertContains(String(result), "cannot be combined");
  });

  it("assembles env command", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    await execute({ command: "env", with_templates: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw setup env --with-templates");
  });

  it("prefers stderr over stdout in non-zero failure diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ command: "env" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
