import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("git_stage wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires add target mode", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "add" });
    assertContains(String(result), "requires either 'stage_all' or 'files'");
  });

  it("rejects invalid file token", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "add", files: ["-bad"] });
    assertContains(String(result), "Invalid files entry");
  });

  it("assembles restore staged", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    await execute({ command: "restore", staged: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw git restore --staged");
  });

});
