import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setDollarText } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_pr_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("created");
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires title", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    const result = await execute({ command: "create-pr", head: "feature" });
    assertContains(String(result), "'title' is required");
  });

  it("validates adw_id", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    const result = await execute({ command: "create-pr", title: "T", head: "h", adw_id: "ABC" });
    assertContains(String(result), "8-character lowercase hex");
  });

  it("assembles create-pr", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    await execute({ command: "create-pr", title: "T", head: "h" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw platform create-pr --title T --head h");
  });
});
