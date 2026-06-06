import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setDollarText } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_issue_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires title for create-issue", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    const result = await execute({ command: "create-issue" });
    assertContains(String(result), "'title' is required");
  });

  it("requires issue_number for update-issue", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    const result = await execute({ command: "update-issue", title: "x" });
    assertContains(String(result), "'issue_number' is required");
  });

  it("assembles create-issue", async () => {
    const execute = await loadToolExecute("../../platform_issue_write.ts");
    await execute({ command: "create-issue", title: "T", body: "B" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw platform create-issue --title T --body B");
  });
});
