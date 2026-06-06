import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setDollarText } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_label_write wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires issue_number", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    const result = await execute({ command: "add-labels", labels: "x" });
    assertContains(String(result), "'issue_number' is required");
  });

  it("requires labels", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    const result = await execute({ command: "remove-labels", issue_number: "1" });
    assertContains(String(result), "'labels' is required");
  });

  it("assembles add-labels", async () => {
    const execute = await loadToolExecute("../../platform_label_write.ts");
    await execute({ command: "add-labels", issue_number: "1", labels: "x" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw platform add-labels 1 --labels x");
  });
});
