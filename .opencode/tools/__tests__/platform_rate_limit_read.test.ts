import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setDollarText } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_rate_limit_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles rate-limit command", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    await execute({ command: "rate-limit" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run adw platform rate-limit");
  });
});
