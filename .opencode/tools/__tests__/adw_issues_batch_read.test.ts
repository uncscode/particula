import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_issues_batch_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("passes metadata section through unchanged", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const result = await execute({
      adw_id: "A1B2C3D4",
      issue: "1",
      section: "metadata",
      raw: true,
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run adw spec batch read --adw-id a1b2c3d4 --issue 1 --section metadata --raw",
    );
  });
});
