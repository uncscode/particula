import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
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

  it("requires title and head including whitespace-only values", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");

    const missingTitle = await execute({ command: "create-pr", head: "feature" });
    assertContains(String(missingTitle), "'title' is required");

    const missingHead = await execute({ command: "create-pr", title: "T", head: "   " });
    assertContains(String(missingHead), "'head' is required");
  });

  it("validates adw_id and draft", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");

    const badAdwId = await execute({ command: "create-pr", title: "T", head: "h", adw_id: "ABC" });
    assertContains(String(badAdwId), "8-character lowercase hex");

    const badDraft = await execute({
      command: "create-pr",
      title: "T",
      head: "h",
      draft: "yes",
    } as any);
    assertContains(String(badDraft), "'draft' must be a boolean when provided");
  });

  it("assembles create-pr with optional flags", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    await execute({
      command: "create-pr",
      title: "T",
      head: "feature",
      body: "Body",
      base: "main",
      adw_id: "6f6fcf1b",
      draft: true,
      prefer_scope: "fork",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform create-pr --title T --head feature --base main --adw-id 6f6fcf1b --body Body --draft --prefer-scope fork",
    );
  });

  it("omits blank optional values", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    await execute({
      command: "create-pr",
      title: "T",
      head: "feature",
      body: "   ",
      base: "   ",
      adw_id: "   ",
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw platform create-pr --title T --head feature",
    );
  });

  it("help mode bypasses required validation", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    setDollarText("usage");
    const result = await execute({ command: "create-pr", help: true });
    expect(String(result)).toContain("usage");
  });

  it("wraps success output with marker compatibility", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    setDollarText("created body");
    const result = await execute({ command: "create-pr", title: "T", head: "feature" });
    expect(String(result)).toContain("PLATFORM_PR_CREATED");
    expect(String(result)).toContain("STATUS: SUCCESS");
  });

  it("preserves delegated marker output when already present", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    setDollarText("PLATFORM_PR_CREATED\n\nready");
    const result = await execute({ command: "create-pr", title: "T", head: "feature" });
    expect(String(result)).toBe("PLATFORM_PR_CREATED\n\nready");
  });

  it("returns deterministic failure envelope", async () => {
    const execute = await loadToolExecute("../../platform_pr_write.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));
    const result = await execute({ command: "create-pr", title: "T", head: "feature" });
    const text = String(result);

    assertContains(text, "PLATFORM_PR_FAILED");
    assertContains(text, "STATUS: FAILED");
    assertContains(text, "fatal stderr");
    expect(text.indexOf("fatal stderr")).toBeLessThan(text.indexOf("shadow stdout"));
    assertErrorPrefix(text, "PLATFORM_PR_FAILED");
  });
});
