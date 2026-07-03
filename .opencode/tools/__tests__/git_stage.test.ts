import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("git_stage wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
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

  it("rejects add stage_all plus files ambiguity", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "add", stage_all: true, files: ["src/a.ts"] });

    assertContains(String(result), "cannot combine 'stage_all' with 'files'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects invalid file token", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "add", files: ["-bad"] });
    assertContains(String(result), "Invalid files entry");
  });

  it("rejects empty add files payload", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "add", files: ["   "] });

    assertContains(String(result), "requires either 'stage_all' or 'files'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects restore with neither files nor staged", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "restore", files: ["  "] });

    assertContains(String(result), "requires 'files' or 'staged: true'");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects option-like worktree_path", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "restore", staged: true, worktree_path: "--repo=/tmp/x" });

    assertContains(String(result), "Invalid worktree_path: --repo=/tmp/x.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("assembles restore staged", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    await execute({ command: "restore", staged: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git restore --staged");
  });

  it("allows help mode without add target validation", async () => {
    const execute = await loadToolExecute("../../git_stage.ts");
    const result = await execute({ command: "add", help: true });

    expect(String(result)).toContain("Git Command: add (help)");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain("uv run --active adw git add --help");
  });
});
