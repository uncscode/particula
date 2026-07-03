import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("workflow_builder compatibility wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("delegates one read command and one mutate command", async () => {
    const execute = await loadToolExecute("../../workflow_builder.ts");

    setDollarText("read ok");
    expect(await execute({ command: "get", workflow_name: "patch" })).toBe("read ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      ".opencode/tools/workflow_builder.py",
      "get",
      "--workflow-name",
      "patch",
    ]);

    setDollarText("mutate ok");
    expect(
      await execute({ command: "create", workflow_name: "demo", description: "Demo flow" }),
    ).toBe("mutate ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      ".opencode/tools/workflow_builder.py",
      "create",
      "--workflow-name",
      "demo",
      "--description",
      "Demo flow",
    ]);
  });

  it("rejects unsupported or blank commands before delegation", async () => {
    const execute = await loadToolExecute("../../workflow_builder.ts");

    expect(await execute({ command: "bogus" })).toBe(
      "ERROR: workflow_builder does not support command 'bogus'. Use: create, add_step, remove_step, get, list, update, validate.",
    );
    expect(await execute({ command: "   " })).toBe(
      "ERROR: workflow_builder does not support command ''. Use: create, add_step, remove_step, get, list, update, validate.",
    );
    expect(getInvocations()).toEqual([]);
  });

  it("documents remove_step selectors as additive rather than mutually exclusive", async () => {
    const source = await Bun.file(new URL("../workflow_builder.ts", import.meta.url)).text();

    expect(source).toContain("Use this alone or together with step_name. At least one selector is required.");
    expect(source).toContain("Use this alone or together with step_index. At least one selector is required.");
    expect(source).toContain("If both are provided, backend precedence is preserved.");
    expect(source).not.toContain("Use this OR step_name, not both.");
    expect(source).not.toContain("Use this OR step_index, not both.");
  });

  it("prefers stdout in delegated failures and falls back to execution error formatting", async () => {
    const execute = await loadToolExecute("../../workflow_builder.ts");

    setDollarError({ stdout: "backend stdout", stderr: "backend stderr", message: "shadowed" });
    expect(await execute({ command: "list" })).toBe(
      "Workflow Builder Error:\nbackend stdout\n\nStderr:\nbackend stderr",
    );

    setDollarError({ message: "spawn exploded" });
    expect(await execute({ command: "list" })).toBe(
      "Workflow Builder Execution Error:\nspawn exploded",
    );
  });

  it("surfaces backend required-argument failures deterministically", async () => {
    const execute = await loadToolExecute("../../workflow_builder.ts");

    const cases: Array<{ args: Record<string, unknown>; error: string }> = [
      { args: { command: "create" }, error: "ERROR: 'create' requires workflow_name and description" },
      { args: { command: "add_step", workflow_name: "demo" }, error: "ERROR: 'add_step' requires workflow_name and step_json" },
      { args: { command: "remove_step" }, error: "ERROR: 'remove_step' requires workflow_name" },
      { args: { command: "get" }, error: "ERROR: 'get' requires workflow_name" },
      { args: { command: "update", workflow_name: "demo" }, error: "ERROR: 'update' requires workflow_name and workflow_json" },
      { args: { command: "validate" }, error: "ERROR: 'validate' requires workflow_json" },
    ];

    for (const testCase of cases) {
      expect(await execute(testCase.args)).toBe(testCase.error);
    }
    expect(getInvocations()).toEqual([]);
  });

  it("surfaces the remove_step selector edge case from the backend", async () => {
    const execute = await loadToolExecute("../../workflow_builder.ts");
    setDollarError({ stdout: "ERROR: 'remove_step' requires either step_index or step_name" });

    const result = await execute({ command: "remove_step", workflow_name: "demo" });
    assertContains(String(result), "ERROR: 'remove_step' requires either step_index or step_name");
  });

  it("omits blank optional flags from spawned argv", async () => {
    const execute = await loadToolExecute("../../workflow_builder.ts");

    setDollarText("create ok");
    await execute({
      command: "create",
      workflow_name: "demo",
      description: "Demo flow",
      version: "   ",
      workflow_type: "custom",
      output: "   ",
    });

    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      ".opencode/tools/workflow_builder.py",
      "create",
      "--workflow-name",
      "demo",
      "--description",
      "Demo flow",
    ]);
  });
});
