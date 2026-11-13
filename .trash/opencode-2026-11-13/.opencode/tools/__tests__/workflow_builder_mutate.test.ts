import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("workflow_builder_mutate wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("accepts create, add_step, remove_step, and update", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");

    for (const [command, args] of [
      ["create", { workflow_name: "demo", description: "demo" }],
      ["add_step", { workflow_name: "demo", step_json: "{}" }],
      ["remove_step", { workflow_name: "demo", step_index: 0 }],
      ["update", { workflow_name: "demo", workflow_json: "{}" }],
    ] as const) {
      setDollarText(`${command} ok`);
      expect(await execute({ command, ...args })).toBe(`${command} ok`);
      expect(getInvocations().at(-1)?.args[2]).toBe(command);
    }
  });

  it("preserves meaningful optional mutate flags while omitting blank ones", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");

    setDollarText("create ok");
    await execute({
      command: "create",
      workflow_name: "demo",
      description: "demo",
      version: "1.2.0",
      workflow_type: "patch",
      output: "json",
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      ".opencode/tools/workflow_builder.py",
      "create",
      "--workflow-name",
      "demo",
      "--description",
      "demo",
      "--version",
      "1.2.0",
      "--workflow-type",
      "patch",
      "--output",
      "json",
    ]);

    setDollarText("add_step ok");
    await execute({
      command: "add_step",
      workflow_name: "demo",
      step_json: "{}",
      position: 2,
      output: "   ",
    });
    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      ".opencode/tools/workflow_builder.py",
      "add_step",
      "--workflow-name",
      "demo",
      "--step-json",
      "{}",
      "--position",
      "2",
    ]);
  });

  it("rejects read commands with deterministic ERROR text", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");

    expect(await execute({ command: "list" })).toBe(
      "ERROR: workflow_builder_mutate does not support command 'list'. Use: create, add_step, remove_step, update.",
    );
    expect(await execute({ command: "get" })).toBe(
      "ERROR: workflow_builder_mutate does not support command 'get'. Use: create, add_step, remove_step, update.",
    );
    expect(await execute({ command: "validate" })).toBe(
      "ERROR: workflow_builder_mutate does not support command 'validate'. Use: create, add_step, remove_step, update.",
    );
  });

  it("rejects read-only payloads at the command gate without delegating", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");

    expect(await execute({ command: "validate", workflow_json: "{}", output: "json" })).toBe(
      "ERROR: workflow_builder_mutate does not support command 'validate'. Use: create, add_step, remove_step, update.",
    );
    expect(getInvocations()).toEqual([]);
  });

  it("rejects blank or whitespace command input after normalization", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");
    expect(await execute({ command: "   " })).toBe(
      "ERROR: workflow_builder_mutate does not support command ''. Use: create, add_step, remove_step, update.",
    );
  });

  it("preserves delegated failure envelopes for supported mutate commands", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");

    setDollarError({ stdout: "backend stdout", stderr: "backend stderr", message: "shadowed" });
    expect(await execute({ command: "create", workflow_name: "demo", description: "Demo" })).toBe(
      "Workflow Builder Error:\nbackend stdout\n\nStderr:\nbackend stderr",
    );

    setDollarError({ message: "spawn exploded" });
    expect(await execute({ command: "update", workflow_name: "demo", workflow_json: "{}" })).toBe(
      "Workflow Builder Execution Error:\nspawn exploded",
    );
  });

  it("surfaces backend required-argument failures for supported mutate commands", async () => {
    const execute = await loadToolExecute("../../workflow_builder_mutate.ts");

    const cases: Array<{ args: Record<string, unknown>; error: string }> = [
      { args: { command: "create" }, error: "ERROR: 'create' requires workflow_name and description" },
      { args: { command: "create", workflow_name: "demo", description: "   " }, error: "ERROR: 'create' requires workflow_name and description" },
      { args: { command: "add_step", workflow_name: "demo" }, error: "ERROR: 'add_step' requires workflow_name and step_json" },
      { args: { command: "remove_step" }, error: "ERROR: 'remove_step' requires workflow_name" },
      { args: { command: "remove_step", workflow_name: "demo", step_name: "   " }, error: "ERROR: 'remove_step' requires either step_index or step_name" },
      { args: { command: "update", workflow_name: "demo" }, error: "ERROR: 'update' requires workflow_name and workflow_json" },
      { args: { command: "update", workflow_name: "   ", workflow_json: "{}" }, error: "ERROR: 'update' requires workflow_name and workflow_json" },
    ];

    for (const testCase of cases) {
      expect(await execute(testCase.args)).toBe(testCase.error);
    }
    expect(getInvocations()).toEqual([]);
  });
});
