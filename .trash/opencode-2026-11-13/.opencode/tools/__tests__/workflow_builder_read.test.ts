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

describe("workflow_builder_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("accepts list, get, and validate", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");

    for (const [command, args] of [
      ["list", {}],
      ["get", { workflow_name: "patch" }],
      ["validate", { workflow_json: "{}" }],
    ] as const) {
      setDollarText(`${command} ok`);
      expect(await execute({ command, ...args })).toBe(`${command} ok`);
      expect(getInvocations().at(-1)?.args[2]).toBe(command);
    }
  });

  it("includes output only when provided for supported read commands", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");

    setDollarText("get ok");
    await execute({ command: "get", workflow_name: "patch", output: "json" });

    expect(getInvocations().at(-1)?.args).toEqual([
      "python3",
      ".opencode/tools/workflow_builder.py",
      "get",
      "--workflow-name",
      "patch",
      "--output",
      "json",
    ]);
  });

  it("rejects mutate commands with deterministic ERROR text", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");

    expect(await execute({ command: "create" })).toBe(
      "ERROR: workflow_builder_read does not support command 'create'. Use: list, get, validate.",
    );
    expect(await execute({ command: "add_step" })).toBe(
      "ERROR: workflow_builder_read does not support command 'add_step'. Use: list, get, validate.",
    );
    expect(await execute({ command: "remove_step" })).toBe(
      "ERROR: workflow_builder_read does not support command 'remove_step'. Use: list, get, validate.",
    );
    expect(await execute({ command: "update" })).toBe(
      "ERROR: workflow_builder_read does not support command 'update'. Use: list, get, validate.",
    );
  });

  it("rejects mutate-only payloads at the command gate without delegating", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");

    expect(
      await execute({
        command: "create",
        workflow_name: "demo",
        description: "demo",
        workflow_type: "patch",
      }),
    ).toBe("ERROR: workflow_builder_read does not support command 'create'. Use: list, get, validate.");
    expect(getInvocations()).toEqual([]);
  });

  it("rejects blank or whitespace command input after normalization", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");
    expect(await execute({ command: "   " })).toBe(
      "ERROR: workflow_builder_read does not support command ''. Use: list, get, validate.",
    );
  });

  it("preserves delegated failure envelopes for supported read commands", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");

    setDollarError({ stdout: "backend stdout", stderr: "backend stderr", message: "shadowed" });
    expect(await execute({ command: "list" })).toBe(
      "Workflow Builder Error:\nbackend stdout\n\nStderr:\nbackend stderr",
    );

    setDollarError({ message: "spawn exploded" });
    expect(await execute({ command: "get", workflow_name: "patch" })).toBe(
      "Workflow Builder Execution Error:\nspawn exploded",
    );
  });

  it("surfaces backend required-argument failures for supported read commands", async () => {
    const execute = await loadToolExecute("../../workflow_builder_read.ts");

    const cases: Array<{ args: Record<string, unknown>; error: string }> = [
      { args: { command: "get" }, error: "ERROR: 'get' requires workflow_name" },
      { args: { command: "get", workflow_name: "   " }, error: "ERROR: 'get' requires workflow_name" },
      { args: { command: "validate" }, error: "ERROR: 'validate' requires workflow_json" },
      { args: { command: "validate", workflow_json: "   " }, error: "ERROR: 'validate' requires workflow_json" },
    ];

    for (const testCase of cases) {
      expect(await execute(testCase.args)).toBe(testCase.error);
    }
    expect(getInvocations()).toEqual([]);
  });
});
