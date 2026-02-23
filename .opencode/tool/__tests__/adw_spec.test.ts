import assert from "node:assert/strict";
import { beforeEach, describe, it, mock as nodeMock } from "node:test";
import { mock as bunMock } from "bun:test";

const createSchema = (kind: string, extras: Record<string, unknown> = {}) => {
  return {
    kind,
    ...extras,
    optional() {
      return { ...this, optional: true };
    },
    describe(description: string) {
      return { ...this, description };
    },
  };
};

const schema = {
  enum: (values: string[]) => createSchema("enum", { values }),
  string: () => createSchema("string"),
  number: () => createSchema("number"),
  boolean: () => createSchema("boolean"),
};

const toolFn = (definition: unknown) => definition;
(toolFn as any).schema = schema;

const mock = (nodeMock as typeof bunMock).module ? nodeMock : bunMock;

mock.module("@opencode-ai/plugin", () => ({
  tool: toolFn,
}));

const encoder = new TextEncoder();
let spawnSyncImpl: (options: { cmd: string[] }) => {
  exitCode: number;
  stdout?: Uint8Array;
  stderr?: Uint8Array;
};

const bun = globalThis.Bun as any;
bun.spawnSync = (options: { cmd: string[] }) => spawnSyncImpl(options);

const adwSpecTool = (await import("../adw_spec")).default as {
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("adw_spec tool", () => {
  beforeEach(() => {
    spawnSyncImpl = () => ({
      exitCode: 0,
      stdout: encoder.encode("ok"),
      stderr: encoder.encode(""),
    });
  });

  it("returns success even when stdout contains Error text", async () => {
    spawnSyncImpl = () => ({
      exitCode: 0,
      stdout: encoder.encode("Error: not a failure"),
      stderr: encoder.encode(""),
    });

    const result = await adwSpecTool.execute({ command: "read", adw_id: "abc12345" });

    assert.ok(result.startsWith("ADW Spec Command: read"));
    assert.ok(result.includes("Error: not a failure"));
  });

  it("returns failed output when exit code is non-zero", async () => {
    spawnSyncImpl = () => ({
      exitCode: 2,
      stdout: encoder.encode("stdout error"),
      stderr: encoder.encode("stderr error"),
    });

    const result = await adwSpecTool.execute({ command: "read", adw_id: "abc12345" });

    assert.ok(result.startsWith("ADW Spec Command Failed (exit 2):"));
    assert.ok(result.includes("stderr error"));
  });

  it("round-trips write then read content", async () => {
    const store = new Map<string, string>();
    spawnSyncImpl = ({ cmd }) => {
      const command = cmd[4];
      const adwIdIndex = cmd.indexOf("--adw-id");
      const adwId = adwIdIndex >= 0 ? cmd[adwIdIndex + 1] : "";
      const contentIndex = cmd.indexOf("--content");
      const content = contentIndex >= 0 ? cmd[contentIndex + 1] : "";
      const key = `${adwId}:spec_content`;

      if (command === "write") {
        store.set(key, content);
        return { exitCode: 0, stdout: encoder.encode("written"), stderr: encoder.encode("") };
      }

      if (command === "read") {
        const value = store.get(key) ?? "";
        return { exitCode: 0, stdout: encoder.encode(value), stderr: encoder.encode("") };
      }

      return { exitCode: 0, stdout: encoder.encode(""), stderr: encoder.encode("") };
    };

    await adwSpecTool.execute({ command: "write", adw_id: "abc12345", content: "hello" });
    const result = await adwSpecTool.execute({ command: "read", adw_id: "abc12345" });

    assert.ok(result.includes("hello"));
  });
});
