import assert from "node:assert/strict";
import { mock as bunMock } from "bun:test";
import { describe, it, mock as nodeMock } from "node:test";

const mock = (nodeMock as typeof bunMock).module ? nodeMock : bunMock;

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

const toolStub = Object.assign((definition: unknown) => definition, { schema });

mock.module("@opencode-ai/plugin", () => ({
  tool: toolStub,
}));

const buildMkdocsTool = (await import("../build_mkdocs")).default;

describe("build_mkdocs shim wiring", () => {
  it("exposes tool definition for test path compliance", () => {
    assert.ok(buildMkdocsTool);
    assert.ok(typeof buildMkdocsTool === "object" || typeof buildMkdocsTool === "function");

    const definition = buildMkdocsTool as Record<string, unknown>;
    assert.ok(
      Object.prototype.hasOwnProperty.call(definition, "description"),
      "tool should expose description"
    );
    assert.equal(typeof definition.description, "string");

    assert.ok(Object.prototype.hasOwnProperty.call(definition, "args"), "tool should expose args");
    assert.equal(typeof definition.args, "object");

    assert.ok(
      Object.prototype.hasOwnProperty.call(definition, "execute"),
      "tool should expose execute"
    );
    assert.equal(typeof definition.execute, "function");
  });
});
