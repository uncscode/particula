import assert from "node:assert/strict";
import { beforeEach, describe, it, mock as bunMock } from "bun:test";
import { mock as nodeMock } from "node:test";

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

let executionBehavior: { kind: "success"; output?: string } | { kind: "error"; error: any } = {
  kind: "success",
};
let lastCommand: (string | number)[] = [];

const setExecutionSuccess = (output?: string) => {
  executionBehavior = { kind: "success", output };
};

const setExecutionError = (error: any) => {
  executionBehavior = { kind: "error", error };
};

const bunDollar = (_strings: TemplateStringsArray, ...values: unknown[]) => {
  const parts = (values[0] || []) as (string | number)[];
  lastCommand = parts;
  return {
    text: async () => {
      if (executionBehavior.kind === "error") {
        throw executionBehavior.error;
      }
      if (executionBehavior.output !== undefined) {
        return executionBehavior.output;
      }
      return parts.join(" ");
    },
  };
};

const bunGlobal = (globalThis as { Bun?: { $?: (...args: unknown[]) => unknown } }).Bun;
if (bunGlobal) {
  bunGlobal.$ = bunDollar;
} else {
  (globalThis as any).Bun = { $: bunDollar };
}

const buildMkdocsTool = (await import("../build_mkdocs")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

const MISSING_HINT = "Missing backing script .opencode/tool/build_mkdocs.py";

describe("build_mkdocs tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
  });

  it("includes key description notes and examples", () => {
    assert.ok(buildMkdocsTool.description.toLowerCase().includes("mkdocs"));
    assert.ok(buildMkdocsTool.description.includes("EXAMPLES"));
    assert.ok(buildMkdocsTool.description.includes("IMPORTANT"));
  });

  it("defines all expected arguments with enum values", () => {
    const { args } = buildMkdocsTool;
    assert.ok(args.outputMode);
    assert.deepEqual(args.outputMode.values, ["summary", "full", "json"]);
    assert.ok(args.timeout);
    assert.ok(String(args.timeout.description || "").toLowerCase().includes("positive"));
    assert.ok(args.cwd);
    assert.ok(args.strict);
    assert.ok(args.clean);
    assert.ok(args.configFile);
    assert.ok(args.validateOnly);
  });

  it("builds command with defaults when optional args omitted", async () => {
    const result = await buildMkdocsTool.execute({});

    assert.ok(result.length > 0);
    assert.ok(lastCommand.includes("python3"));
    assert.ok(lastCommand.includes("--output=summary"));
    assert.ok(lastCommand.includes("--timeout=120"));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--cwd=")));
    assert.ok(!lastCommand.includes("--strict"));
    assert.ok(!lastCommand.includes("--no-clean"));
    assert.ok(!lastCommand.some((part) => String(part).startsWith("--config-file=")));
    assert.ok(!lastCommand.includes("--validate-only"));
  });

  it("maps parameters into the command", async () => {
    await buildMkdocsTool.execute({
      outputMode: "json",
      timeout: 42,
      cwd: "/repo",
      strict: true,
      clean: false,
      configFile: "docs/mkdocs.yml",
      validateOnly: true,
    });

    assert.ok(lastCommand.includes("--output=json"));
    assert.ok(lastCommand.includes("--timeout=42"));
    assert.ok(lastCommand.includes("--cwd=/repo"));
    assert.ok(lastCommand.includes("--strict"));
    assert.ok(lastCommand.includes("--no-clean"));
    assert.ok(lastCommand.includes("--config-file=docs/mkdocs.yml"));
    assert.ok(lastCommand.includes("--validate-only"));
  });

  it("returns fallback message when command output is empty", async () => {
    setExecutionSuccess("");
    const result = await buildMkdocsTool.execute({});

    assert.equal(result, "mkdocs build completed but returned no output.");
  });

  it("rejects non-positive timeout before executing", async () => {
    const timeoutResult = await buildMkdocsTool.execute({ timeout: 0 });
    assert.ok(timeoutResult.includes("Timeout must be positive"));
    assert.equal(lastCommand.length, 0);
  });

  it("prefers stdout over stderr and message on errors", async () => {
    setExecutionError({ stdout: "from-stdout", stderr: "from-stderr", message: "msg" });
    const result = await buildMkdocsTool.execute({});

    assert.equal(result, "from-stdout");
  });

  it("uses stderr with prefix when stdout is empty", async () => {
    setExecutionError({ stdout: "", stderr: "from-stderr", message: "msg" });
    const result = await buildMkdocsTool.execute({});

    assert.ok(result.startsWith("ERROR: MkDocs build failed"));
    assert.ok(result.includes("from-stderr"));
  });

  it("falls back to message and adds hint on ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "ENOENT build_mkdocs.py missing" });
    const result = await buildMkdocsTool.execute({});

    assert.ok(result.includes("Failed to run mkdocs build"));
    assert.ok(result.includes(MISSING_HINT));
  });

  it("falls back to message when no stdout or stderr is provided", async () => {
    setExecutionError({ stdout: "", stderr: "", message: "generic failure" });
    const result = await buildMkdocsTool.execute({});

    assert.ok(result.includes("ERROR: Failed to run mkdocs build: generic failure"));
    assert.ok(!result.includes(MISSING_HINT));
  });

  it("adds missing-script hint when stderr contains ENOENT", async () => {
    setExecutionError({ stdout: "", stderr: "ENOENT build_mkdocs.py missing", message: "ENOENT" });
    const result = await buildMkdocsTool.execute({});

    assert.ok(result.includes(MISSING_HINT));
  });
});
