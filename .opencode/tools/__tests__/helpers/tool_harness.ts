import { mock } from "bun:test";
import { pathToFileURL } from "node:url";

type SchemaNode = {
  optional: () => SchemaNode;
  describe: (_description: string) => SchemaNode;
};

const makeSchemaNode = (): SchemaNode => ({
  optional: () => makeSchemaNode(),
  describe: () => makeSchemaNode(),
});

type ToolDefinition = {
  execute?: (args: Record<string, unknown>) => Promise<string> | string;
};

let capturedDefinition: ToolDefinition | null = null;
let importCounter = 0;

const toolStub = (definition: ToolDefinition): ToolDefinition => {
  capturedDefinition = definition;
  return definition;
};

(toolStub as unknown as { schema: unknown }).schema = {
  string: () => makeSchemaNode(),
  number: () => makeSchemaNode(),
  boolean: () => makeSchemaNode(),
  enum: (_values: string[]) => makeSchemaNode(),
  array: (_node: unknown) => makeSchemaNode(),
  any: () => makeSchemaNode(),
};

mock.module("@opencode-ai/plugin", () => ({
  tool: toolStub,
}));

/** Reset captured wrapper definition between tests. */
export const resetCapturedToolDefinition = (): void => {
  capturedDefinition = null;
};

/**
 * Loads a wrapper module and returns its registered execute handler.
 * Throws a deterministic assertion-style error when handler registration is missing.
 */
export const loadToolExecute = async (
  modulePath: string,
): Promise<(args: Record<string, unknown>) => Promise<string> | string> => {
  importCounter += 1;
  return loadToolExecuteFromSpecifier(`${modulePath}?test=${importCounter}`);
};

/**
 * Loads a wrapper module from an absolute filesystem path and returns execute handler.
 * Useful for symlink-path import scenarios where relative imports are insufficient.
 */
export const loadToolExecuteFromAbsolutePath = async (
  absoluteModulePath: string,
): Promise<(args: Record<string, unknown>) => Promise<string> | string> => {
  const url = pathToFileURL(absoluteModulePath).href;
  importCounter += 1;
  return loadToolExecuteFromSpecifier(`${url}?test=${importCounter}`);
};

const loadToolExecuteFromSpecifier = async (
  specifier: string,
): Promise<(args: Record<string, unknown>) => Promise<string> | string> => {
  resetCapturedToolDefinition();
  await import(specifier);
  if (!capturedDefinition?.execute) {
    throw new Error(`ASSERT: no tool.execute handler registered for module '${specifier}'`);
  }
  return capturedDefinition.execute;
};
