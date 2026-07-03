import { mock } from "bun:test";
import { pathToFileURL } from "node:url";

type SchemaNode = {
  optional: () => SchemaNode;
  or: (_other: SchemaNode) => SchemaNode;
  describe: (_description: string) => SchemaNode;
};

const makeSchemaNode = (): SchemaNode => ({
  optional: () => makeSchemaNode(),
  or: () => makeSchemaNode(),
  describe: () => makeSchemaNode(),
});

type ToolDefinition = {
  args?: Record<string, unknown>;
  execute?: (args: Record<string, unknown>) => Promise<string> | string;
};

type PublicSchemaExpectation = {
  counted: string[];
  exempt?: string[];
  actualCounted?: string[];
  actualExempt?: string[];
};

let capturedDefinition: ToolDefinition | null = null;
let importCounter = 0;

const toolStub = (definition: ToolDefinition): ToolDefinition => {
  capturedDefinition = definition;
  return definition;
};

const registerToolPluginMock = (): void => {
  mock.module("@opencode-ai/plugin", () => ({
    tool: toolStub,
  }));
};

(toolStub as unknown as { schema: unknown }).schema = {
  string: () => makeSchemaNode(),
  number: () => makeSchemaNode(),
  boolean: () => makeSchemaNode(),
  enum: (_values: string[]) => makeSchemaNode(),
  array: (_node: unknown) => makeSchemaNode(),
  any: () => makeSchemaNode(),
};

registerToolPluginMock();

/** Reset captured wrapper definition between tests. */
export const resetCapturedToolDefinition = (): void => {
  capturedDefinition = null;
};

export const getCapturedToolDefinition = (): ToolDefinition | null => capturedDefinition;

const formatNoExecuteHandlerError = (specifier: string): string =>
  `ASSERT: no tool.execute handler registered for module '${specifier}'`;

const ensureObjectRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`ASSERT: ${label} must be an object-like record`);
  }
  return value as Record<string, unknown>;
};

export const getPublicSchemaKeys = (definition: ToolDefinition | null | undefined): string[] =>
  Object.keys(ensureObjectRecord(definition?.args, "tool.args schema")).sort();

export const assertPublicSchemaIncludesKeys = (
  definition: ToolDefinition | null | undefined,
  expectedKeys: string[],
): void => {
  const actualKeys = getPublicSchemaKeys(definition);
  const missingKeys = expectedKeys.filter((key) => !actualKeys.includes(key));
  if (missingKeys.length > 0) {
    throw new Error(
      `ASSERT: public schema missing expected keys: ${missingKeys.join(", ")} (actual: ${actualKeys.join(", ")})`,
    );
  }
};

export const assertPublicSchemaOmitsKeys = (
  definition: ToolDefinition | null | undefined,
  unexpectedKeys: string[],
): void => {
  const actualKeys = getPublicSchemaKeys(definition);
  const presentKeys = unexpectedKeys.filter((key) => actualKeys.includes(key));
  if (presentKeys.length > 0) {
    throw new Error(
      `ASSERT: public schema unexpectedly included keys: ${presentKeys.join(
        ", ",
      )} (actual: ${actualKeys.join(", ")})`,
    );
  }
};

export const assertCountedAndExemptFields = (
  definition: ToolDefinition | null | undefined,
  expectation: PublicSchemaExpectation,
): void => {
  const actualKeys = getPublicSchemaKeys(definition);
  const counted = [...expectation.counted].sort();
  const exempt = [...(expectation.exempt ?? [])].sort();
  const expectedKeys = [...counted, ...exempt].sort();
  if (actualKeys.join("\u0000") !== expectedKeys.join("\u0000")) {
    throw new Error(
      `ASSERT: public schema keys did not match counted+exempt expectations (expected: ${expectedKeys.join(
        ", ",
      )}; actual: ${actualKeys.join(", ")})`,
    );
  }

  if (expectation.actualCounted) {
    const actualCounted = [...expectation.actualCounted].sort();
    if (actualCounted.join("\u0000") !== counted.join("\u0000")) {
      throw new Error(
        `ASSERT: counted field classification did not match expectation (expected: ${counted.join(
          ", ",
        )}; actual: ${actualCounted.join(", ")})`,
      );
    }
  }

  if (expectation.actualExempt) {
    const actualExempt = [...expectation.actualExempt].sort();
    if (actualExempt.join("\u0000") !== exempt.join("\u0000")) {
      throw new Error(
        `ASSERT: exempt field classification did not match expectation (expected: ${exempt.join(
          ", ",
        )}; actual: ${actualExempt.join(", ")})`,
      );
    }
  }
};

/**
 * Loads a wrapper module and returns its registered execute handler.
 * Throws a deterministic assertion-style error when handler registration is missing.
 */
export const loadToolExecute = async (
  modulePath: string,
): Promise<(args: Record<string, unknown>) => Promise<string> | string> => {
  registerToolPluginMock();
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
  registerToolPluginMock();
  const url = pathToFileURL(absoluteModulePath).href;
  importCounter += 1;
  return loadToolExecuteFromSpecifier(`${url}?test=${importCounter}`);
};

const loadToolExecuteFromSpecifier = async (
  specifier: string,
): Promise<(args: Record<string, unknown>) => Promise<string> | string> => {
  resetCapturedToolDefinition();
  const importedModule = await import(specifier);
  if (capturedDefinition?.execute) {
    return capturedDefinition.execute;
  }

  const defaultExport = importedModule?.default as ToolDefinition | undefined;
  if (defaultExport?.execute) {
    return defaultExport.execute;
  }

  if (!capturedDefinition?.execute) {
    throw new Error(formatNoExecuteHandlerError(specifier));
  }
  return capturedDefinition.execute;
};
