import { buildDollarFailure } from "../helpers/fixture-builders";

export const COMPACT_SCHEMA_FIELD_FIXTURES = {
  findFiles: {
    counted: ["path", "pattern"],
    exempt: ["options"],
    omitted: ["contentPattern", "contextLines", "filesWithMatches"],
  },
  runPytestBasicOmittedKeys: [
    "outputMode",
    "failFast",
    "testFilter",
    "covReport",
    "coverage",
    "coverageSource",
    "coverageThreshold",
    "durations",
    "durationsMin",
    "overrideIni",
    "pytestArgs",
  ],
  runPytestAdvancedOmittedKeys: [
    "outputMode",
    "failFast",
    "testFilter",
    "covReport",
    "durations",
    "durationsMin",
  ],
  runBunTestOmittedKeys: [
    "outputMode",
    "failFast",
    "testFilter",
  ],
  sparseOptionalInventory: {
    counted: ["command", "path", "limit"],
    exempt: ["help"],
    sourceLines: [
      "import { tool } from '@opencode-ai/plugin';",
      "export default tool({",
      "  args: {",
      "    command: tool.schema.enum(['show']),",
      "    path: tool.schema.string().optional(),",
      "    limit: tool.schema.number().optional(),",
      "    help: tool.schema.boolean().optional(),",
      "  },",
      "  async execute() {",
      "    return 'ok';",
      "  },",
      "});",
      "",
    ],
  },
} as const;

export const ERROR_PRECEDENCE_FIXTURES = {
  stderrFirst: {
    preferred: "stderr-first diagnostic",
    shadowed: "stdout shadow",
    failure: buildDollarFailure({
      stderr: "stderr-first diagnostic",
      stdout: "stdout shadow",
      message: "message fallback",
    }),
  },
  stdoutFirst: {
    preferred: "stdout-first diagnostic",
    shadowed: "stderr shadow",
    failure: buildDollarFailure({
      stdout: "stdout-first diagnostic",
      stderr: "stderr shadow",
      message: "message fallback",
    }),
  },
  stderrFallback: buildDollarFailure({
    stdout: "",
    stderr: "stderr fallback diagnostic",
    message: "message fallback",
  }),
  messageOnly: buildDollarFailure({
    stdout: "",
    stderr: "",
    message: "message-only diagnostic",
  }),
  jsonStdout: '{"ok":false,"error":"details"}',
} as const;

export const COMPATIBILITY_METADATA_FIXTURES = {
  approvedHistoricalConfig: {
    version: 1,
    wrappers: [{ name: "run_pytest", status: "exception_approved" }],
  },
  malformedMetadataText: '{\n  "wrappers": [\n',
  activeDisallowedConfig: {
    version: 1,
    wrappers: [],
  },
} as const;
