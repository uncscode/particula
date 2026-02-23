/**
 * Bun Test Runner Tool
 *
 * Wraps the Python backing script run_bun_test.py to execute bun test with
 * filtering and structured outputs. Mirrors the run_ctest.ts pattern.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_bun_test.py.";

const normalizeFailFast = (value: unknown): boolean => {
  if (typeof value === "string") {
    return value.toLowerCase() === "true";
  }
  return Boolean(value);
};

export default tool({
  description: `Run bun test for TypeScript test execution with parsing and validation.

EXAMPLES:
- Run all tests: run_bun_test({ testPath: '__tests__/' })
- Single test file: run_bun_test({ testPath: '__tests__/get_datetime.test.ts' })
- Filter by name: run_bun_test({ testFilter: 'datetime' })
- Fail fast: run_bun_test({ testPath: '__tests__/', failFast: true })
- JSON output: run_bun_test({ outputMode: 'json' })

IMPORTANT: Requires bun to be installed on the host system.
NOTE: Default working directory is .opencode/tool/ (where package.json lives).`,
  args: {
    testPath: tool.schema
      .string()
      .optional()
      .describe("Path to test file or directory. Example: '__tests__/', '__tests__/get_datetime.test.ts'."),
    testFilter: tool.schema
      .string()
      .optional()
      .describe("Test name pattern filter (maps to --filter). Example: 'datetime'."),
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode: 'summary' (default), 'full', 'json'."),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 300). Must be positive."),
    minTests: tool.schema
      .number()
      .optional()
      .describe("Minimum expected test count (default: 1). Must be positive."),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory override (default: .opencode/tool/)."),
    failFast: tool.schema
      .boolean()
      .optional()
      .describe("Stop on first failure (maps to --bail)."),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const testPath = args.testPath as string | undefined;
    const testFilter = args.testFilter as string | undefined;
    const timeout = args.timeout ?? 300;
    const minTests = args.minTests ?? 1;
    const cwd = args.cwd as string | undefined;
    const failFast = normalizeFailFast(args.failFast);

    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    if (minTests <= 0) {
      return `ERROR: minTests must be positive (received ${minTests}).`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_bun_test.py`,
      `--output=${outputMode}`,
      `--min-tests=${minTests}`,
      `--timeout=${timeout}`,
    ];

    if (testPath) {
      cmdParts.push(`--test-path=${testPath}`);
    }

    if (testFilter) {
      cmdParts.push(`--filter=${testFilter}`);
    }

    if (failFast) {
      cmdParts.push("--bail");
    }

    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "bun test completed but returned no output.";
    } catch (error: any) {
      const stdout = error?.stdout?.toString?.() || "";
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";
      const combinedLower = `${stderr} ${message}`.toLowerCase();

      if (stdout.trim()) {
        return stdout;
      }

      if (stderr.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        return `ERROR: Bun test failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run bun test: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run bun test: ${message}`;
    }
  },
});
