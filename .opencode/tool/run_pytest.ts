/**
 * Pytest Runner Tool with Coverage
 *
 * Runs pytest with coverage and validation, returning either full output or summary.
 * This tool validates test results to prevent false positives.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Run pytest with coverage and comprehensive validation. Returns test results, coverage metrics, and validates test count to prevent false positives. Examples: run_pytest({minTests: 1700}), run_pytest({outputMode: 'summary'}), run_pytest({pytestArgs: ['adw/core/tests/', '--maxfail=1']}), run_pytest({timeout: 900})",
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode: 'full' (default, complete pytest output + summary), 'summary' (human-readable summary only), 'json' (structured data)"),
    minTests: tool.schema
      .number()
      .optional()
      .describe("Minimum expected test count for validation (default: 1). Warns if fewer tests pass. Example: 1700"),
    pytestArgs: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Additional arguments to pass to pytest. Examples: ['--maxfail=1'], ['adw/core/tests/'], ['-k', 'test_agent']"),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 600 = 10 minutes). Maximum time to allow pytest to run before terminating."),
  },
  async execute(args) {
    const outputMode = args.outputMode || "full";
    const minTests = args.minTests || 1;
    const pytestArgs = args.pytestArgs || [];
    const timeout = args.timeout || 600;

    // Build command
    const cmdParts = [
      "python3",
      `${import.meta.dir}/run_pytest.py`,
      `--output=${outputMode}`,
      `--min-tests=${minTests}`,
      `--timeout=${timeout}`
    ];

    // Add pytest arguments
    if (pytestArgs.length > 0) {
      cmdParts.push(...pytestArgs);
    }

    try {
      // Execute the Python script
      const result = await Bun.$`${cmdParts}`.text();
      return result;
    } catch (error: any) {
      // Pytest or validation failed - return the output anyway
      // The Python script provides detailed error information
      if (error.stdout) {
        return error.stdout.toString();
      }
      return `ERROR: Failed to run pytest: ${error.message}`;
    }
  },
});
