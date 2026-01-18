/**
 * CTest Runner Tool
 *
 * Wraps the Python backing script run_ctest.py to execute CTest with
 * filtering, parallelism, and structured outputs. Mirrors the
 * run_pytest.ts pattern for consistency across OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_ctest.py (dependency #1363).";

export default tool({
  description: `Run CTest for C++ test execution with parsing and validation.

EXAMPLES:
- Run all tests: run_ctest({ buildDir: 'build' })
- Filter tests: run_ctest({ buildDir: 'build', testFilter: 'test_add' })
- Exclude tests: run_ctest({ buildDir: 'build', excludeFilter: 'slow' })
- Parallel execution: run_ctest({ buildDir: 'build', parallel: 4 })
- With minimum test count: run_ctest({ buildDir: 'build', minTests: 5 })
- JSON output: run_ctest({ buildDir: 'build', outputMode: 'json' })

IMPORTANT: buildDir must point to a CMake build directory containing CTestTestfile.cmake.
NOTE: Use testFilter for -R pattern (include), excludeFilter for -E pattern (exclude).`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe(
        "Output mode: 'summary' (default, concise), 'full' (complete CTest output + summary), 'json' (structured).",
      ),
    buildDir: tool.schema
      .string()
      .describe(
        "CMake build directory containing CTestTestfile.cmake (required). Example: 'build', 'example_cpp_dev/build'.",
      ),
    testFilter: tool.schema
      .string()
      .optional()
      .describe("Regex pattern to include tests (maps to ctest -R). Example: 'test_add', 'math_*'."),
    excludeFilter: tool.schema
      .string()
      .optional()
      .describe("Regex pattern to exclude tests (maps to ctest -E). Example: 'slow', 'integration_*'."),
    parallel: tool.schema
      .number()
      .optional()
      .describe("Number of parallel test jobs (maps to ctest -j). Must be positive. Example: 4."),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 300 = 5 minutes). Must be positive."),
    minTests: tool.schema
      .number()
      .optional()
      .describe(
        "Minimum expected test count for validation (default: 1). Set higher for full suite validation. Must be positive.",
      ),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const buildDir = args.buildDir as string | undefined;
    const testFilter = args.testFilter as string | undefined;
    const excludeFilter = args.excludeFilter as string | undefined;
    const parallel = args.parallel as number | undefined;
    const timeout = args.timeout ?? 300;
    const minTests = args.minTests ?? 1;

    if (!buildDir) {
      return "ERROR: buildDir is required. Provide the CMake build directory containing CTestTestfile.cmake.";
    }

    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    if (minTests <= 0) {
      return `ERROR: minTests must be positive (received ${minTests}).`;
    }

    if (parallel !== undefined && parallel <= 0) {
      return `ERROR: parallel must be positive (received ${parallel}).`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_ctest.py`,
      `--build-dir=${buildDir}`,
      `--output=${outputMode}`,
      `--min-tests=${minTests}`,
      `--timeout=${timeout}`,
    ];

    if (testFilter) {
      cmdParts.push("-R", testFilter);
    }

    if (excludeFilter) {
      cmdParts.push("-E", excludeFilter);
    }

    if (parallel !== undefined) {
      cmdParts.push("-j", parallel);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "CTest completed but returned no output.";
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
        return `ERROR: CTest failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run CTest: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run CTest: ${message}`;
    }
  },
});
