/**
 * C++ Linter Runner Tool
 *
 * Wraps the Python backing script run_cpp_linters.py to execute clang-format,
 * clang-tidy, and cppcheck with optional auto-fixing and structured output.
 * Mirrors run_ctest.ts/run_linters.ts patterns for consistent OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_cpp_linters.py (dependency #1365).";

export default tool({
  description: `Run C++ linters (clang-format, clang-tidy, cppcheck) with optional auto-fix.

EXAMPLES:
- Check all linters: run_cpp_linters({ sourceDir: 'example_cpp_dev' })
- Format only: run_cpp_linters({ sourceDir: 'src', linters: ['clang-format'] })
- Auto-fix formatting: run_cpp_linters({ sourceDir: 'src', autoFix: true })
- clang-tidy (requires compile_commands.json): run_cpp_linters({ sourceDir: 'src', buildDir: 'build', linters: ['clang-tidy'] })
- JSON output: run_cpp_linters({ sourceDir: 'src', outputMode: 'json' })

IMPORTANT:
- clang-tidy requires compile_commands.json in buildDir (cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
- Linters must be installed and on PATH; missing linters are skipped with a warning
- autoFix applies clang-format -i and clang-tidy --fix when available`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe(
        "Output mode: 'summary' (default, human-readable), 'full' (complete linter output), 'json' (structured data).",
      ),
    sourceDir: tool.schema
      .string()
      .describe(
        "Directory containing C++ sources to lint (required). Example: 'src', 'example_cpp_dev/src'.",
      ),
    buildDir: tool.schema
      .string()
      .optional()
      .describe(
        "Build directory containing compile_commands.json (required for clang-tidy). Example: 'build', 'example_cpp_dev/build'.",
      ),
    linters: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe(
        "Specific linters to run: ['clang-format', 'clang-tidy', 'cppcheck']. Default: all available linters.",
      ),
    autoFix: tool.schema
      .boolean()
      .optional()
      .describe(
        "Automatically fix issues where possible (default: false). Applies clang-format -i and clang-tidy --fix when supported.",
      ),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds for all linters (default: 300). Must be positive."),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const sourceDir = args.sourceDir as string | undefined;
    const buildDir = args.buildDir as string | undefined;
    const linters = args.linters || ["clang-format", "clang-tidy", "cppcheck"];
    const autoFix = args.autoFix || false;
    const timeout = args.timeout ?? 300;

    if (!sourceDir) {
      return "ERROR: sourceDir is required. Provide the directory containing C++ source files to lint.";
    }

    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_cpp_linters.py`,
      `--source-dir=${sourceDir}`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
      `--linters=${linters.join(",")}`,
    ];

    if (buildDir) {
      cmdParts.push(`--build-dir=${buildDir}`);
    }

    if (autoFix) {
      cmdParts.push("--auto-fix");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ linters completed but returned no output.";
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
        return `ERROR: C++ linters failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run C++ linters: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run C++ linters: ${message}`;
    }
  },
});
