/**
 * C++ Linter Runner Tool
 *
 * Wraps the Python backing script run_cpp_linters.py to execute clang-format,
 * clang-tidy, and cppcheck with optional auto-fixing and structured output.
 * Mirrors run_ctest.ts/run_linters.ts patterns for consistent OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

import {
  buildCppLintDiagnostics,
  parseCppLintOptions,
  rejectLegacyDirectFields,
  resolveExistingDirectoryWithinRepo,
  sanitizeDiagnosticValue,
  SUPPORTED_CPP_LINTERS,
  SUPPORTED_CPP_LINTER_SET,
} from "./run_cpp_wrapper_shared";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cpp_linters.py (dependency #1365).";
const DEFAULT_TIMEOUT_SECONDS = 300;
const MIN_TIMEOUT_SECONDS = 1;
const MAX_TIMEOUT_SECONDS = 3_600;

export default tool({
  description: `Run C++ linters (clang-format, clang-tidy, cppcheck) with optional auto-fix.

EXAMPLES:
- Check all linters: run_cpp_linters({ sourceDir: 'example_cpp_dev' })
- Format only: run_cpp_linters({ sourceDir: 'src', options: 'linters=clang-format' })
- Auto-fix formatting: run_cpp_linters({ sourceDir: 'src', autoFix: true })
- clang-tidy (requires compile_commands.json): run_cpp_linters({ sourceDir: 'src', buildDir: 'build', options: 'linters=clang-tidy output=full' })
- JSON output: run_cpp_linters({ sourceDir: 'src', options: 'output=json' })

IMPORTANT:
- clang-tidy requires compile_commands.json in buildDir (cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
- Linters must be installed and on PATH; missing linters are skipped with a warning
- autoFix applies clang-format -i and clang-tidy --fix when available`,
  args: {
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
    options: tool.schema
      .string()
      .optional()
      .describe("Bounded options: output=<summary|full|json>, linters=<clang-format,clang-tidy,cppcheck comma-list>."),
  },
  async execute(args) {
    const legacyFieldError = rejectLegacyDirectFields(args, "run_cpp_linters", ["outputMode", "linters"]);
    if (legacyFieldError) {
      return legacyFieldError;
    }

    const parsedOptions = parseCppLintOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const sourceDir = args.sourceDir as string | undefined;
    const buildDir = args.buildDir as string | undefined;
    const linters = parsedOptions.options.linters ?? [...SUPPORTED_CPP_LINTERS];
    const autoFix = args.autoFix || false;
    const timeout = args.timeout ?? DEFAULT_TIMEOUT_SECONDS;

    const trimmedSourceDir = typeof sourceDir === "string" ? sourceDir.trim() : "";
    if (!trimmedSourceDir) {
      return "ERROR: sourceDir is required. Provide the directory containing C++ source files to lint.";
    }
    const sourceDirResult = resolveExistingDirectoryWithinRepo(trimmedSourceDir, "sourceDir");
    if (!sourceDirResult.ok) {
      return sourceDirResult.error;
    }

    const trimmedBuildDir = typeof buildDir === "string" ? buildDir.trim() : "";
    let resolvedBuildDir: string | undefined;
    if (trimmedBuildDir) {
      const buildDirResult = resolveExistingDirectoryWithinRepo(trimmedBuildDir, "buildDir");
      if (!buildDirResult.ok) {
        return buildDirResult.error;
      }
      resolvedBuildDir = buildDirResult.path;
    }

    if (!Number.isInteger(timeout)) {
      return `ERROR: Timeout must be an integer in seconds (received ${timeout}).`;
    }
    if (timeout < MIN_TIMEOUT_SECONDS || timeout > MAX_TIMEOUT_SECONDS) {
      return `ERROR: Timeout must be between ${MIN_TIMEOUT_SECONDS} and ${MAX_TIMEOUT_SECONDS} seconds (received ${timeout}).`;
    }

    if (!Array.isArray(linters) || linters.length === 0) {
      return "ERROR: linters must be a non-empty array. Valid values: clang-format, clang-tidy, cppcheck.";
    }
    const invalidLinters = linters.filter((linter) => !SUPPORTED_CPP_LINTER_SET.has(linter));
    if (invalidLinters.length > 0) {
      return `ERROR: Unsupported linter(s): ${invalidLinters.join(", ")}. Valid values: ${SUPPORTED_CPP_LINTERS.join(", ")}.`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_cpp_linters.py`,
      `--source-dir=${sourceDirResult.path}`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
      `--linters=${linters.join(",")}`,
    ];

    if (resolvedBuildDir) {
      cmdParts.push(`--build-dir=${resolvedBuildDir}`);
    }

    if (autoFix) {
      cmdParts.push("--auto-fix");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ linters completed but returned no output.";
    } catch (error: any) {
      const stdout = sanitizeDiagnosticValue(error?.stdout?.toString?.() || "");
      const stderr = sanitizeDiagnosticValue(error?.stderr?.toString?.() || "");
      const message = sanitizeDiagnosticValue(error?.message || "Unknown error");
      const combinedLower = `${stderr} ${stdout} ${message}`.toLowerCase();

      if (stdout.trim()) {
        return stdout;
      }

      if (stderr.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        return `ERROR: C++ linters failed\n\n${stderr}${hint}`;
      }

      const diagnostics = buildCppLintDiagnostics({
        sourceDir: trimmedSourceDir,
        buildDir: resolvedBuildDir,
        linters,
        autoFix,
        timeout,
        command: cmdParts,
      });

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run C++ linters: ${message}\n${MISSING_SCRIPT_HINT}\n\n${diagnostics}`;
      }

      return `ERROR: Failed to run C++ linters: ${message}\n\n${diagnostics}`;
    }
  },
});
