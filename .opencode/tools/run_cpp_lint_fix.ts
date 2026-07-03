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
  description: `Run C++ lint fix mode (clang-format, clang-tidy, cppcheck) with always-on auto-fix semantics.

EXAMPLES:
- Fix all linters: run_cpp_lint_fix({ sourceDir: 'example_cpp_dev' })
- Fix format-only: run_cpp_lint_fix({ sourceDir: 'src', options: 'linters=clang-format output=summary' })

IMPORTANT:
- Mutating by design: this wrapper always appends --auto-fix
- clang-tidy requires compile_commands.json in buildDir`,
  args: {
    sourceDir: tool.schema.string(),
    buildDir: tool.schema.string().optional(),
    timeout: tool.schema.number().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    const legacyFieldError = rejectLegacyDirectFields(args, "run_cpp_lint_fix", ["outputMode", "linters"]);
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
      "--auto-fix",
    ];
    if (resolvedBuildDir) {
      cmdParts.push(`--build-dir=${resolvedBuildDir}`);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ lint fix completed but returned no output.";
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
        return `ERROR: C++ lint fix failed\n\n${stderr}${hint}`;
      }

      const diagnostics = buildCppLintDiagnostics({
        sourceDir: trimmedSourceDir,
        buildDir: resolvedBuildDir,
        linters,
        timeout,
        command: cmdParts,
      });
      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run C++ lint fix: ${message}\n${MISSING_SCRIPT_HINT}\n\n${diagnostics}`;
      }
      return `ERROR: Failed to run C++ lint fix: ${message}\n\n${diagnostics}`;
    }
  },
});
