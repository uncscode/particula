/**
 * Sanitizer Runner Tool
 *
 * Wraps the Python backing script run_sanitizers.py to execute ASAN, TSAN, or
 * UBSAN enabled binaries with structured outputs. Mirrors the patterns used by
 * run_ctest.ts and run_cpp_linters.ts for consistency across OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_sanitizers.py (dependency #1379).";
const ALLOWED_OUTPUT_MODES = ["summary", "full", "json"] as const;
const ALLOWED_SANITIZERS = ["asan", "tsan", "ubsan"] as const;

export default tool({
  description: `Run sanitizer-enabled binaries (ASAN, TSAN, UBSAN) with parsing and validation.

EXAMPLES:
- AddressSanitizer: run_sanitizers({ buildDir: 'build', executable: './a.out', sanitizer: 'asan' })
- ThreadSanitizer with baseline: run_sanitizers({ buildDir: 'build', executable: './race', sanitizer: 'tsan', normalDuration: 1.2 })
- UBSan JSON output: run_sanitizers({ buildDir: 'build', executable: './ubsan_target', sanitizer: 'ubsan', outputMode: 'json' })

IMPORTANT:
- Sanitizers add 2–10x overhead; adjust timeouts accordingly.
- MSAN is not supported.
- Build artifacts must be produced with the selected sanitizer and live in buildDir.
- Suppressions/options are passed through to the sanitizer environment variables.`,
  args: {
    outputMode: tool.schema
      .enum(ALLOWED_OUTPUT_MODES)
      .optional()
      .describe("Output mode: 'summary' (default), 'full' (complete output), 'json' (structured)."),
    buildDir: tool.schema
      .string()
      .describe("Working directory containing the sanitizer-enabled binary (required). Example: 'build', 'example_cpp_dev/build'."),
    executable: tool.schema
      .string()
      .describe("Path to the sanitizer-enabled executable (required). Example: './a.out', './bin/target'."),
    sanitizer: tool.schema
      .enum(ALLOWED_SANITIZERS)
      .describe("Sanitizer to run (asan, tsan, ubsan)."),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 600). Must be positive."),
    suppressions: tool.schema
      .string()
      .optional()
      .describe("Path to suppressions file (tokens per line, # comments supported)."),
    options: tool.schema
      .string()
      .optional()
      .describe("Additional sanitizer options appended to the relevant *_OPTIONS environment variable."),
    normalDuration: tool.schema
      .number()
      .optional()
      .describe("Baseline duration in seconds for overhead ratio calculations. Must be positive when provided."),
    extraArgs: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Additional arguments passed to the executable after '--'."),
  },
  async execute(args) {
    const outputMode = (args.outputMode as (typeof ALLOWED_OUTPUT_MODES)[number] | undefined) || "summary";
    const buildDir = args.buildDir as string | undefined;
    const executable = args.executable as string | undefined;
    const sanitizer = args.sanitizer as (typeof ALLOWED_SANITIZERS)[number] | string | undefined;
    const timeout = args.timeout ?? 600;
    const suppressions = args.suppressions as string | undefined;
    const options = args.options as string | undefined;
    const normalDuration = args.normalDuration as number | undefined;
    const extraArgs = (args.extraArgs as string[] | undefined) || [];

    if (!buildDir) {
      return "ERROR: buildDir is required. Provide the working directory containing the sanitizer-enabled binary.";
    }

    if (!executable) {
      return "ERROR: executable is required. Provide the sanitizer-enabled binary path.";
    }

    if (!sanitizer) {
      return "ERROR: sanitizer is required. Choose one of: asan, tsan, ubsan.";
    }

    if (!ALLOWED_SANITIZERS.includes(sanitizer as any)) {
      return `ERROR: sanitizer must be one of asan, tsan, ubsan (received ${sanitizer}).`;
    }

    if (!ALLOWED_OUTPUT_MODES.includes(outputMode as any)) {
      return `ERROR: outputMode must be one of summary, full, json (received ${outputMode}).`;
    }

    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    if (normalDuration !== undefined && normalDuration <= 0) {
      return `ERROR: normalDuration must be positive when provided (received ${normalDuration}).`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_sanitizers.py`,
      `--build-dir=${buildDir}`,
      `--executable=${executable}`,
      `--sanitizer=${sanitizer}`,
      `--output-mode=${outputMode}`,
      `--timeout=${timeout}`,
    ];

    if (suppressions) {
      cmdParts.push(`--suppressions=${suppressions}`);
    }

    if (options) {
      cmdParts.push(`--options=${options}`);
    }

    if (normalDuration !== undefined) {
      cmdParts.push(`--normal-duration=${normalDuration}`);
    }

    if (extraArgs.length > 0) {
      cmdParts.push("--", ...extraArgs);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Sanitizer run completed but returned no output.";
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
        return `ERROR: Sanitizer run failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run sanitizer: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run sanitizer: ${message}`;
    }
  },
});
