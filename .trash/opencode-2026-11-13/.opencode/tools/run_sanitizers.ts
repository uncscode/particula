/**
 * Sanitizer Runner Tool
 *
 * Wraps the Python backing script run_sanitizers.py to execute ASAN, TSAN, or
 * UBSAN enabled binaries with structured outputs. Mirrors the patterns used by
 * run_ctest.ts and run_cpp_linters.ts for consistency across OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

import { ALLOWED_OUTPUT_MODES, ALLOWED_SANITIZERS, hasAdvancedSanitizerKey } from "./run_sanitizers_shared";
import runSanitizersAdvanced from "./run_sanitizers_advanced";
import runSanitizersBasic from "./run_sanitizers_basic";

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
    if (hasAdvancedSanitizerKey(args as Record<string, unknown>)) {
      return runSanitizersAdvanced.execute!(args);
    }

    return runSanitizersBasic.execute!(args);
  },
});
