/**
 * Pytest Runner Tool with Coverage
 *
 * Runs pytest with coverage and validation, returning either full output or summary.
 * This tool validates test results to prevent false positives.
 * Delegation note: advanced-key routing is implemented via hasExplicitAdvancedKey
 * in split-wrapper surfaces; keep this wrapper contract documentation in sync.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: `Run pytest with coverage and comprehensive validation. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

  Full suite:    { minTests: 1700 }
  Scoped tests:  { pytestArgs: ['adw/core/tests/'], minTests: 1 }
  By name:       { pytestArgs: ['-k', 'test_agent'], minTests: 1 }
  Skip slow:     { pytestArgs: ['-m', 'not slow and not performance'], minTests: 1 }
  Fail fast:     { failFast: true, pytestArgs: ['adw/core/tests/'], minTests: 1 }
  With coverage: { coverage: true, coverageThreshold: 80 }
  In worktree:   { cwd: '/path/to/worktree', pytestArgs: ['adw/'], minTests: 1 }

RULES:
- For scoped/targeted tests, always set minTests: 1 (default expects full suite).
- -v and --tb=short are always included. Do NOT pass these in pytestArgs.
- Omit optional fields entirely -- blank strings are treated as omitted.

See .opencode/tools/run_pytest.md for full parameter reference and advanced usage.`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode: 'summary' (default, human-readable summary only), 'full' (complete pytest output + summary), 'json' (structured data)"),
    minTests: tool.schema
      .number()
      .optional()
      .describe("Minimum expected test count for validation (default: 1). Set to 1 for scoped tests, ~1700 for full suite. Must be positive when provided."),
    pytestArgs: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Additional pytest arguments (do NOT include -v or --tb, they are already set). Examples: ['adw/core/tests/'], ['-k', 'test_agent'], ['-m', 'not slow']"),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 600 = 10 minutes). Must be positive when provided."),
    coverage: tool.schema
      .boolean()
      .optional()
      .describe("Enable coverage reporting (default: true). Set false to skip coverage for faster runs."),
    coverageSource: tool.schema
      .string()
      .optional()
      .describe(
        "Source module/path for coverage measurement. Comma-separated values are supported (e.g., 'adw.core,adw.utils'). If omitted, blank, or 'all', uses pyproject.toml [tool.coverage.run].source config. Examples: 'adw', 'src/my_package'",
      ),
    coverageThreshold: tool.schema
      .number()
      .optional()
      .describe("Minimum coverage percentage required (0-100). Fails validation if coverage is below threshold. Example: 80"),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory for pytest (default: project root). Blank strings are treated as omitted. Use for worktrees: '/path/to/trees/abc12345'"),
    failFast: tool.schema
      .boolean()
      .optional()
      .describe("Stop on first failure (-x flag). Useful for quick feedback during development. Default: false"),
    covReport: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Coverage report format(s) (default: ['term-missing']). Examples: ['term-missing'], ['html', 'xml'], ['term-missing', 'html:coverage_html']"),
    durations: tool.schema
      .number()
      .optional()
      .describe("Show N slowest test durations (0 for all). Maps to pytest --durations=N. Examples: 10 (slowest 10), 0 (all tests)"),
    durationsMin: tool.schema
      .number()
      .optional()
      .describe("Minimum duration in seconds for inclusion in slowest list (default: 0.005). Only applies when durations is set."),
    overrideIni: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Override ini options passed to pytest (--override-ini). Example: ['addopts=']"),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const minTests = args.minTests ?? 1;
    const pytestArgs = args.pytestArgs || [];
    const timeout = args.timeout ?? 600;
    const coverage = args.coverage !== false; // default true
    const coverageSource = typeof args.coverageSource === "string" ? args.coverageSource.trim() : undefined;
    const coverageThreshold = args.coverageThreshold;
    const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
    const failFast = args.failFast || false;
    const covReport = args.covReport || ["term-missing"];
    const durations = args.durations;
    const durationsMin = args.durationsMin;
    const overrideIni = args.overrideIni || [];

    if (minTests <= 0) {
      return "ERROR: minTests must be positive.";
    }
    if (timeout <= 0) {
      return "ERROR: timeout must be positive.";
    }

    // Build command
    const cmdParts = [
      "python3",
      `${import.meta.dir}/run_pytest.py`,
      `--output=${outputMode}`,
      `--min-tests=${minTests}`,
      `--timeout=${timeout}`,
    ];

     // Coverage options
     if (coverage) {
       cmdParts.push("--coverage");
       // Only pass --coverage-source if explicitly provided and not 'all'
       // Otherwise let pytest-cov use pyproject.toml [tool.coverage.run].source
       if (coverageSource && coverageSource !== "all") {
         const sources = coverageSource
           .split(",")
           .map((source) => source.trim())
           .filter((source) => source.length > 0);
         sources.forEach((source) => {
           cmdParts.push(`--coverage-source=${source}`);
         });
       }
       cmdParts.push(`--cov-report=${covReport.join(",")}`);
     } else {
       cmdParts.push("--no-coverage");
     }


    if (coverageThreshold !== undefined) {
      cmdParts.push(`--coverage-threshold=${coverageThreshold}`);
    }

    // Working directory
    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    // Fail fast
    if (failFast) {
      cmdParts.push("--fail-fast");
    }

    // Durations
    if (durations !== undefined) {
      cmdParts.push(`--durations=${durations}`);
      if (durationsMin !== undefined) {
        cmdParts.push(`--durations-min=${durationsMin}`);
      }
    }

    // Ini overrides
    if (overrideIni.length > 0) {
      overrideIni.forEach((entry) => {
        cmdParts.push(`--override-ini=${entry}`);
      });
    }

    // Add pytest arguments
    if (pytestArgs.length > 0) {
      cmdParts.push(...pytestArgs);
    }

    try {
      // Execute the Python script
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Pytest completed but returned no output.";
    } catch (error: any) {
      // Pytest or validation failed - return the output anyway
      // The Python script provides detailed error information
      const stdout = error.stdout?.toString?.() || "";
      const stderr = error.stderr?.toString?.() || "";
      const message = error.message || "Unknown error";

      // Prefer stdout if available (contains pytest output)
      if (stdout.trim()) {
        return stdout;
      }

      // Fall back to stderr if stdout is empty
      if (stderr.trim()) {
        return `ERROR: Pytest failed\n\n${stderr}`;
      }

      // Last resort: return the error message
      return `ERROR: Failed to run pytest: ${message}`;
    }
  },
});
