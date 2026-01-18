/**
 * Linter Runner Tool
 *
 * Runs configured linters (ruff, mypy) for the Agent repository.
 * Automatically fixes issues where possible and reports remaining problems.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Run all configured linters (ruff, mypy) with auto-fixing. Follows .github/workflows/lint.yml workflow. Returns comprehensive linting results with pass/fail status.",
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode: 'summary' (default, human-readable), 'full' (complete linter output), 'json' (structured data)"),
    autoFix: tool.schema
      .boolean()
      .optional()
      .describe("Automatically fix issues where possible (default: true). Runs ruff check --fix + ruff format + ruff check"),
    linters: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Specific linters to run: ['ruff', 'mypy']. Default: ['ruff', 'mypy'] matching CI workflow"),
    targetDir: tool.schema
      .string()
      .optional()
      .describe("Target directory to lint. If omitted, uses pyproject.toml config (lints from project root)."),
    ruffTimeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds for ruff commands (default: 120 = 2 minutes)"),
    mypyTimeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds for mypy command (default: 180 = 3 minutes)"),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const autoFix = args.autoFix !== false; // Default to true
    const linters = args.linters || ["ruff", "mypy"]; // Match CI workflow
    const targetDir = args.targetDir;
    const ruffTimeout = args.ruffTimeout || 120;
    const mypyTimeout = args.mypyTimeout || 180;

    // Build command
    const cmdParts = [
      "python3",
      `${import.meta.dir}/run_linters.py`,
      `--output=${outputMode}`,
      `--ruff-timeout=${ruffTimeout}`,
      `--mypy-timeout=${mypyTimeout}`,
    ];

    // Only pass --target-dir if explicitly provided
    // Otherwise let ruff/mypy use pyproject.toml config from project root
    if (targetDir) {
      cmdParts.push(`--target-dir=${targetDir}`);
    }

    if (autoFix) {
      cmdParts.push("--auto-fix");
    }

    if (linters.length > 0) {
      cmdParts.push(`--linters=${linters.join(",")}`);
    }

    try {
      // Execute the Python script
      const result = await Bun.$`${cmdParts}`.text();
      return result;
    } catch (error: any) {
      // Linter failed - return the output anyway
      // The Python script provides detailed error information
      if (error.stdout) {
        return error.stdout.toString();
      }
      return `ERROR: Failed to run linters: ${error.message}`;
    }
  },
});
