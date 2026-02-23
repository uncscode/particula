/**
 * MkDocs Build Tool
 *
 * Wraps the Python backing script build_mkdocs.py to run mkdocs build with
 * structured output options. Mirrors the run_bun_test.ts facade patterns.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT =
  "Encountered an ENOENT error. Ensure python3 is installed and on your PATH, mkdocs is " +
  "installed, and the backing script .opencode/tool/build_mkdocs.py exists.";

export default tool({
  description: `Run mkdocs build with output handling and validation.

EXAMPLES:
- Default build: build_mkdocs({})
- JSON output: build_mkdocs({ outputMode: 'json' })
- Strict + no clean: build_mkdocs({ strict: true, clean: false })
- Custom config: build_mkdocs({ configFile: 'docs/mkdocs.yml' })
- Validate only: build_mkdocs({ validateOnly: true })

IMPORTANT:
- Requires mkdocs to be installed on the host system.
- Uses python3 to run the backing script.
- Default timeout is 120 seconds.`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode: 'summary' (default), 'full', 'json'."),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 120). Must be positive."),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory override for mkdocs build."),
    strict: tool.schema
      .boolean()
      .optional()
      .describe("Enable mkdocs strict mode (maps to --strict)."),
    clean: tool.schema
      .boolean()
      .optional()
      .describe("Clean build directory before building (default: true)."),
    configFile: tool.schema
      .string()
      .optional()
      .describe("Path to mkdocs configuration file (default: mkdocs.yml)."),
    validateOnly: tool.schema
      .boolean()
      .optional()
      .describe("Build to a temporary directory and discard output."),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const timeout = args.timeout ?? 120;
    const cwd = args.cwd as string | undefined;
    const strict = args.strict === true;
    const clean = args.clean !== false;
    const configFile = args.configFile as string | undefined;
    const validateOnly = args.validateOnly === true;

    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/build_mkdocs.py`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
    ];

    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    if (strict) {
      cmdParts.push("--strict");
    }

    if (!clean) {
      cmdParts.push("--no-clean");
    }

    if (configFile) {
      cmdParts.push(`--config-file=${configFile}`);
    }

    if (validateOnly) {
      cmdParts.push("--validate-only");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "mkdocs build completed but returned no output.";
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
        return `ERROR: MkDocs build failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run mkdocs build: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run mkdocs build: ${message}`;
    }
  },
});
