/**
 * Read-only Notebook Validation Tool
 *
 * Supports only non-mutating validation/check-sync flows.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/validate_notebook.py";

const FORBIDDEN_MUTATING_KEYS = ["convertToPy", "convertToIpynb", "sync", "outputDir"] as const;

const VALIDATION_FLAGS_WITH_CHECK_SYNC = [
  "outputMode",
  "skipSyntax",
  "validationMode",
  "fast",
  "full",
] as const;

export default tool({
  description: `Validate or check-sync Jupyter notebooks using read-only operations only.

EXAMPLES:
- Validation: validate_notebook_readonly({notebookPath: 'notebook.ipynb'})
- Recursive validation: validate_notebook_readonly({notebookPath: 'docs/Examples/', recursive: true})
- Check-sync (CI): validate_notebook_readonly({notebookPath: 'docs/Examples/', recursive: true, checkSync: true})

NOT SUPPORTED:
- convertToPy / convertToIpynb / sync / outputDir
- Use convert_notebook_to_py, convert_py_to_notebook, or sync_notebook_pair for mutating operations.`,
  args: {
    notebookPath: tool.schema
      .string()
      .describe("Path to notebook file (.ipynb/.py) or directory"),
    recursive: tool.schema
      .boolean()
      .optional()
      .describe("Search recursively when notebookPath is a directory"),
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode (validation only): summary (default), full, or json"),
    skipSyntax: tool.schema
      .boolean()
      .optional()
      .describe("Skip Python syntax checking (validation only)"),
    validationMode: tool.schema
      .enum(["fast", "full"])
      .optional()
      .describe("Validation profile (validation only): fast/full"),
    fast: tool.schema.boolean().optional().describe("Alias for validationMode=fast"),
    full: tool.schema.boolean().optional().describe("Alias for validationMode=full"),
    checkSync: tool.schema
      .boolean()
      .optional()
      .describe("Check notebook/script sync state (read-only; exit 1 when out of sync)"),
  },
  async execute(args) {
    const hasForbiddenMutatingKey = FORBIDDEN_MUTATING_KEYS.some((key) => Object.hasOwn(args, key));
    if (hasForbiddenMutatingKey) {
      return "ERROR: validate_notebook_readonly does not support mutating options (convertToPy, convertToIpynb, sync, outputDir). Use convert_notebook_to_py, convert_py_to_notebook, or sync_notebook_pair for mutating operations.";
    }

    const notebookPathRaw = args.notebookPath as string;
    const notebookPath = notebookPathRaw?.trim?.() ?? "";
    if (!notebookPath) {
      return "ERROR: 'notebookPath' is required and must be a non-empty string.";
    }

    const recursive = Boolean(args.recursive);
    const outputMode = (args.outputMode as string | undefined) || "summary";
    const skipSyntax = Boolean(args.skipSyntax);
    const validationMode = args.validationMode as string | undefined;
    const fast = Boolean(args.fast);
    const full = Boolean(args.full);
    const checkSync = Boolean(args.checkSync);

    if (validationMode && (fast || full)) {
      return "ERROR: Conflicting validation options: use either 'validationMode' or 'fast/full' aliases, not both.";
    }
    if (fast && full) {
      return "ERROR: Conflicting validation options: 'fast' and 'full' cannot both be true.";
    }
    if (checkSync) {
      const hasValidationFlag = VALIDATION_FLAGS_WITH_CHECK_SYNC.some((key) => Object.hasOwn(args, key));
      if (hasValidationFlag) {
        return "ERROR: Conflicting options: 'checkSync' cannot be combined with validation/output flags (outputMode, skipSyntax, validationMode, fast, full).";
      }
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/validate_notebook.py`,
      notebookPath,
    ];

    if (checkSync) {
      cmdParts.push("--check-sync");
    } else {
      cmdParts.push(`--output=${outputMode}`);
      if (skipSyntax) {
        cmdParts.push("--skip-syntax");
      }
      if (validationMode) {
        cmdParts.push("--validation-mode", validationMode);
      }
      if (fast) {
        cmdParts.push("--fast");
      }
      if (full) {
        cmdParts.push("--full");
      }
    }

    if (recursive) {
      cmdParts.push("--recursive");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Notebook tool completed but returned no output.";
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
        return `ERROR: Notebook tool failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run notebook tool: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run notebook tool: ${message}`;
    }
  },
});
