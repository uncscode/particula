/**
 * Notebook Validation Tool
 *
 * Validates Jupyter notebook structure and syntax without execution.
 * Mirrors run_notebook patterns with summary/full/json modes.
 *
 * Exit codes: 0=all valid, 1=invalid notebooks, 2=tool error.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/validate_notebook.py";

export default tool({
  description: `Validate, convert, or sync Jupyter notebooks using Jupytext.

EXAMPLES:
- Validation: validate_notebook({notebookPath: 'notebook.ipynb'})
- Convert: validate_notebook({notebookPath: 'notebook.ipynb', convertToPy: true})
- Convert with output dir: validate_notebook({notebookPath: 'docs/Examples', recursive: true, convertToPy: true, outputDir: 'scripts'})
- Convert to notebook: validate_notebook({notebookPath: 'script.py', convertToIpynb: true})
- Sync: validate_notebook({notebookPath: 'notebook.ipynb', sync: true})
- Check-sync (CI): validate_notebook({notebookPath: 'docs/Examples', recursive: true, checkSync: true})

Exit codes: 0=success, 1=functional failure (invalid/out-of-sync/convert failure), 2=tool error`,
  args: {
    notebookPath: tool.schema
      .string()
      .describe("Path to notebook file (.ipynb) or directory"),
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
      .describe("Skip Python syntax checking (validation only); syntax issues become warnings"),
    convertToPy: tool.schema
      .boolean()
      .optional()
      .describe("Convert notebooks to .py:percent format"),
    convertToIpynb: tool.schema
      .boolean()
      .optional()
      .describe("Convert py:percent scripts to .ipynb notebooks"),
    sync: tool.schema
      .boolean()
      .optional()
      .describe("Bidirectional sync between notebook and script (newer wins)"),
    checkSync: tool.schema
      .boolean()
      .optional()
      .describe("Check notebook/script sync state (read-only; exit 1 when out of sync)"),
    outputDir: tool.schema
      .string()
      .optional()
      .describe("Output directory for converted files (only with convertToPy or convertToIpynb)"),
  },
  async execute(args) {
    const notebookPath = args.notebookPath as string;
    const recursive = args.recursive || false;
    const outputMode = args.outputMode || "summary";
    const skipSyntax = args.skipSyntax || false;
    const convertToPy = args.convertToPy || false;
    const convertToIpynb = args.convertToIpynb || false;
    const sync = args.sync || false;
    const checkSync = args.checkSync || false;
    const outputDir = args.outputDir;

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/validate_notebook.py`,
      notebookPath,
    ];

    if (convertToPy) {
      cmdParts.push("--convert-to-py");
    }
    if (convertToIpynb) {
      cmdParts.push("--convert-to-ipynb");
    }
    if (sync) {
      cmdParts.push("--sync");
    }
    if (checkSync) {
      cmdParts.push("--check-sync");
    }
    if (outputDir) {
      cmdParts.push("--output-dir", outputDir);
    }

    const usesValidationOnlyFlags = !convertToPy && !convertToIpynb && !sync && !checkSync;

    if (usesValidationOnlyFlags) {
      cmdParts.push(`--output=${outputMode}`);
      if (skipSyntax) {
        cmdParts.push("--skip-syntax");
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
