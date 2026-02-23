/**
 * Notebook Execution Tool
 *
 * Executes Jupyter notebooks and validates outputs. Mirrors the run_pytest/run_linters/run_ctest
 * patterns with summary/full/json modes and structured results.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_notebook.py (dependency #1466).";

export default tool({
  description: `Execute Jupyter notebooks and Python scripts with validation and structured outputs.

BREAKING CHANGE: Default behavior overwrites the source notebook and creates a .ipynb.bak backup. Use noOverwrite to keep the source unchanged or noBackup to skip backups. writeExecuted can still be combined with default overwrite.

Validation runs before execution to fail fast on corrupted notebooks; use skipValidation when debugging known-invalid notebooks.

EXAMPLES:
- Single notebook: run_notebook({notebookPath: 'docs/Examples/setup-template-init-tutorial.ipynb'})
- All notebooks in dir: run_notebook({notebookPath: 'docs/Examples/', recursive: true})
- With timeout: run_notebook({notebookPath: 'notebook.ipynb', timeout: 300})
- JSON output: run_notebook({notebookPath: 'notebook.ipynb', outputMode: 'json'})
- Validate output: run_notebook({notebookPath: 'notebook.ipynb', expectOutput: ['DataFrame', 'plot']})
- Single script (auto-detected): run_notebook({notebookPath: 'examples/demo.py'})
- Scripts in directory: run_notebook({notebookPath: 'examples/', script: true, recursive: true})
- Script with output validation: run_notebook({notebookPath: 'script.py', expectOutput: ['result']})

py:percent SCRIPTS:
Pass a py:percent (.py) file directly to execute it and get stdout/stderr/errors back.
The # %% cell markers are plain comments to Python, so the script runs top-to-bottom.
- Run py:percent script: run_notebook({notebookPath: 'docs/Examples/panel-methods/regime-selection.py', outputMode: 'full', timeout: 300})
- Run all py:percent scripts in dir: run_notebook({notebookPath: 'docs/Examples/panel-methods/', script: true, recursive: true})
Use outputMode 'full' to see stdout/stderr per script, or 'json' for structured results.

IMPORTANT: Requires nbconvert and nbclient packages (dev dependencies) for notebook execution.
NOTE: Default timeout is 600 seconds per notebook/script. Scripts run via sys.executable (same Python/venv as the tool).`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe(
        "Output mode: 'summary' (default, concise), 'full' (per-notebook details), 'json' (structured payload)",
      ),
    notebookPath: tool.schema
      .string()
      .describe("Path to notebook file (.ipynb) or directory containing notebooks"),
    recursive: tool.schema
      .boolean()
      .optional()
      .describe("If notebookPath is a directory, search recursively (default: false)"),
    script: tool.schema
      .boolean()
      .optional()
      .describe(
        "When notebookPath is a directory, collect .py scripts instead of .ipynb notebooks. Single .py files are auto-detected without this flag.",
      ),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds per notebook (default: 600)"),
    expectOutput: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("List of strings that should appear in executed notebook outputs"),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory for execution (default: project root)"),
    writeExecuted: tool.schema
      .string()
      .optional()
      .describe(
        "Directory where executed notebook copies should be written; when overwriting (default) source is also updated",
      ),
    noOverwrite: tool.schema
      .boolean()
      .optional()
      .describe(
        "Do not overwrite the source notebook; execute into writeExecuted or validation temp directory and implicitly skip backups (equivalent to enabling noBackup)",
      ),
    noBackup: tool.schema
      .boolean()
      .optional()
      .describe("Skip creation of the .ipynb.bak backup when overwriting the source notebook"),
    skipValidation: tool.schema
      .boolean()
      .optional()
      .describe("Skip pre-execution validation when running known-invalid notebooks"),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const notebookPath = args.notebookPath as string;
    const recursive = args.recursive || false;
    const timeout = args.timeout || 600;
    const expectOutput = args.expectOutput || [];
    const cwd = args.cwd as string | undefined;
    const writeExecuted = args.writeExecuted as string | undefined;
    const noOverwrite = args.noOverwrite || false;
    const noBackup = args.noBackup || false;
    const skipValidation = args.skipValidation || false;
    const script = args.script || false;

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_notebook.py`,
      notebookPath,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
    ];

    if (writeExecuted) {
      cmdParts.push(`--write-executed=${writeExecuted}`);
    }

    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    if (recursive) {
      cmdParts.push("--recursive");
    }

    if (script) {
      cmdParts.push("--script");
    }

    if (expectOutput.length > 0) {
      cmdParts.push("--expect-output", ...expectOutput);
    }

    if (noOverwrite) {
      cmdParts.push("--no-overwrite");
    }

    if (noBackup) {
      cmdParts.push("--no-backup");
    }

    if (skipValidation) {
      cmdParts.push("--skip-validation");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Notebook execution completed but returned no output.";
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
        return `ERROR: Notebook execution failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run notebook tool: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run notebook tool: ${message}`;
    }
  },
});
