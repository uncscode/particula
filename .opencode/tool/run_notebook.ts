/**
 * Notebook Execution Tool
 *
 * Executes Jupyter notebooks and validates outputs. Mirrors the run_pytest/run_linters/run_ctest
 * patterns with summary/full/json modes and structured results.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_notebook.py (dependency #1466).";

export default tool({
  description: `Execute Jupyter notebooks with validation and structured outputs.

EXAMPLES:
- Single notebook: run_notebook({notebookPath: 'docs/Examples/setup-template-init-tutorial.ipynb'})
- All notebooks in dir: run_notebook({notebookPath: 'docs/Examples/', recursive: true})
- With timeout: run_notebook({notebookPath: 'notebook.ipynb', timeout: 300})
- JSON output: run_notebook({notebookPath: 'notebook.ipynb', outputMode: 'json'})
- Validate output: run_notebook({notebookPath: 'notebook.ipynb', expectOutput: ['DataFrame', 'plot']})

IMPORTANT: Requires nbconvert and nbclient packages (dev dependencies).
NOTE: Default timeout is 600 seconds per notebook.`,
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
      .describe("Directory where executed notebook copies should be written")
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const notebookPath = args.notebookPath as string;
    const recursive = args.recursive || false;
    const timeout = args.timeout || 600;
    const expectOutput = args.expectOutput || [];
    const cwd = args.cwd as string | undefined;
    const writeExecuted = args.writeExecuted as string | undefined;

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

    if (expectOutput.length > 0) {
      cmdParts.push("--expect-output", ...expectOutput);
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
