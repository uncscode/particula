import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/validate_notebook.py";

export default tool({
  description: `Sync notebook/script pairs using Jupytext (newer file wins).

EXAMPLES:
- Single pair: sync_notebook_pair({ notebookPath: 'docs/Examples/demo.ipynb' })
- Script input: sync_notebook_pair({ notebookPath: 'docs/Examples/demo.py' })
- Recursive sync: sync_notebook_pair({ notebookPath: 'docs/Examples/', recursive: true })`,
  args: {
    notebookPath: tool.schema
      .string()
      .describe("Path to notebook/script file (.ipynb/.py) or directory"),
    recursive: tool.schema
      .boolean()
      .optional()
      .describe("Search recursively when notebookPath is a directory"),
    outputDir: tool.schema
      .string()
      .optional()
      .describe("Not supported for sync; conversion-only option"),
  },
  async execute(args) {
    const notebookPath = typeof args.notebookPath === "string" ? args.notebookPath.trim() : "";
    if (!notebookPath) {
      return "ERROR: notebookPath is required and must be non-empty.";
    }
    const recursive = Boolean(args.recursive);

    if (Object.hasOwn(args, "outputDir")) {
      return "ERROR: sync_notebook_pair does not support outputDir. Use conversion tools for output directory targeting.";
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/validate_notebook.py`,
      notebookPath,
      "--sync",
    ];

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
