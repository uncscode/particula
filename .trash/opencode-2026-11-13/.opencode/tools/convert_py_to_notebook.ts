import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/validate_notebook.py";

export default tool({
  description: `Convert py:percent scripts to Jupyter notebooks.

EXAMPLES:
- Single script: convert_py_to_notebook({ notebookPath: 'docs/Examples/demo.py' })
- Recursive conversion: convert_py_to_notebook({ notebookPath: 'docs/Examples/', recursive: true })
- Custom output dir: convert_py_to_notebook({ notebookPath: 'docs/Examples/', recursive: true, outputDir: 'notebooks' })`,
  args: {
    notebookPath: tool.schema.string().describe("Path to py:percent file (.py) or directory"),
    recursive: tool.schema
      .boolean()
      .optional()
      .describe("Search recursively when notebookPath is a directory"),
    outputDir: tool.schema
      .string()
      .optional()
      .describe("Output directory for converted notebooks"),
  },
  async execute(args) {
    const notebookPath = typeof args.notebookPath === "string" ? args.notebookPath.trim() : "";
    if (!notebookPath) {
      return "ERROR: notebookPath is required and must be non-empty.";
    }
    const recursive = Boolean(args.recursive);
    const outputDirRaw = args.outputDir as string | undefined;
    const outputDir = outputDirRaw?.trim?.() ?? "";

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/validate_notebook.py`,
      notebookPath,
      "--convert-to-ipynb",
    ];

    if (outputDir) {
      cmdParts.push("--output-dir", outputDir);
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
