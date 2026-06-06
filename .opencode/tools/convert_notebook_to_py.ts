import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/validate_notebook.py";

export default tool({
  description: `Convert Jupyter notebooks to py:percent scripts.

EXAMPLES:
- Single notebook: convert_notebook_to_py({ notebookPath: 'docs/Examples/demo.ipynb' })
- Recursive conversion: convert_notebook_to_py({ notebookPath: 'docs/Examples/', recursive: true })
- Custom output dir: convert_notebook_to_py({ notebookPath: 'docs/Examples/', recursive: true, outputDir: 'scripts' })`,
  args: {
    notebookPath: tool.schema.string().describe("Path to notebook file (.ipynb) or directory"),
    recursive: tool.schema
      .boolean()
      .optional()
      .describe("Search recursively when notebookPath is a directory"),
    outputDir: tool.schema
      .string()
      .optional()
      .describe("Output directory for converted scripts"),
  },
  async execute(args) {
    const notebookPath = args.notebookPath as string;
    if (!notebookPath || !notebookPath.trim()) {
      return "ERROR: notebookPath is required and must be non-empty.";
    }
    const recursive = Boolean(args.recursive);
    const outputDirRaw = args.outputDir as string | undefined;
    const outputDir = outputDirRaw?.trim?.() ?? "";

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/validate_notebook.py`,
      notebookPath,
      "--convert-to-py",
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
