import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/clear_build.py (dependency #1354).";

function normalizeBuildDir(input: unknown): string {
  if (typeof input !== "string") {
    return "build";
  }

  const trimmed = input.trim();
  return trimmed.length > 0 ? trimmed : "build";
}

export default tool({
  description: `Delete build directories with explicit authorization.

EXAMPLES:
- Delete default build dir: clear_build_delete({ force: true })
- Delete explicit dir: clear_build_delete({ buildDir: "build/debug", force: true })

SAFETY:
- Fails closed unless force === true
- Never performs implicit dry-run`,
  args: {
    buildDir: tool.schema
      .string()
      .optional()
      .describe("Build directory to delete (default: 'build'). Blank values normalize to 'build'."),
    force: tool.schema
      .boolean()
      .describe("Required authorization gate. Must be true to perform deletion."),
  },
  async execute(args) {
    if (args.force !== true) {
      return "ERROR: clear_build_delete requires force: true to perform destructive deletion.";
    }

    const buildDir = normalizeBuildDir(args.buildDir);
    const cmdParts = ["python3", `${import.meta.dir}/clear_build.py`, `--build-dir=${buildDir}`, "--force"];

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Clear build delete completed but returned no output.";
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
        return `ERROR: Clear build delete failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run clear_build delete: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run clear_build delete: ${message}`;
    }
  },
});
