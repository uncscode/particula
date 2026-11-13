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
  description: `Preview build directory cleanup (read-only dry-run mode).

EXAMPLES:
- Preview default build dir: clear_build_preview({})
- Preview explicit dir: clear_build_preview({ buildDir: "build/debug" })

SAFETY:
- Always executes with --dry-run
- Never accepts destructive force options`,
  args: {
    buildDir: tool.schema
      .string()
      .optional()
      .describe("Build directory to preview (default: 'build'). Blank values normalize to 'build'."),
  },
  async execute(args) {
    const buildDir = normalizeBuildDir(args.buildDir);
    const cmdParts = ["python3", `${import.meta.dir}/clear_build.py`, `--build-dir=${buildDir}`, "--dry-run"];

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Clear build preview completed but returned no output.";
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
        return `ERROR: Clear build preview failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run clear_build preview: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run clear_build preview: ${message}`;
    }
  },
});
