import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/clear_build.py (dependency #1354).";

export default tool({
  description: `Clear build directories with validation, dry-run previews, and force gating.

⚠️ WARNING: This tool permanently deletes files. Always run with dryRun: true first to verify.

EXAMPLES:
- Preview deletion: clear_build({ buildDir: "build", dryRun: true })
- Delete build dir: clear_build({ buildDir: "build", force: true })
- Clear specific variant: clear_build({ buildDir: "build/debug", force: true })
- Safer workflow:
    1) clear_build({ buildDir: "build", dryRun: true })
    2) clear_build({ buildDir: "build", force: true })

SAFETY FEATURES:
- Path validation prevents deletion outside the project root
- Dry-run mode shows what would be deleted without removing files
- Force flag required for actual deletion (default: false)
- Reports file count and size before deletion`,
  args: {
    buildDir: tool.schema
      .string()
      .optional()
      .describe("Build directory to clear (default: 'build'). Examples: 'build', 'build/debug'."),
    dryRun: tool.schema
      .boolean()
      .optional()
      .describe(
        "Preview deletion without removing files (default: false). Recommended: set dryRun: true first."
      ),
    force: tool.schema
      .boolean()
      .optional()
      .describe("Perform deletion (default: false). Required for actual removal after preview."),
  },
  async execute(args) {
    const buildDir = args.buildDir ?? "build";
    const dryRun = args.dryRun ?? false;
    const force = args.force ?? false;

    const cmdParts = [
      "python3",
      `${import.meta.dir}/clear_build.py`,
      `--build-dir=${buildDir}`,
    ];

    if (dryRun) {
      cmdParts.push("--dry-run");
    }

    if (force) {
      cmdParts.push("--force");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Clear build completed but returned no output.";
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
        return `ERROR: Clear build failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run clear_build: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run clear_build: ${message}`;
    }
  },
});
