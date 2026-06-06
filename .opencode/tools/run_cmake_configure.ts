import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cmake.py (dependency #1352).";

export default tool({
  description:
    "Run CMake configure-only operations. Accepts preset/manual configure args and never emits build flags.",
  args: {
    outputMode: tool.schema.enum(["summary", "full", "json"]).optional(),
    preset: tool.schema.string().optional(),
    sourceDir: tool.schema.string().optional(),
    buildDir: tool.schema.string().optional(),
    ninja: tool.schema.boolean().optional(),
    timeout: tool.schema.number().optional(),
    cmakeArgs: tool.schema.array(tool.schema.string()).optional(),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const preset = typeof args.preset === "string" ? args.preset.trim() : undefined;
    const sourceDir = typeof args.sourceDir === "string" ? args.sourceDir.trim() : args.sourceDir;
    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : args.buildDir;
    const ninja = args.ninja || false;
    const timeout = args.timeout ?? 300;
    const cmakeArgs = (args.cmakeArgs || []).map((arg) => String(arg).trim());

    if (typeof args.preset === "string" && !preset) {
      return "ERROR: preset must not be blank when provided.";
    }
    if (typeof args.sourceDir === "string" && !sourceDir) {
      return "ERROR: sourceDir must not be blank when provided.";
    }
    if (typeof args.buildDir === "string" && !buildDir) {
      return "ERROR: buildDir must not be blank when provided.";
    }
    if (cmakeArgs.some((arg) => !arg)) {
      return "ERROR: cmakeArgs must not contain blank values.";
    }

    if (!Number.isFinite(timeout)) {
      return `ERROR: timeout must be a finite number (received ${timeout}).`;
    }
    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    const cmdParts = [
      "python3",
      `${import.meta.dir}/run_cmake.py`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
    ];

    if (preset) {
      cmdParts.push(`--preset=${preset}`);
    } else {
      cmdParts.push(`--source-dir=${sourceDir || "."}`);
      cmdParts.push(`--build-dir=${buildDir || "build"}`);
      if (ninja) {
        cmdParts.push("--ninja");
      }
    }

    if (cmakeArgs.length > 0) {
      cmdParts.push("--", ...cmakeArgs);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "CMake configuration completed but returned no output.";
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
        return `ERROR: CMake configuration failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run CMake: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run CMake: ${message}`;
    }
  },
});
