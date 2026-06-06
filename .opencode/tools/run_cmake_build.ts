import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cmake.py (dependency #1352).";

export default tool({
  description:
    "Run CMake build-only operations. Requires build context and owns build flags/jobs/timeouts.",
  args: {
    outputMode: tool.schema.enum(["summary", "full", "json"]).optional(),
    preset: tool.schema.string().optional(),
    buildDir: tool.schema.string().optional(),
    jobs: tool.schema.number().optional(),
    buildTimeout: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const preset = typeof args.preset === "string" ? args.preset.trim() : undefined;
    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : undefined;
    const jobs = args.jobs ?? 0;
    const buildTimeout = args.buildTimeout ?? 1800;
    const buildTimeoutProvided = args.buildTimeout !== undefined;
    const timeout = args.timeout ?? 300;

    if (typeof args.preset === "string" && !preset) {
      return "ERROR: preset must not be blank when provided.";
    }
    if (typeof args.buildDir === "string" && !buildDir) {
      return "ERROR: buildDir must not be blank when provided.";
    }

    if (!preset && !buildDir) {
      return "ERROR: Build context required: provide either 'preset' or 'buildDir'.";
    }

    const isFiniteInteger = (value: number) => Number.isFinite(value) && Number.isInteger(value);

    if (!Number.isFinite(timeout)) {
      return `ERROR: timeout must be a finite number (received ${timeout}).`;
    }
    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    if (!isFiniteInteger(jobs)) {
      return `ERROR: jobs must be a finite integer (received ${jobs}).`;
    }
    if (jobs < 0) {
      return `ERROR: jobs must be non-negative (received ${jobs}).`;
    }

    if (!isFiniteInteger(buildTimeout)) {
      return `ERROR: buildTimeout must be a finite integer (received ${buildTimeout}).`;
    }
    if (buildTimeout <= 0) {
      return `ERROR: buildTimeout must be positive (received ${buildTimeout}).`;
    }

    const cmdParts = [
      "python3",
      `${import.meta.dir}/run_cmake.py`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
      "--build",
    ];

    if (preset) {
      cmdParts.push(`--preset=${preset}`);
    } else {
      cmdParts.push(`--build-dir=${buildDir}`);
    }

    if (jobs > 0) {
      cmdParts.push(`--jobs=${jobs}`);
    }

    if (buildTimeoutProvided || buildTimeout !== 1800) {
      cmdParts.push(`--build-timeout=${buildTimeout}`);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "CMake build completed but returned no output.";
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
        return `ERROR: CMake build failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run CMake build: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run CMake build: ${message}`;
    }
  },
});
