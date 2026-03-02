/**
 * CMake Configuration Tool
 *
 * Wraps the Python backing script run_cmake.py to configure CMake projects
 * with preset and Ninja support. Mirrors the run_pytest.ts pattern for
 * consistency across OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tool/run_cmake.py (dependency #1352).";

export default tool({
  description: `Run CMake configuration with preset and Ninja support. Executes the Python backing script to keep behavior consistent with run_cmake.py.

EXAMPLES:
- Configure with preset: run_cmake({ preset: "ninja-release" })
- Configure with Ninja: run_cmake({ ninja: true, sourceDir: "example_cpp_dev" })
- Custom build dir: run_cmake({ sourceDir: ".", buildDir: "build/debug" })
- With timeout: run_cmake({ preset: "default", timeout: 600 })
- Configure and build: run_cmake({ preset: "debug", build: true })
- Build with parallel jobs: run_cmake({ preset: "debug", build: true, jobs: 8 })
- Custom build timeout: run_cmake({ preset: "release", build: true, buildTimeout: 3600 })
- Full output: run_cmake({ preset: "debug", outputMode: "full" })
- JSON output: run_cmake({ preset: "debug", outputMode: "json" })
- Custom args: run_cmake({ cmakeArgs: ["-DCMAKE_BUILD_TYPE=Release"] })

IMPORTANT: Preset mode requires CMakePresets.json in the source directory.
NOTE: Ninja is recommended for faster builds and is ignored when preset is provided.
Dependency: Backing script .opencode/tool/run_cmake.py from #1352 must be present; errors will mention it if missing.`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe("Output mode: 'summary' (default, concise), 'full' (complete output), 'json' (structured)."),
    preset: tool.schema
      .string()
      .optional()
      .describe("CMake preset name from CMakePresets.json (e.g., 'default', 'debug', 'ninja-release', 'asan')."),
    sourceDir: tool.schema
      .string()
      .optional()
      .describe("CMake source directory (default: current directory). Ignored when preset is provided."),
    buildDir: tool.schema
      .string()
      .optional()
      .describe("CMake build directory (default: 'build'). Ignored when preset is provided."),
    ninja: tool.schema
      .boolean()
      .optional()
      .describe("Use Ninja generator (default: false). Ignored when preset is provided."),
    build: tool.schema
      .boolean()
      .optional()
      .describe("Run cmake --build after configuration (default: false)."),
    jobs: tool.schema
      .number()
      .optional()
      .describe("Parallel build jobs (default: 0 = auto). Only used when build is true."),
    buildTimeout: tool.schema
      .number()
      .optional()
      .describe("Build timeout in seconds (default: 1800 = 30 minutes). Only used when build is true."),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 300 = 5 minutes). Must be positive."),
    cmakeArgs: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe("Additional CMake arguments. Examples: ['-DCMAKE_BUILD_TYPE=Release'], ['-DENABLE_TESTS=ON']"),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const preset = args.preset;
    const sourceDir = args.sourceDir || ".";
    const buildDir = args.buildDir || "build";
    const ninja = args.ninja || false;
    const timeout = args.timeout ?? 300;
    const build = args.build || false;
    const jobs = args.jobs ?? 0;
    const buildTimeout = args.buildTimeout ?? 1800;
    const buildTimeoutProvided = args.buildTimeout !== undefined;
    const cmakeArgs = args.cmakeArgs || [];

    if (timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    const isFiniteInteger = (value: number) => Number.isFinite(value) && Number.isInteger(value);

    if (build) {
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
      cmdParts.push(`--source-dir=${sourceDir}`);
      cmdParts.push(`--build-dir=${buildDir}`);
      if (ninja) {
        cmdParts.push("--ninja");
      }
    }

    if (cmakeArgs.length > 0) {
      cmdParts.push("--", ...cmakeArgs);
    }

    if (build) {
      cmdParts.push("--build");
    }

    if (build && jobs > 0) {
      cmdParts.push(`--jobs=${jobs}`);
    }

    if (build && (buildTimeoutProvided || buildTimeout !== 1800)) {
      cmdParts.push(`--build-timeout=${buildTimeout}`);
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
