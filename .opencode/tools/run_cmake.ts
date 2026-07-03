/**
 * CMake Configuration Tool
 *
 * Wraps the Python backing script run_cmake.py to configure CMake projects
 * with preset and Ninja support for OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync } from "node:fs";
import path from "node:path";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cmake.py (dependency #1352).";

type OutputMode = "summary" | "full" | "json";

type ParsedRunCmakeOptions = {
  outputMode?: OutputMode;
  ninja?: true;
  jobs?: number;
};

const RUN_CMAKE_OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

function tokenizeOptions(options: string): { ok: true; tokens: string[] } | { ok: false; error: string } {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;

  for (let index = 0; index < options.length; index += 1) {
    const char = options[index];
    if (quote) {
      current += char;
      if (char === quote) {
        quote = undefined;
      }
      continue;
    }

    if (char === "'" || char === '"') {
      quote = char;
      current += char;
      continue;
    }

    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }

    current += char;
  }

  if (quote) {
    return { ok: false, error: "ERROR: Invalid options string: unterminated quoted value." };
  }
  if (current) {
    tokens.push(current);
  }

  return { ok: true, tokens };
}

function stripOptionalQuotes(value: string): string {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === '"' || first === "'") && last === first) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function parseRunCmakeOptions(rawOptions: unknown):
  | { ok: true; options: ParsedRunCmakeOptions }
  | { ok: false; error: string } {
  if (rawOptions === undefined || rawOptions === null) {
    return { ok: true, options: {} };
  }
  if (typeof rawOptions !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalized = rawOptions.trim();
  if (!normalized) {
    return { ok: true, options: {} };
  }

  const tokenized = tokenizeOptions(normalized);
  if (!tokenized.ok) {
    return tokenized;
  }

  const parsed: ParsedRunCmakeOptions = {};
  for (const token of tokenized.tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }

    if (separatorIndex === -1) {
      if (token === "ninja") {
        if (parsed.ninja) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
        }
        parsed.ninja = true;
        continue;
      }

      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    const name = token.slice(0, separatorIndex);
    const rawValue = token.slice(separatorIndex + 1);
    if (!rawValue) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    const value = stripOptionalQuotes(rawValue).trim();
    if (!value) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    if (name === "output") {
      if (!RUN_CMAKE_OUTPUT_MODES.has(value as OutputMode)) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': output must be one of summary, full, json.`,
        };
      }
      if (parsed.outputMode !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.outputMode = value as OutputMode;
      continue;
    }

    if (name === "jobs") {
      if (!/^\d+$/.test(value)) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': jobs must be a non-negative integer.`,
        };
      }
      if (parsed.jobs !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.jobs = Number(value);
      continue;
    }

    if (name === "ninja") {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token does not accept a value.`,
      };
    }

    return {
      ok: false,
      error: `ERROR: Invalid options token '${token}': token is not supported.`,
    };
  }

  return { ok: true, options: parsed };
}

function resolvePathForRepoCheck(value: string): string {
  const repoRoot = realpathSync(process.cwd());
  const candidate = path.isAbsolute(value) ? path.normalize(value) : path.resolve(repoRoot, value);
  let probe = candidate;
  const tailSegments: string[] = [];

  while (!existsSync(probe)) {
    const parent = path.dirname(probe);
    if (parent === probe) {
      throw new Error(`path does not exist: ${value}`);
    }
    tailSegments.unshift(path.basename(probe));
    probe = parent;
  }

  const resolvedProbe = realpathSync(probe);
  return tailSegments.length > 0 ? path.resolve(resolvedProbe, ...tailSegments) : resolvedProbe;
}

function validatePathWithinRepoRoot(
  value: string,
  parameterName: "sourceDir" | "buildDir",
): string | undefined {
  try {
    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = resolvePathForRepoCheck(value);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: ${parameterName} path resolves outside repository root: ${value} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid ${parameterName} path: ${value} (${message})`;
  }

  return undefined;
}

export default tool({
  description: `Run CMake configuration with preset and Ninja support. Executes the Python backing script to keep behavior consistent with run_cmake.py.

EXAMPLES:
- Configure with preset: run_cmake({ preset: "ninja-release" })
- Configure with Ninja: run_cmake({ sourceDir: "example_cpp_dev", options: "ninja" })
- Custom build dir: run_cmake({ sourceDir: ".", buildDir: "build/debug" })
- With timeout: run_cmake({ preset: "default", timeout: 600 })
- Configure and build: run_cmake({ preset: "debug", build: true })
- Build with parallel jobs: run_cmake({ preset: "debug", build: true, options: "jobs=8" })
- Custom build timeout: run_cmake({ preset: "release", build: true, buildTimeout: 3600 })
- Full output: run_cmake({ preset: "debug", options: "output=full" })
- JSON output: run_cmake({ preset: "debug", options: "output=json" })
- Custom args: run_cmake({ cmakeArgs: ["-DCMAKE_BUILD_TYPE=Release"] })

IMPORTANT: Preset mode requires CMakePresets.json in the source directory.
NOTE: Ninja is recommended for faster builds and is ignored when preset is provided.
Dependency: Backing script .opencode/tools/run_cmake.py from #1352 must be present; errors will mention it if missing.`,
  args: {
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
    build: tool.schema
      .boolean()
      .optional()
      .describe("Run cmake --build after configuration (default: false)."),
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
    options: tool.schema
      .string()
      .optional()
      .describe("Bounded options: output=<summary|full|json>, ninja, jobs=<non-negative-int>."),
  },
  async execute(args) {
    const parsedOptions = parseRunCmakeOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const preset = args.preset;
    const sourceDir = args.sourceDir || ".";
    const buildDir = args.buildDir || "build";
    const ninja = parsedOptions.options.ninja === true;
    const timeout = args.timeout ?? 300;
    const build = args.build || false;
    const jobs = parsedOptions.options.jobs ?? 0;
    const buildTimeout = args.buildTimeout ?? 1800;
    const buildTimeoutProvided = args.buildTimeout !== undefined;
    const cmakeArgs = args.cmakeArgs || [];

    if (!Number.isFinite(timeout) || timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    if (!preset) {
      const sourceDirError = validatePathWithinRepoRoot(sourceDir, "sourceDir");
      if (sourceDirError) {
        return sourceDirError;
      }
      const buildDirError = validatePathWithinRepoRoot(buildDir, "buildDir");
      if (buildDirError) {
        return buildDirError;
      }
    }

    const isFiniteInteger = (value: number) => Number.isFinite(value) && Number.isInteger(value);

    if (!build && parsedOptions.options.jobs !== undefined) {
      return "ERROR: Invalid options: 'jobs' requires build: true.";
    }

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

    if (build) {
      cmdParts.push("--build");
    }

    if (build && jobs > 0) {
      cmdParts.push(`--jobs=${jobs}`);
    }

    if (build && (buildTimeoutProvided || buildTimeout !== 1800)) {
      cmdParts.push(`--build-timeout=${buildTimeout}`);
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
