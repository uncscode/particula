import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync } from "node:fs";
import path from "node:path";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cmake.py (dependency #1352).";

type OutputMode = "summary" | "full" | "json";

type ParsedBuildOptions = {
  outputMode?: OutputMode;
  jobs?: number;
};

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

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

function validatePathWithinRepoRoot(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }

  try {
    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = resolvePathForRepoCheck(value);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: buildDir path resolves outside repository root: ${value} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid buildDir path: ${value} (${message})`;
  }

  return undefined;
}

function parseBuildOptions(rawOptions: unknown):
  | { ok: true; options: ParsedBuildOptions }
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

  const parsed: ParsedBuildOptions = {};
  for (const token of normalized.split(/\s+/)) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }
    if (separatorIndex === -1) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    const name = token.slice(0, separatorIndex);
    const value = token.slice(separatorIndex + 1).trim();
    if (!value) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    if (name === "output") {
      if (!OUTPUT_MODES.has(value as OutputMode)) {
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

    return {
      ok: false,
      error: `ERROR: Invalid options token '${token}': token is not supported.`,
    };
  }

  return { ok: true, options: parsed };
}

export default tool({
  description:
    "Run CMake build-only operations. Requires build context and owns build flags/jobs/timeouts.",
  args: {
    preset: tool.schema.string().optional(),
    buildDir: tool.schema.string().optional(),
    buildTimeout: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    const parsedOptions = parseBuildOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const preset = typeof args.preset === "string" ? args.preset.trim() : undefined;
    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : undefined;
    const jobs = parsedOptions.options.jobs ?? 0;
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
    if (!preset) {
      const buildDirError = validatePathWithinRepoRoot(buildDir);
      if (buildDirError) {
        return buildDirError;
      }
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
