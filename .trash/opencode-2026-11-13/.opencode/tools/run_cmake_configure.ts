import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync } from "node:fs";
import path from "node:path";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cmake.py (dependency #1352).";

type OutputMode = "summary" | "full" | "json";

type ParsedConfigureOptions = {
  outputMode?: OutputMode;
  ninja?: true;
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

function validatePathWithinRepoRoot(
  value: string | undefined,
  parameterName: "sourceDir" | "buildDir",
): string | undefined {
  if (!value) {
    return undefined;
  }

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

function parseConfigureOptions(rawOptions: unknown):
  | { ok: true; options: ParsedConfigureOptions }
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

  const parsed: ParsedConfigureOptions = {};
  for (const token of normalized.split(/\s+/)) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }

    if (separatorIndex === -1) {
      if (token !== "ninja") {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
        };
      }
      if (parsed.ninja) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.ninja = true;
      continue;
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

export default tool({
  description:
    "Run CMake configure-only operations. Accepts preset/manual configure args and never emits build flags.",
  args: {
    preset: tool.schema.string().optional(),
    sourceDir: tool.schema.string().optional(),
    buildDir: tool.schema.string().optional(),
    timeout: tool.schema.number().optional(),
    cmakeArgs: tool.schema.array(tool.schema.string()).optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    const parsedOptions = parseConfigureOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const preset = typeof args.preset === "string" ? args.preset.trim() : undefined;
    const sourceDir = typeof args.sourceDir === "string" ? args.sourceDir.trim() : args.sourceDir;
    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : args.buildDir;
    const ninja = parsedOptions.options.ninja === true;
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

    if (!preset) {
      const sourceDirError = validatePathWithinRepoRoot(sourceDir || ".", "sourceDir");
      if (sourceDirError) {
        return sourceDirError;
      }
      const buildDirError = validatePathWithinRepoRoot(buildDir || "build", "buildDir");
      if (buildDirError) {
        return buildDirError;
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
