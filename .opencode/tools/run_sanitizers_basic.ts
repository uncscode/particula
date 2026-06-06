import { tool } from "@opencode-ai/plugin";
import path from "node:path";
import { existsSync, realpathSync, statSync } from "node:fs";

// --- Inlined from run_sanitizers_advanced.ts ---

const S_IFMT = 0o170000;
const S_IFREG = 0o100000;
const S_IFDIR = 0o040000;

function isStatFile(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isFile === "function") return s.isFile();
  if (typeof s.isFile === "boolean") return s.isFile;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFREG;
}

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

function validatePathWithinRepo(pathValue: string, label: "sourceDir" | "buildDir"): string | undefined {
  try {
    if (!existsSync(pathValue)) {
      return `ERROR: ${label} path does not exist: ${pathValue}`;
    }
    if (!isStatDirectory(statSync(pathValue))) {
      return `ERROR: ${label} path is not a directory: ${pathValue}`;
    }

    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = realpathSync(pathValue);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: ${label} path resolves outside repository root: ${pathValue} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid ${label} path: ${pathValue} (${message})`;
  }

  return undefined;
}

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_sanitizers.py (dependency #1379).";
const ALLOWED_OUTPUT_MODES = ["summary", "full", "json"] as const;
const ALLOWED_SANITIZERS = ["asan", "tsan", "ubsan"] as const;

type SanitizerArgs = Record<string, unknown>;

function validateExecutablePathWithinRepo(buildDirPath: string, pathValue: string): {
  resolvedPath?: string;
  error?: string;
} {
  try {
    const repoRoot = realpathSync(process.cwd());
    const canonicalBuildDir = realpathSync(path.resolve(buildDirPath));
    const candidatePath = path.isAbsolute(pathValue)
      ? path.resolve(pathValue)
      : path.resolve(canonicalBuildDir, pathValue);

    if (!existsSync(candidatePath)) {
      return { error: `ERROR: executable path does not exist: ${pathValue}` };
    }

    const canonicalPath = realpathSync(candidatePath);

    const repoRel = path.relative(repoRoot, canonicalPath);
    if (repoRel.startsWith("..") || path.isAbsolute(repoRel)) {
      return {
        error: `ERROR: executable path resolves outside repository root: ${pathValue} (canonical: ${canonicalPath})`,
      };
    }

    const buildRel = path.relative(canonicalBuildDir, canonicalPath);
    if (buildRel.startsWith("..") || path.isAbsolute(buildRel)) {
      return {
        error: `ERROR: executable path resolves outside buildDir: ${pathValue} (canonical: ${canonicalPath})`,
      };
    }

    if (!isStatFile(statSync(canonicalPath))) {
      return { error: `ERROR: executable path is not a file: ${pathValue}` };
    }

    return { resolvedPath: canonicalPath };
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return { error: `ERROR: invalid executable path: ${pathValue} (${message})` };
  }
}

async function executeSanitizersAdvanced(args: SanitizerArgs): Promise<string> {
  const outputMode = (args.outputMode as (typeof ALLOWED_OUTPUT_MODES)[number] | undefined) || "summary";
  const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : "";
  const executable = typeof args.executable === "string" ? args.executable.trim() : "";
  const sanitizer = args.sanitizer as (typeof ALLOWED_SANITIZERS)[number] | string | undefined;
  const timeout = args.timeout ?? 600;
  const suppressions = typeof args.suppressions === "string" ? args.suppressions.trim() : "";
  const options = typeof args.options === "string" ? args.options.trim() : "";
  const normalDuration = args.normalDuration as number | undefined;
  const extraArgs = args.extraArgs as unknown;

  if (!buildDir) {
    return "ERROR: buildDir is required. Provide the working directory containing the sanitizer-enabled binary.";
  }
  if (!executable) {
    return "ERROR: executable is required. Provide the sanitizer-enabled binary path.";
  }
  if (!sanitizer) {
    return "ERROR: sanitizer is required. Choose one of: asan, tsan, ubsan.";
  }
  if (!ALLOWED_SANITIZERS.includes(sanitizer as any)) {
    return `ERROR: sanitizer must be one of asan, tsan, ubsan (received ${sanitizer}).`;
  }
  if (!ALLOWED_OUTPUT_MODES.includes(outputMode as any)) {
    return `ERROR: outputMode must be one of summary, full, json (received ${outputMode}).`;
  }
  const buildDirPathError = validatePathWithinRepo(buildDir, "buildDir");
  if (buildDirPathError) {
    return buildDirPathError;
  }
  const canonicalBuildDir = realpathSync(path.resolve(buildDir));

  const executablePathValidation = validateExecutablePathWithinRepo(canonicalBuildDir, executable);
  if (executablePathValidation.error) {
    return executablePathValidation.error;
  }
  const validatedExecutablePath = executablePathValidation.resolvedPath as string;

  if (typeof timeout !== "number" || !Number.isFinite(timeout) || timeout <= 0) {
    return `ERROR: Timeout must be positive (received ${timeout}).`;
  }
  if (
    normalDuration !== undefined
    && (typeof normalDuration !== "number" || !Number.isFinite(normalDuration) || normalDuration <= 0)
  ) {
    return `ERROR: normalDuration must be positive when provided (received ${normalDuration}).`;
  }
  if (extraArgs !== undefined) {
    if (!Array.isArray(extraArgs) || !extraArgs.every((item) => typeof item === "string")) {
      return "ERROR: extraArgs must be an array of strings when provided.";
    }
  }

  const cmdParts: (string | number)[] = [
    "python3",
    `${import.meta.dir}/run_sanitizers.py`,
    `--build-dir=${canonicalBuildDir}`,
    `--executable=${validatedExecutablePath}`,
    `--sanitizer=${sanitizer}`,
    `--output-mode=${outputMode}`,
    `--timeout=${timeout}`,
  ];

  if (suppressions) {
    cmdParts.push(`--suppressions=${suppressions}`);
  }
  if (options) {
    cmdParts.push(`--options=${options}`);
  }
  if (normalDuration !== undefined) {
    cmdParts.push(`--normal-duration=${normalDuration}`);
  }
  if (Array.isArray(extraArgs) && extraArgs.length > 0) {
    cmdParts.push("--", ...extraArgs);
  }

  try {
    const result = await Bun.$`${cmdParts}`.text();
    return result || "Sanitizer run completed but returned no output.";
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
      return `ERROR: Sanitizer run failed\n\n${stderr}${hint}`;
    }
    if (combinedLower.includes("enoent")) {
      return `ERROR: Failed to run sanitizer: ${message}\n${MISSING_SCRIPT_HINT}`;
    }
    return `ERROR: Failed to run sanitizer: ${message}`;
  }
}

// --- Basic wrapper gate ---

const FORBIDDEN_ADVANCED_KEYS = ["suppressions", "options", "normalDuration", "extraArgs"] as const;

// --- Tool definition ---

export default tool({
  description: `Run routine sanitizer checks (buildDir + executable + sanitizer only).

Use this routine wrapper for standard sanitizer runs.
Advanced controls (suppressions/options/normalDuration/extraArgs) are intentionally blocked; use run_sanitizers_advanced for those options.`,
  args: {
    outputMode: tool.schema.enum(ALLOWED_OUTPUT_MODES).optional(),
    buildDir: tool.schema.string(),
    executable: tool.schema.string(),
    sanitizer: tool.schema.enum(ALLOWED_SANITIZERS),
    timeout: tool.schema.number().optional(),
    suppressions: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    normalDuration: tool.schema.number().optional(),
    extraArgs: tool.schema.array(tool.schema.string()).optional(),
  },
  async execute(args) {
    for (const key of FORBIDDEN_ADVANCED_KEYS) {
      if (Object.hasOwn(args, key)) {
        return `ERROR: run_sanitizers_basic does not accept advanced option '${key}'. Use run_sanitizers_advanced.`;
      }
    }

    return executeSanitizersAdvanced(args as Record<string, unknown>);
  },
});
