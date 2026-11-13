import path from "node:path";
import { accessSync, constants, existsSync, realpathSync, statSync } from "node:fs";

const S_IFMT = 0o170000;
const S_IFREG = 0o100000;
const S_IFDIR = 0o040000;

export const MISSING_SCRIPT_HINT =
  "Missing backing script .opencode/tools/run_sanitizers.py (dependency #1379).";
export const ALLOWED_OUTPUT_MODES = ["summary", "full", "json"] as const;
export const ALLOWED_SANITIZERS = ["asan", "tsan", "ubsan"] as const;
export const ADVANCED_ONLY_KEYS = ["suppressions", "options", "normalDuration", "extraArgs"] as const;

export type SanitizerArgs = Record<string, unknown>;

function getRepoRoot(): string {
  return realpathSync(path.resolve(import.meta.dir, "../.."));
}

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

function validatePathWithinRepo(pathValue: string, label: "buildDir"): string | undefined {
  try {
    if (!existsSync(pathValue)) {
      return `ERROR: ${label} path does not exist: ${pathValue}`;
    }
    if (!isStatDirectory(statSync(pathValue))) {
      return `ERROR: ${label} path is not a directory: ${pathValue}`;
    }

    const repoRoot = getRepoRoot();
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

function validateExecutablePathWithinRepo(buildDirPath: string, pathValue: string): {
  resolvedPath?: string;
  error?: string;
} {
  try {
    const repoRoot = getRepoRoot();
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

function validateSuppressionsPathWithinRepo(pathValue: string): {
  resolvedPath?: string;
  error?: string;
} {
  try {
    const repoRoot = getRepoRoot();
    const candidatePath = path.isAbsolute(pathValue)
      ? path.resolve(pathValue)
      : path.resolve(repoRoot, pathValue);

    if (!existsSync(candidatePath)) {
      return { error: `ERROR: suppressions path does not exist: ${pathValue}` };
    }

    const canonicalPath = realpathSync(candidatePath);
    const repoRel = path.relative(repoRoot, canonicalPath);
    if (repoRel.startsWith("..") || path.isAbsolute(repoRel)) {
      return {
        error: `ERROR: suppressions path resolves outside repository root: ${pathValue} (canonical: ${canonicalPath})`,
      };
    }

    if (!isStatFile(statSync(canonicalPath))) {
      return { error: `ERROR: suppressions path is not a file: ${pathValue}` };
    }

    accessSync(canonicalPath, constants.R_OK);
    return { resolvedPath: canonicalPath };
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return { error: `ERROR: suppressions path is not readable: ${pathValue} (${message})` };
  }
}

function formatSanitizerExecutionFailure(error: any): string {
  const stdout = error?.stdout?.toString?.() || "";
  const stderr = error?.stderr?.toString?.() || "";
  const message = error?.message || "Unknown error";
  const combinedLower = `${stderr} ${message}`.toLowerCase();
  const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
  const diagnostic = stdout.trim() || stderr.trim() || `Failed to run sanitizer: ${message}`;

  return `ERROR: Sanitizer run failed\n\n${diagnostic}${hint}`;
}

export function hasAdvancedSanitizerKey(args: SanitizerArgs): boolean {
  return ADVANCED_ONLY_KEYS.some((key) => Object.hasOwn(args, key));
}

export async function executeSanitizers(args: SanitizerArgs): Promise<string> {
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

  if (typeof timeout !== "number" || !Number.isInteger(timeout) || timeout <= 0) {
    return `ERROR: Timeout must be a positive integer (received ${timeout}).`;
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

  let validatedSuppressionsPath: string | undefined;
  if (suppressions) {
    const suppressionsPathValidation = validateSuppressionsPathWithinRepo(suppressions);
    if (suppressionsPathValidation.error) {
      return suppressionsPathValidation.error;
    }
    validatedSuppressionsPath = suppressionsPathValidation.resolvedPath;
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

  if (validatedSuppressionsPath) {
    cmdParts.push(`--suppressions=${validatedSuppressionsPath}`);
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
    return formatSanitizerExecutionFailure(error);
  }
}
