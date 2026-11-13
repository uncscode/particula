import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

export function validateCwdWithinRepo(cwd: string | undefined): string | undefined {
  if (!cwd) {
    return undefined;
  }

  try {
    if (!existsSync(cwd)) {
      return `ERROR: cwd path does not exist: ${cwd}`;
    }
    if (!isStatDirectory(statSync(cwd))) {
      return `ERROR: cwd path is not a directory: ${cwd}`;
    }

    const repoRoot = realpathSync(process.cwd());
    const resolvedCwd = realpathSync(cwd);
    const rel = path.relative(repoRoot, resolvedCwd);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: cwd path resolves outside repository root: ${cwd} (canonical: ${resolvedCwd})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid cwd path: ${cwd} (${message})`;
  }

  return undefined;
}

export function resolvePathForRepoCheck(value: string, baseDir: string): string {
  const candidate = path.isAbsolute(value) ? path.normalize(value) : path.resolve(baseDir, value);
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

export function validatePathWithinRepo(
  value: string | undefined,
  parameterName: string,
  cwd?: string,
): string | undefined {
  if (!value) {
    return undefined;
  }

  try {
    const repoRoot = realpathSync(process.cwd());
    const baseDir = cwd ? realpathSync(cwd) : repoRoot;
    const resolvedPath = resolvePathForRepoCheck(value, baseDir);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: ${parameterName} resolves outside repository root: ${value} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid ${parameterName} path: ${value} (${message})`;
  }

  return undefined;
}
