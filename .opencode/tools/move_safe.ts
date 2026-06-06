import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "path";

// --- inlined from lib/move_shared.ts ---

interface MoveSuccessDetails {
  sourceType: "file" | "directory";
  sourcePath: string;
  destinationPath: string;
  overwritten: boolean;
}

function formatError(code: string, message: string, hint?: string): string {
  const lines = [`ERROR [${code}]: ${message}`];
  if (hint) lines.push("", `Hint: ${hint}`);
  return lines.join("\n");
}

function formatMoveSuccess(message: string, details: MoveSuccessDetails): string {
  const lines = [
    `SUCCESS: ${message}`,
    "",
    "Details:",
    `  Type: ${details.sourceType}`,
    `  From: ${details.sourcePath}`,
    `  To:   ${details.destinationPath}`,
  ];
  if (details.overwritten) lines.push("  Note: Overwrote existing destination");
  return lines.join("\n");
}

function findRepoRoot(startPath: string): string {
  let currentPath = fs.realpathSync.native(path.resolve(startPath));
  const root = path.parse(currentPath).root;
  while (currentPath !== root) {
    const gitPath = path.join(currentPath, ".git");
    try {
      const gitStat = fs.lstatSync(gitPath);
      if (gitStat.isSymbolicLink()) {
        currentPath = path.dirname(currentPath);
        continue;
      }
      if (gitStat.isDirectory() || gitStat.isFile()) {
        const canonicalCurrent = fs.realpathSync.native(currentPath);
        return path.normalize(canonicalCurrent);
      }
    } catch {
      // continue
    }
    currentPath = path.dirname(currentPath);
  }
  return path.resolve(startPath);
}

function normalizePath(input: string, cwd: string): string {
  return path.isAbsolute(input) ? path.normalize(input) : path.resolve(cwd, input);
}

function resolveCanonicalPath(targetPath: string): string {
  const resolved = path.resolve(targetPath);
  if (fs.existsSync(resolved)) {
    return path.normalize(fs.realpathSync.native(resolved));
  }

  let current = path.dirname(resolved);
  const root = path.parse(current).root;
  while (current !== root && !fs.existsSync(current)) {
    current = path.dirname(current);
  }

  if (!fs.existsSync(current)) {
    return path.normalize(resolved);
  }

  const canonicalExisting = path.normalize(fs.realpathSync.native(current));
  const suffix = path.relative(current, resolved);
  return path.normalize(path.join(canonicalExisting, suffix));
}

function isWithinRepository(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = resolveCanonicalPath(targetPath);
  const normalizedRoot = path.normalize(fs.realpathSync.native(path.resolve(repoRoot)));
  return normalizedTarget === normalizedRoot || normalizedTarget.startsWith(normalizedRoot + path.sep);
}

function resolveDestinationPath(sourcePath: string, destination: string, cwd: string): string {
  const isDestDir = destination.endsWith("/") || destination.endsWith(path.sep);
  if (isDestDir) {
    const destDir = normalizePath(destination, cwd);
    return path.join(destDir, path.basename(sourcePath));
  }
  return normalizePath(destination, cwd);
}

function safeRelative(base: string, target: string): string {
  return path.relative(base, target) || target;
}

function validateSourceAndDestination(
  source: string,
  destination: string,
  cwd: string,
): { sourcePath?: string; destPath?: string; error?: string } {
  if (!source || typeof source !== "string" || !source.trim()) {
    return { error: formatError("INVALID_SOURCE", "'source' parameter is required and cannot be empty.") };
  }
  if (!destination || typeof destination !== "string" || !destination.trim()) {
    return {
      error: formatError("INVALID_DESTINATION", "'destination' parameter is required and cannot be empty."),
    };
  }
  const sourcePath = normalizePath(source, cwd);
  const destPath = resolveDestinationPath(sourcePath, destination, cwd);

  if (sourcePath === destPath) {
    return { error: formatError("SAME_PATH", "Source and destination paths are identical.") };
  }

  return { sourcePath, destPath };
}

function ensureRepoBoundary(sourcePath: string, destPath: string, repoRoot: string): string | null {
  if (!isWithinRepository(sourcePath, repoRoot)) {
    return formatError("SOURCE_OUTSIDE_REPO", `Source is outside the repository: ${sourcePath}`);
  }
  if (!isWithinRepository(destPath, repoRoot)) {
    return formatError("DEST_OUTSIDE_REPO", `Destination is outside the repository: ${destPath}`);
  }
  return null;
}

function statSourceType(sourcePath: string): { sourceType?: "file" | "directory"; error?: string } {
  if (!fs.existsSync(sourcePath)) {
    return { error: formatError("SOURCE_NOT_FOUND", `Source does not exist: ${sourcePath}`) };
  }
  try {
    const stat = fs.statSync(sourcePath);
    return { sourceType: stat.isDirectory() ? "directory" : "file" };
  } catch (err) {
    return { error: formatError("SOURCE_STAT_ERROR", `Cannot read source metadata: ${sourcePath}`, String(err)) };
  }
}

function tryRenameWithCleanup(
  sourcePath: string,
  destPath: string,
  sourceType: "file" | "directory",
): string | null {
  try {
    fs.renameSync(sourcePath, destPath);
    return null;
  } catch (err) {
    const error = err as NodeJS.ErrnoException;
    const existsSource = fs.existsSync(sourcePath);
    const existsDest = fs.existsSync(destPath);
    if (!existsSource && existsDest) {
      return formatError(
        `MOVE_ERROR_${error.code || "UNKNOWN"}`,
        `Move operation reached partial state for ${sourceType}: ${sourcePath} -> ${destPath}`,
        "Manual intervention required: destination exists while source is missing.",
      );
    }
    if (existsSource && existsDest) {
      try {
        fs.rmSync(destPath, { recursive: true, force: true });
      } catch {
        return formatError(
          `MOVE_ERROR_${error.code || "UNKNOWN"}`,
          `Failed to move ${sourceType}: ${sourcePath} -> ${destPath}`,
          "Cleanup failed; remove incomplete destination and retry.",
        );
      }
      return formatError(
        `MOVE_ERROR_${error.code || "UNKNOWN"}`,
        `Failed to move ${sourceType}: ${sourcePath} -> ${destPath}`,
        "Incomplete destination cleanup succeeded. Source preserved.",
      );
    }
    if (error.code === "EXDEV") {
      return formatError(
        "CROSS_DEVICE_NOT_ALLOWED",
        `Cross-filesystem moves are not supported: ${sourcePath} -> ${destPath}`,
      );
    }
    return formatError(
      `MOVE_ERROR_${error.code || "UNKNOWN"}`,
      `Failed to move ${sourceType}: ${sourcePath} -> ${destPath}`,
      error.message,
    );
  }
}

// --- end inlined ---

async function executeMoveSafe(args: { source: string; destination: string }): Promise<string> {
  const cwd = process.cwd();
  const repoRoot = findRepoRoot(cwd);
  const parsed = validateSourceAndDestination(args.source, args.destination, cwd);
  if (parsed.error || !parsed.sourcePath || !parsed.destPath) return parsed.error!;

  const boundaryError = ensureRepoBoundary(parsed.sourcePath, parsed.destPath, repoRoot);
  if (boundaryError) return boundaryError;

  const sourceInfo = statSourceType(parsed.sourcePath);
  if (sourceInfo.error || !sourceInfo.sourceType) return sourceInfo.error!;

  if (fs.existsSync(parsed.destPath)) {
    return formatError("DEST_EXISTS", `Destination already exists: ${parsed.destPath}`);
  }

  fs.mkdirSync(path.dirname(parsed.destPath), { recursive: true });
  const moveError = tryRenameWithCleanup(parsed.sourcePath, parsed.destPath, sourceInfo.sourceType);
  if (moveError) return moveError;

  return formatMoveSuccess(
    `Moved ${sourceInfo.sourceType}: ${safeRelative(cwd, parsed.sourcePath)} -> ${safeRelative(cwd, parsed.destPath)}`,
    {
      sourceType: sourceInfo.sourceType,
      sourcePath: safeRelative(cwd, parsed.sourcePath),
      destinationPath: safeRelative(cwd, parsed.destPath),
      overwritten: false,
    },
  );
}

export default tool({
  description: "Move a file or folder safely (no overwrite, no trash).",
  args: {
    source: tool.schema.string(),
    destination: tool.schema.string(),
  },
  async execute(args) {
    return executeMoveSafe(args as { source: string; destination: string });
  },
});
