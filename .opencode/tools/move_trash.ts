import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "path";

// --- inlined from lib/move_shared.ts ---

const TRASH_FOLDER = ".trash" as const;

function formatError(code: string, message: string, hint?: string): string {
  const lines = [`ERROR [${code}]: ${message}`];
  if (hint) lines.push("", `Hint: ${hint}`);
  return lines.join("\n");
}

function formatTrashSuccess(originalPath: string, trashPath: string, sourceType: string): string {
  return [
    `SUCCESS: Moved ${sourceType} to trash: ${originalPath}`,
    "",
    "Details:",
    `  Original: ${originalPath}`,
    `  Trash:    ${trashPath}`,
  ].join("\n");
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

function safeRelative(base: string, target: string): string {
  return path.relative(base, target) || target;
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

async function executeMoveTrash(args: { source: string }): Promise<string> {
  const cwd = process.cwd();
  const repoRoot = findRepoRoot(cwd);
  if (!args.source || typeof args.source !== "string" || !args.source.trim()) {
    return formatError("INVALID_SOURCE", "'source' parameter is required and cannot be empty.");
  }

  const sourcePath = normalizePath(args.source, cwd);
  if (!isWithinRepository(sourcePath, repoRoot)) {
    return formatError("SOURCE_OUTSIDE_REPO", `Source is outside the repository: ${sourcePath}`);
  }

  const trashRoot = path.join(repoRoot, TRASH_FOLDER);
  if (sourcePath === trashRoot || sourcePath.startsWith(trashRoot + path.sep)) {
    return formatError("ALREADY_IN_TRASH", `Source is already in trash folder: ${sourcePath}`);
  }

  const sourceInfo = statSourceType(sourcePath);
  if (sourceInfo.error || !sourceInfo.sourceType) return sourceInfo.error!;

  const relativePath = path.relative(repoRoot, sourcePath);
  const trashPath = path.join(trashRoot, relativePath);
  if (fs.existsSync(trashPath)) {
    return formatError("TRASH_PATH_EXISTS", `Trash path already exists: ${trashPath}`);
  }

  fs.mkdirSync(path.dirname(trashPath), { recursive: true });
  const moveError = tryRenameWithCleanup(sourcePath, trashPath, sourceInfo.sourceType);
  if (moveError) return moveError;

  return formatTrashSuccess(
    safeRelative(cwd, sourcePath),
    safeRelative(cwd, trashPath),
    sourceInfo.sourceType,
  );
}

export default tool({
  description: "Move a file or folder to .trash/<relative-path>.",
  args: {
    source: tool.schema.string(),
  },
  async execute(args) {
    return executeMoveTrash(args as { source: string });
  },
});
