/**
 * Move Tool for OpenCode Integration
 *
 * Moves a single file or folder to another location within the repository.
 * This is a safe, atomic move operation that handles both files and directories.
 *
 * SECURITY: This tool enforces repository boundary restrictions:
 * - Destination must be within the current working directory (repository root)
 * - Cross-device moves are not supported (prevents moving to external filesystems)
 * - Trash mode moves files to .trash/ folder for audit trail (git tracks the move)
 */

import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "path";

/**
 * Check if a path is within the repository boundary.
 * Prevents moving files outside the current working directory.
 */
function isWithinRepository(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));

  // Ensure the target path starts with the repo root
  // Add path.sep to prevent matching partial directory names
  // e.g., /home/user/repo vs /home/user/repo-other
  return (
    normalizedTarget === normalizedRoot ||
    normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

/** Trash folder for soft-deleted files (tracked by git for audit trail) */
const TRASH_FOLDER = ".trash" as const;

/** Result type for move operations */
interface MoveResult {
  success: boolean;
  message: string;
  details?: {
    sourceType: "file" | "directory";
    sourcePath: string;
    destinationPath: string;
    overwritten: boolean;
    crossDevice: boolean;
  };
}

/** Result type for trash operations */
interface TrashResult {
  success: boolean;
  message: string;
  details?: {
    originalPath: string;
    trashPath: string;
  };
}

/** Format a successful result */
function formatSuccess(result: MoveResult): string {
  const { message, details } = result;
  if (!details) return `SUCCESS: ${message}`;

  const lines = [
    `SUCCESS: ${message}`,
    "",
    "Details:",
    `  Type: ${details.sourceType}`,
    `  From: ${details.sourcePath}`,
    `  To:   ${details.destinationPath}`,
  ];

  if (details.overwritten) {
    lines.push("  Note: Overwrote existing destination");
  }
  if (details.crossDevice) {
    lines.push("  Note: Cross-device move (copied then deleted source)");
  }

  return lines.join("\n");
}

/** Format an error result */
function formatError(code: string, message: string, hint?: string): string {
  const lines = [`ERROR [${code}]: ${message}`];
  if (hint) {
    lines.push("", `Hint: ${hint}`);
  }
  return lines.join("\n");
}

/** Format a successful trash result */
function formatTrashSuccess(result: TrashResult): string {
  const { message, details } = result;
  if (!details) return `SUCCESS: ${message}`;

  const lines = [
    `SUCCESS: ${message}`,
    "",
    "Details:",
    `  Original: ${details.originalPath}`,
    `  Trash:    ${details.trashPath}`,
    "",
    "Note: File moved to .trash/ folder. Git will track this as a move (not deletion).",
    "      Commit this change to preserve the audit trail.",
    "      Permanently delete .trash/ contents in a separate cleanup PR.",
  ];

  return lines.join("\n");
}

/**
 * Find the repository root by looking for .git directory.
 * Handles both regular repos and worktrees.
 */
function findRepoRoot(startPath: string): string {
  let currentPath = startPath;
  const root = path.parse(currentPath).root;

  while (currentPath !== root) {
    const gitPath = path.join(currentPath, ".git");
    try {
      const stat = fs.statSync(gitPath);
      // .git can be a directory (regular repo) or file (worktree)
      if (stat.isDirectory() || stat.isFile()) {
        return currentPath;
      }
    } catch {
      // .git doesn't exist here, keep looking
    }
    currentPath = path.dirname(currentPath);
  }

  // Fallback to cwd if no .git found
  return startPath;
}

/**
 * Handle the trash flag operation - moves file to .trash/ folder.
 * Preserves directory structure for clarity and audit trail.
 * Works with both files and directories.
 */
async function handleTrash(source: string): Promise<string> {
  const cwd: string = process.cwd();
  const repoRoot: string = findRepoRoot(cwd);
  const sourcePath: string = path.isAbsolute(source)
    ? path.normalize(source)
    : path.resolve(cwd, source);

  // Check source exists
  let sourceExists: boolean;
  try {
    sourceExists = fs.existsSync(sourcePath);
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    return formatError(
      "SOURCE_ACCESS_ERROR",
      `Cannot access source path: ${sourcePath}`,
      `Check permissions and path validity. System error: ${errorMessage}`
    );
  }

  if (!sourceExists) {
    return formatError(
      "SOURCE_NOT_FOUND",
      `Source does not exist: ${sourcePath}`,
      "Verify the path is correct and the file/directory exists."
    );
  }

  // Get source stats
  let sourceStats: fs.Stats;
  try {
    sourceStats = fs.statSync(sourcePath);
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    return formatError(
      "SOURCE_STAT_ERROR",
      `Cannot read source metadata: ${sourcePath}`,
      `Check permissions. System error: ${errorMessage}`
    );
  }

  const sourceType: "file" | "directory" = sourceStats.isDirectory()
    ? "directory"
    : "file";

  // Check if source is already in .trash/
  const trashFolder = path.join(repoRoot, TRASH_FOLDER);
  if (sourcePath.startsWith(trashFolder + path.sep) || sourcePath === trashFolder) {
    return formatError(
      "ALREADY_IN_TRASH",
      `Source is already in trash folder: ${sourcePath}`,
      "This file/directory is already in .trash/. To restore, move it back to its original location."
    );
  }

  // Build the trash path preserving directory structure relative to repo root
  const relativePath: string = path.relative(repoRoot, sourcePath);
  const trashPath: string = path.join(trashFolder, relativePath);

  // Check if trash path already exists
  let trashExists: boolean;
  try {
    trashExists = fs.existsSync(trashPath);
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    return formatError(
      "TRASH_PATH_ACCESS_ERROR",
      `Cannot access trash path: ${trashPath}`,
      `Check permissions. System error: ${errorMessage}`
    );
  }

  if (trashExists) {
    return formatError(
      "TRASH_PATH_EXISTS",
      `Trash path already exists: ${trashPath}`,
      "A file/directory already exists at this location in .trash/. " +
        "Remove it first or use a different approach."
    );
  }

  // Create .trash/ folder and parent directories if needed
  const trashDir: string = path.dirname(trashPath);
  try {
    fs.mkdirSync(trashDir, { recursive: true });
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    return formatError(
      "TRASH_DIR_CREATE_ERROR",
      `Cannot create trash directory: ${trashDir}`,
      `Check permissions and disk space. System error: ${errorMessage}`
    );
  }

  // Perform the move to trash
  try {
    fs.renameSync(sourcePath, trashPath);
  } catch (err: unknown) {
    const error = err as NodeJS.ErrnoException;
    const errorMessage = error.message || String(error);
    const errorCode = error.code || "UNKNOWN";

    return formatError(
      `TRASH_MOVE_ERROR_${errorCode}`,
      `Failed to move to trash: ${sourcePath}`,
      `System error: ${errorMessage}`
    );
  }

  // Success
  const relativeSrc: string = path.relative(cwd, sourcePath) || sourcePath;
  const relativeTrash: string = path.relative(cwd, trashPath) || trashPath;

  const result: TrashResult = {
    success: true,
    message: `Moved ${sourceType} to trash: ${relativeSrc}`,
    details: {
      originalPath: relativeSrc,
      trashPath: relativeTrash,
    },
  };

  return formatTrashSuccess(result);
}

export default tool({
  description: `Move a single file or folder to another location within the repository.

This tool performs a safe move operation for files and directories. It validates
that the source exists and the destination path is valid before performing the move.

USAGE EXAMPLES:
• Move a file: { source: "src/old.ts", destination: "src/new.ts" }
• Move to directory: { source: "src/file.ts", destination: "lib/" }
• Move a folder: { source: "src/old-dir", destination: "src/new-dir" }
• Rename in place: { source: "src/utils.ts", destination: "src/helpers.ts" }
• Move to trash: { source: "src/deprecated.ts", destination: "", trash: true }

BEHAVIOR:
- If destination is a directory (ends with /), the source is moved INTO it
- If destination is a file path, the source is renamed to that path
- Parent directories are created automatically if they don't exist
- Fails if destination already exists (use overwrite: true to force)

SECURITY RESTRICTIONS:
- Destination MUST be within the repository (current working directory)
- Moving files outside the repository is NOT allowed
- Cross-filesystem moves are NOT supported

TRASH MODE (soft-delete with audit trail):
- If trash: true, moves file/directory to .trash/<original-path>
- The destination parameter is IGNORED in trash mode
- Works with both files AND directories
- Git tracks this as a MOVE (not a deletion) preserving audit trail
- The .trash/ folder is created automatically at the repository root
- Directory structure is preserved: src/foo.ts -> .trash/src/foo.ts
- Permanently delete .trash/ contents in a separate cleanup PR after review

WHY TRASH MODE EXISTS:
- Git tracks moves, so the file history is preserved
- Unlike renaming to .TO_BE_DELETED, git add --all won't stage a deletion
- Maintainers can review .trash/ before permanent removal
- Files can be restored by moving them back out of .trash/

OUTPUT FORMAT:
- Success: "SUCCESS: Moved <type>: <source> -> <destination>" with details
- Error: "ERROR [<CODE>]: <message>" with optional hint`,

  args: {
    source: tool.schema
      .string()
      .describe(`Source path of the file or folder to move.

Can be absolute or relative to the current working directory.
Must exist and be accessible.

EXAMPLES:
• "src/components/Button.tsx"
• "lib/utils"
• "./old-name.ts"`),

    destination: tool.schema
      .string()
      .describe(`Destination path for the file or folder.

Can be absolute or relative to the current working directory.

BEHAVIOR:
• If ends with "/", source is moved INTO this directory
• Otherwise, source is renamed/moved to this exact path
• Parent directories are created if they don't exist

EXAMPLES:
• "src/components/ui/Button.tsx" - move and/or rename
• "src/components/ui/" - move into this directory (keeps original name)
• "lib/helpers" - move/rename folder`),

    overwrite: tool.schema
      .boolean()
      .optional()
      .describe(`Allow overwriting existing destination.

DEFAULT: false

When true, will replace any existing file or directory at the destination.
Use with caution as this cannot be undone.

EXAMPLE: overwrite: true`),

    trash: tool.schema
      .boolean()
      .optional()
      .describe(`Move file/directory to .trash/ folder for soft-delete with audit trail.

DEFAULT: false

PURPOSE:
Move files to a .trash/ folder instead of deleting them. Git tracks this as a
MOVE operation, preserving the audit trail in git history. This solves the problem
where renaming files (like adding .TO_BE_DELETED suffix) causes git to see it as
a deletion when staged.

WHEN TO USE:
- Deprecating files that should be reviewed before permanent removal
- Staged deletions in automated workflows
- When you need git history to show a move (not a deletion)
- When files need review before permanent removal in a cleanup PR

BEHAVIOR:
- Source is moved to .trash/<original-path> preserving directory structure
- The 'destination' parameter is IGNORED (pass empty string or omit)
- Works with both files AND directories
- The .trash/ folder is auto-created at repository root (handles worktrees)
- Fails if file already exists in .trash/ at that path

EXAMPLE: { source: "src/deprecated.ts", destination: "", trash: true }
RESULT:  src/deprecated.ts -> .trash/src/deprecated.ts

EXAMPLE: { source: "adw/old-module/", destination: "", trash: true }
RESULT:  adw/old-module/ -> .trash/adw/old-module/`),
  },

  async execute(args): Promise<string> {
    const { source, destination, overwrite, trash: moveToTrash } = args;
    const shouldOverwrite: boolean = overwrite ?? false;
    const shouldMoveToTrash: boolean = moveToTrash ?? false;

    // === VALIDATION ===

    // Validate source parameter
    if (!source || typeof source !== "string" || !source.trim()) {
      return formatError(
        "INVALID_SOURCE",
        "'source' parameter is required and cannot be empty.",
        'Provide a valid file or directory path, e.g., { source: "src/file.ts", destination: "lib/" }'
      );
    }

    // Handle trash flag - move to .trash/ folder
    if (shouldMoveToTrash) {
      return handleTrash(source);
    }

    // Validate destination parameter (only required when not deleting)
    if (!destination || typeof destination !== "string" || !destination.trim()) {
      return formatError(
        "INVALID_DESTINATION",
        "'destination' parameter is required and cannot be empty.",
        'Provide a valid destination path, e.g., { source: "src/file.ts", destination: "lib/file.ts" }'
      );
    }

    // === PATH RESOLUTION ===

    const cwd: string = process.cwd();
    const sourcePath: string = path.isAbsolute(source)
      ? path.normalize(source)
      : path.resolve(cwd, source);

    const isDestDir: boolean =
      destination.endsWith("/") || destination.endsWith(path.sep);

    let destPath: string;
    if (isDestDir) {
      // Move INTO the destination directory, keeping the original name
      const destDir: string = path.isAbsolute(destination)
        ? path.normalize(destination)
        : path.resolve(cwd, destination);
      destPath = path.join(destDir, path.basename(sourcePath));
    } else {
      // Move/rename to the exact destination path
      destPath = path.isAbsolute(destination)
        ? path.normalize(destination)
        : path.resolve(cwd, destination);
    }

    // Prevent moving to same location
    if (sourcePath === destPath) {
      return formatError(
        "SAME_PATH",
        "Source and destination paths are identical.",
        "Provide a different destination path to move or rename the file/directory."
      );
    }

    // === REPOSITORY BOUNDARY CHECK ===

    // Ensure destination is within the repository
    if (!isWithinRepository(destPath, cwd)) {
      return formatError(
        "DEST_OUTSIDE_REPO",
        `Destination is outside the repository: ${destPath}`,
        `All moves must stay within the repository root (${cwd}). ` +
          "Use { trash: true } to move files to .trash/ folder instead of moving them outside."
      );
    }

    // Also verify source is within repository (prevent accessing external files)
    if (!isWithinRepository(sourcePath, cwd)) {
      return formatError(
        "SOURCE_OUTSIDE_REPO",
        `Source is outside the repository: ${sourcePath}`,
        `All operations must be within the repository root (${cwd}).`
      );
    }

    // === SOURCE VALIDATION ===

    // Check source exists
    let sourceExists: boolean;
    try {
      sourceExists = fs.existsSync(sourcePath);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      return formatError(
        "SOURCE_ACCESS_ERROR",
        `Cannot access source path: ${sourcePath}`,
        `Check permissions and path validity. System error: ${errorMessage}`
      );
    }

    if (!sourceExists) {
      return formatError(
        "SOURCE_NOT_FOUND",
        `Source does not exist: ${sourcePath}`,
        "Verify the path is correct and the file/directory exists."
      );
    }

    // Get source stats
    let sourceStats: fs.Stats;
    try {
      sourceStats = fs.statSync(sourcePath);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      return formatError(
        "SOURCE_STAT_ERROR",
        `Cannot read source metadata: ${sourcePath}`,
        `Check permissions. System error: ${errorMessage}`
      );
    }

    const sourceType: "file" | "directory" = sourceStats.isDirectory()
      ? "directory"
      : "file";

    // === DESTINATION VALIDATION ===

    let destExists: boolean;
    let didOverwrite: boolean = false;

    try {
      destExists = fs.existsSync(destPath);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      return formatError(
        "DEST_ACCESS_ERROR",
        `Cannot access destination path: ${destPath}`,
        `Check permissions and path validity. System error: ${errorMessage}`
      );
    }

    if (destExists) {
      if (!shouldOverwrite) {
        let destType: "file" | "directory";
        try {
          const destStats: fs.Stats = fs.statSync(destPath);
          destType = destStats.isDirectory() ? "directory" : "file";
        } catch {
          destType = "file"; // Default assumption
        }

        return formatError(
          "DEST_EXISTS",
          `Destination already exists (${destType}): ${destPath}`,
          "Use { overwrite: true } to replace the existing destination."
        );
      }

      // Remove existing destination with overwrite flag
      try {
        fs.rmSync(destPath, { recursive: true, force: true });
        didOverwrite = true;
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        return formatError(
          "DEST_REMOVE_ERROR",
          `Cannot remove existing destination: ${destPath}`,
          `Check permissions. System error: ${errorMessage}`
        );
      }
    }

    // === CREATE PARENT DIRECTORIES ===

    const destDir: string = path.dirname(destPath);
    let destDirExists: boolean;

    try {
      destDirExists = fs.existsSync(destDir);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      return formatError(
        "DEST_DIR_ACCESS_ERROR",
        `Cannot access destination directory: ${destDir}`,
        `Check permissions. System error: ${errorMessage}`
      );
    }

    if (!destDirExists) {
      try {
        fs.mkdirSync(destDir, { recursive: true });
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        return formatError(
          "DEST_DIR_CREATE_ERROR",
          `Cannot create destination directory: ${destDir}`,
          `Check permissions and disk space. System error: ${errorMessage}`
        );
      }
    }

    // === PERFORM MOVE ===

    try {
      fs.renameSync(sourcePath, destPath);
    } catch (err: unknown) {
      const error = err as NodeJS.ErrnoException;
      const errorMessage = error.message || String(error);
      const errorCode = error.code || "UNKNOWN";

      // Cross-device moves are not supported - this prevents moving to external filesystems
      if (error.code === "EXDEV") {
        return formatError(
          "CROSS_DEVICE_NOT_ALLOWED",
          `Cross-filesystem moves are not supported: ${sourcePath} -> ${destPath}`,
          "Source and destination appear to be on different filesystems. " +
            "All moves must stay within the same filesystem. " +
            "Use { trash: true } to move files to .trash/ folder instead."
        );
      }

      return formatError(
        `MOVE_ERROR_${errorCode}`,
        `Failed to move ${sourceType}: ${sourcePath} -> ${destPath}`,
        `System error: ${errorMessage}`
      );
    }

    // === SUCCESS ===

    const relativeSrc: string = path.relative(cwd, sourcePath) || sourcePath;
    const relativeDst: string = path.relative(cwd, destPath) || destPath;

    const result: MoveResult = {
      success: true,
      message: `Moved ${sourceType}: ${relativeSrc} -> ${relativeDst}`,
      details: {
        sourceType,
        sourcePath: relativeSrc,
        destinationPath: relativeDst,
        overwritten: didOverwrite,
        crossDevice: false, // Cross-device moves are no longer supported
      },
    };

    return formatSuccess(result);
  },
});
