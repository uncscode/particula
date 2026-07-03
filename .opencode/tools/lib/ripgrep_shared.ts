import * as fs from "fs";
import * as path from "node:path";

/** Default bounded result count for ripgrep-backed wrappers. */
export const DEFAULT_MAX_RESULTS = 5000;

const MAX_STDOUT_CAPTURE_BYTES = 4 * 1024 * 1024;

export interface SearchParams {
  pattern?: string;
  contentPattern?: string;
  searchPath: string;
  ignoreGitignore?: boolean;
  includeHidden?: boolean;
  unrestricted?: number;
  fileType?: string;
  excludeFileType?: string;
  globCaseInsensitive?: boolean;
  compactOutput?: boolean;
  compactOutputBase?: string;
  filesWithMatches?: boolean;
  filesWithoutMatches?: boolean;
  maxResults?: number;
  maxMatchesPerFile?: number;
  contextLines?: number;
  beforeContext?: number;
  afterContext?: number;
  targetKind?: "file" | "directory";
}

export interface SearchResult {
  files: string[];
  rawLines?: string[];
  exitCode: number;
  errorMessage?: string;
  outputClipped?: boolean;
}

export interface ValidatedSearchPathResult {
  canonicalPath?: string;
  compactOutputBase?: string;
  targetKind?: "file" | "directory";
  error?: string;
}

export function normalizeNumericParam(value: number | undefined): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) return undefined;
  return value;
}

export function validateNonNegativeInt(value: unknown, paramName: string): string | undefined {
  if (value === undefined) return undefined;
  const num = typeof value === "number" ? value : Number(String(value).trim());
  if (!Number.isInteger(num) || num < 0) {
    return `ERROR: Invalid ${paramName} value. It must be a non-negative integer.`;
  }
  return undefined;
}

function isWithinRepository(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));
  return (
    normalizedTarget === normalizedRoot ||
    normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

function isWithinRepositoryRealpath(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));
  return (
    normalizedTarget === normalizedRoot || normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

function isSupportedSearchTarget(stats: fs.Stats): boolean {
  return stats.isFile() || stats.isDirectory();
}

/**
 * Validate a scoped ripgrep target and classify it once for all wrappers.
 *
 * - file input => search only that file
 * - directory input => search only that subtree
 * - missing/invalid/out-of-repo input => deterministic fail-closed error
 */
export async function resolveValidatedSearchPath(
  resolvedSearchPath: string,
  cwd: string,
): Promise<ValidatedSearchPathResult> {
  if (!isWithinRepository(resolvedSearchPath, cwd)) {
    return {
      error: `ERROR: Search path is outside the repository: ${resolvedSearchPath}\n\nHint: All searches must stay within the repository root (${cwd}).`,
    };
  }

  let targetKind: "file" | "directory";
  try {
    const stats = await fs.promises.stat(resolvedSearchPath);
    if (!isSupportedSearchTarget(stats)) {
      return {
        error:
          `ERROR: Unsupported search path type: ${resolvedSearchPath}\n\n` +
          "Hint: Search paths must be regular files or directories.",
      };
    }
    targetKind = stats.isDirectory() ? "directory" : "file";
  } catch {
    return {
      error: `ERROR: Search path does not exist: ${resolvedSearchPath}\n\nHint: Verify the path is correct.`,
    };
  }

  try {
    const [canonicalSearchPath, canonicalRepoRoot] = await Promise.all([
      fs.promises.realpath(resolvedSearchPath),
      fs.promises.realpath(cwd),
    ]);
    if (!isWithinRepositoryRealpath(canonicalSearchPath, canonicalRepoRoot)) {
      return {
        error:
          `ERROR: Search path is outside the repository: ${resolvedSearchPath}\n\n` +
          `Hint: All searches must stay within the repository root (${cwd}).`,
      };
    }

    return {
      canonicalPath: canonicalSearchPath,
      compactOutputBase:
        targetKind === "directory" ? canonicalSearchPath : path.dirname(canonicalSearchPath),
      targetKind,
    };
  } catch {
    return {
      error:
        `ERROR: Unable to resolve canonical search path: ${resolvedSearchPath}\n\n` +
        "Hint: Verify the path exists and is accessible.",
    };
  }
}

function rewriteCompactOutputLine(line: string, compactOutputBase: string): string {
  if (!line || line === "--") return line;
  const match = line.match(/^(.+?)([:\-])(\d+)([:\-])(.*)$/);
  if (!match) {
    return path.relative(compactOutputBase, line) || line;
  }

  const [, filePath, firstSeparator, lineNumber, secondSeparator, remainder] = match;
  const relativePath = path.relative(compactOutputBase, filePath) || filePath;
  return `${relativePath}${firstSeparator}${lineNumber}${secondSeparator}${remainder}`;
}

function applyCompactOutputToLines(lines: string[], compactOutputBase: string | undefined): string[] {
  if (!compactOutputBase) return lines;
  return lines.map((line) => rewriteCompactOutputLine(line, compactOutputBase));
}

function collectBoundedNonEmptyLines(output: string, limit: number): string[] {
  if (!output) return [];

  const lines: string[] = [];
  const boundedLimit = Math.max(1, Math.trunc(limit));
  let start = 0;

  for (let i = 0; i <= output.length; i++) {
    const atEnd = i === output.length;
    const isNewline = !atEnd && output.charCodeAt(i) === 10;
    if (!atEnd && !isNewline) continue;

    const rawLine = output.slice(start, i).replace(/\r$/, "");
    if (rawLine.trim().length > 0) {
      lines.push(rawLine);
      if (lines.length >= boundedLimit) {
        return lines;
      }
    }
    start = i + 1;
  }

  return lines;
}

export function buildTruncationWarning(
  limit: number,
  total: number,
  unit: "files" | "lines",
  options?: { approximateTotal?: boolean },
): string {
  const qualifier = options?.approximateTotal ? `at least ${total}` : `${total}`;
  return `[WARNING: Results truncated to ${limit} ${unit} (${qualifier} total found). Use maxResults parameter to increase limit.]`;
}

export async function executeRipgrepSearch(params: SearchParams): Promise<SearchResult> {
  const {
    pattern,
    contentPattern,
    searchPath,
    ignoreGitignore,
    includeHidden,
    unrestricted,
    fileType,
    excludeFileType,
    globCaseInsensitive,
    compactOutput,
    maxResults,
    maxMatchesPerFile,
    contextLines,
    beforeContext,
    afterContext,
    targetKind,
    filesWithMatches,
    filesWithoutMatches,
    compactOutputBase,
  } = params;

  const isContentSearch = contentPattern !== undefined;
  const cmdArgs: string[] = isContentSearch
    ? ["rg", "-n", "-e", contentPattern]
    : ["rg", "--files"];

  if (isContentSearch && targetKind === "file" && !filesWithMatches && !filesWithoutMatches) {
    cmdArgs.push("--with-filename");
  }

  if (isContentSearch) {
    if (filesWithMatches) {
      cmdArgs.push("-l");
    } else if (filesWithoutMatches) {
      cmdArgs.push("-L");
    }
  }

  if (pattern && pattern.trim()) cmdArgs.push("--glob", pattern);

  if (unrestricted !== undefined) {
    cmdArgs.push("-" + "u".repeat(unrestricted));
  } else {
    if (ignoreGitignore) cmdArgs.push("--no-ignore-vcs");
    if (includeHidden) cmdArgs.push("--hidden");
  }

  if (fileType) cmdArgs.push("-t", fileType);
  if (excludeFileType) cmdArgs.push("-T", excludeFileType);
  if (globCaseInsensitive) cmdArgs.push("--glob-case-insensitive");

  if (isContentSearch) {
    for (const [val, name] of [
      [contextLines, "contextLines"],
      [beforeContext, "beforeContext"],
      [afterContext, "afterContext"],
    ] as const) {
      const err = validateNonNegativeInt(val, name);
      if (err) return { files: [], exitCode: 2, errorMessage: err };
    }
  }

  if (isContentSearch) {
    const maxMatchesError = validateNonNegativeInt(maxMatchesPerFile, "maxMatchesPerFile");
    if (maxMatchesError) return { files: [], exitCode: 2, errorMessage: maxMatchesError };
  }

  const hasContext =
    (contextLines !== undefined && contextLines > 0) ||
    (beforeContext !== undefined && beforeContext > 0) ||
    (afterContext !== undefined && afterContext > 0);
  const effectiveMaxCount = maxMatchesPerFile ?? maxResults;
  const filesOnlyMode = Boolean(filesWithMatches || filesWithoutMatches);

  if (isContentSearch && !filesOnlyMode && effectiveMaxCount !== undefined && !hasContext) {
    cmdArgs.push("--max-count", String(effectiveMaxCount));
  }

  if (isContentSearch) {
    const hasDirectionalContext =
      (beforeContext !== undefined && beforeContext > 0) ||
      (afterContext !== undefined && afterContext > 0);
    if (hasDirectionalContext) {
      if (beforeContext !== undefined && beforeContext > 0) cmdArgs.push("-B", String(beforeContext));
      if (afterContext !== undefined && afterContext > 0) cmdArgs.push("-A", String(afterContext));
    } else if (contextLines !== undefined && contextLines > 0) {
      cmdArgs.push("-C", String(contextLines));
    }
  }

  cmdArgs.push("--", searchPath);

  const lineLimit = isContentSearch
    ? Math.max(1, (maxResults ?? DEFAULT_MAX_RESULTS) + 1)
    : undefined;

  try {
    const subprocess = Bun.spawn(cmdArgs, { stdout: "pipe", stderr: "pipe" } as never);

    const readBoundedStream = async (
      stream: ReadableStream<Uint8Array> | null | undefined,
      options: { maxBytes?: number; maxNonEmptyLines?: number },
    ): Promise<{ text: string; clipped: boolean; nonEmptyLines: number }> => {
      if (!stream) return { text: "", clipped: false, nonEmptyLines: 0 };
      const reader = stream.getReader();
      const chunks: Uint8Array[] = [];
      let totalBytes = 0;
      let clipped = false;
      let pending = "";
      let nonEmptyLines = 0;

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          const chunk = value ?? new Uint8Array();
          let chunkToStore = chunk;
          if (options.maxBytes !== undefined && totalBytes + chunk.length > options.maxBytes) {
            const remaining = Math.max(0, options.maxBytes - totalBytes);
            chunkToStore = chunk.subarray(0, remaining);
            clipped = true;
          }
          if (chunkToStore.length > 0) {
            chunks.push(chunkToStore);
            totalBytes += chunkToStore.length;
            pending += Buffer.from(chunkToStore).toString();
            const segments = pending.split("\n");
            pending = segments.pop() ?? "";
            for (const segment of segments) {
              if (segment.replace(/\r$/, "").trim().length > 0) {
                nonEmptyLines += 1;
                if (options.maxNonEmptyLines !== undefined && nonEmptyLines >= options.maxNonEmptyLines) {
                  clipped = true;
                  break;
                }
              }
            }
          }
          if (
            clipped ||
            (options.maxNonEmptyLines !== undefined && nonEmptyLines >= options.maxNonEmptyLines)
          ) {
            subprocess.kill();
            break;
          }
        }
      } finally {
        reader.releaseLock();
      }

      if (!clipped && pending.replace(/\r$/, "").trim().length > 0) {
        nonEmptyLines += 1;
      }

      return { text: Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString(), clipped, nonEmptyLines };
    };

    const [stdoutResult, stderrResult, exitCodeRaw] = await Promise.all([
      readBoundedStream(subprocess.stdout, {
        maxBytes: MAX_STDOUT_CAPTURE_BYTES,
        maxNonEmptyLines: lineLimit,
      }),
      readBoundedStream(subprocess.stderr, {}),
      subprocess.exited,
    ]);

    const output = stdoutResult.text;
    const exitCode = stdoutResult.clipped ? 0 : Number(exitCodeRaw ?? 0);
    const outputClipped = stdoutResult.clipped;

    if (exitCode === 2) {
      const stderr = stderrResult.text;
      if (stderr.includes("regex parse error") || stderr.includes("invalid")) {
        if (isContentSearch) {
          return {
            files: [],
            exitCode,
            errorMessage: `ERROR: Invalid contentPattern regex: ${contentPattern}\n\nRipgrep error: ${stderr}`,
          };
        }
        return {
          files: [],
          exitCode,
          errorMessage: `ERROR: Invalid glob pattern: ${pattern}\n\nPattern syntax help:\n- * matches any characters except /\n- ** matches any characters including /\n- ? matches any single character\n- [abc] matches a, b, or c\n\nRipgrep error: ${stderr}`,
        };
      }
      return {
        files: [],
        exitCode,
        errorMessage: `ERROR: Ripgrep failed with exit code 2.\n\n${stderr}`,
      };
    }

    if (isContentSearch) {
      const rawLines = applyCompactOutputToLines(
        collectBoundedNonEmptyLines(output, lineLimit ?? Math.max(1, (maxResults ?? DEFAULT_MAX_RESULTS) + 1)),
        compactOutput ? compactOutputBase : undefined,
      );
      const files = filesWithMatches || filesWithoutMatches ? rawLines : [];
      return { files, rawLines, exitCode, outputClipped };
    }

    const files = output
      .trim()
      .split("\n")
      .filter((line) => line.trim().length > 0);

    return { files, exitCode, outputClipped };
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    if (
      errorMessage.includes("ENOENT") ||
      errorMessage.includes("not found") ||
      errorMessage.includes("No such file")
    ) {
      return {
        files: [],
        exitCode: 2,
        errorMessage:
          "ERROR: ripgrep (rg) is not installed or not in PATH.\n\nInstallation:\n- macOS: brew install ripgrep\n- Ubuntu/Debian: apt install ripgrep\n- Windows: choco install ripgrep\n- Rust: cargo install ripgrep",
      };
    }
    return {
      files: [],
      exitCode: 2,
      errorMessage: `ERROR: Failed to execute ripgrep: ${errorMessage}`,
    };
  }
}
