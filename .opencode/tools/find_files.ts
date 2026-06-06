import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "node:path";

// --- Inlined from lib/ripgrep_shared.ts ---

/** Maximum files to process (default). Prevents unbounded I/O for large searches. */
const DEFAULT_MAX_RESULTS = 5000;

/** Batch size for parallel stat operations. Prevents EMFILE errors. */
const STAT_BATCH_SIZE = 100;

/** File entry with modification time for sorting. */
interface FileWithMtime {
  path: string;
  mtime: number;
}

/** Parameters for executing a ripgrep search. */
interface SearchParams {
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
  filesWithMatches?: boolean;
  filesWithoutMatches?: boolean;
  maxResults?: number;
  maxMatchesPerFile?: number;
  contextLines?: number;
  beforeContext?: number;
  afterContext?: number;
}

/** Result from a ripgrep search execution. */
interface SearchResult {
  files: string[];
  rawLines?: string[];
  exitCode: number;
  errorMessage?: string;
  outputClipped?: boolean;
}

const MAX_STDOUT_CAPTURE_BYTES = 4 * 1024 * 1024;

interface ValidatedSearchPathResult {
  canonicalPath?: string;
  error?: string;
}

/**
 * Normalize numeric parameters used by ripgrep wrapper.
 * Only positive integers are treated as meaningful; all other values are omitted.
 */
function normalizeNumericParam(value: number | undefined): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) return undefined;
  return value;
}

/** Validate that a value is a non-negative integer. */
function validateNonNegativeInt(value: unknown, paramName: string): string | undefined {
  if (value === undefined) return undefined;
  const num = typeof value === "number" ? value : Number(String(value).trim());
  if (!Number.isInteger(num) || num < 0) {
    return `ERROR: Invalid ${paramName} value. It must be a non-negative integer.`;
  }
  return undefined;
}

/** Check if a path is within repository using lexical normalized paths. */
function isWithinRepository(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));
  return (
    normalizedTarget === normalizedRoot ||
    normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

/**
 * Check repository containment for already-canonicalized paths.
 */
function isWithinRepositoryRealpath(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));
  return (
    normalizedTarget === normalizedRoot || normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

/**
 * Validate and canonicalize ripgrep search path.
 * Returns canonicalPath on success so callers can execute against validated path.
 */
async function resolveValidatedSearchPath(
  resolvedSearchPath: string,
  cwd: string
): Promise<ValidatedSearchPathResult> {
  if (!isWithinRepository(resolvedSearchPath, cwd)) {
    return {
      error: `ERROR: Search path is outside the repository: ${resolvedSearchPath}\n\nHint: All searches must stay within the repository root (${cwd}).`,
    };
  }

  try {
    const stats = await fs.promises.stat(resolvedSearchPath);
    if (!stats.isDirectory()) {
      return {
        error: `ERROR: Search path is not a directory: ${resolvedSearchPath}\n\nHint: Provide a directory path to search in.`,
      };
    }
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

    return { canonicalPath: canonicalSearchPath };
  } catch {
    return {
      error:
        `ERROR: Unable to resolve canonical search path: ${resolvedSearchPath}\n\n` +
        "Hint: Verify the path exists and is accessible.",
    };
  }
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

/** Get file modification times in parallel with batching. */
async function getFilesWithMtime(files: string[]): Promise<FileWithMtime[]> {
  const results: FileWithMtime[] = [];

  for (let i = 0; i < files.length; i += STAT_BATCH_SIZE) {
    const batch = files.slice(i, i + STAT_BATCH_SIZE);
    const batchResults = await Promise.all(
      batch.map(async (filePath) => {
        try {
          const stats = await fs.promises.stat(filePath);
          return { path: filePath, mtime: stats.mtimeMs };
        } catch {
          return null;
        }
      })
    );

    results.push(...batchResults.filter((r): r is FileWithMtime => r !== null));
  }

  return results;
}

/** Build deterministic truncation warning text. */
function buildTruncationWarning(
  limit: number,
  total: number,
  unit: "files" | "lines",
  options?: { approximateTotal?: boolean }
): string {
  const qualifier = options?.approximateTotal ? `at least ${total}` : `${total}`;
  return `[WARNING: Results truncated to ${limit} ${unit} (${qualifier} total found). Use maxResults parameter to increase limit.]`;
}

/** Execute a ripgrep search with the given parameters. */
async function executeRipgrepSearch(params: SearchParams): Promise<SearchResult> {
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
    maxResults,
    maxMatchesPerFile,
    contextLines,
    beforeContext,
    afterContext,
    filesWithMatches,
    filesWithoutMatches,
  } = params;

  const isContentSearch = contentPattern !== undefined;
  const cmdArgs: string[] = isContentSearch
    ? ["rg", "-n", "-e", contentPattern]
    : ["rg", "--files"];

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

  // End option parsing before positional path operands.
  cmdArgs.push("--", searchPath);

  try {
    const result = Bun.spawnSync(cmdArgs);
    const stdoutBuffer = Buffer.from(result.stdout ?? "");
    const outputClipped = stdoutBuffer.length > MAX_STDOUT_CAPTURE_BYTES;
    const boundedStdout = outputClipped
      ? stdoutBuffer.subarray(0, MAX_STDOUT_CAPTURE_BYTES)
      : stdoutBuffer;
    const output = boundedStdout.toString();
    const exitCode = result.exitCode;

    if (exitCode === 2) {
      const stderr = result.stderr.toString();
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
      const lineLimit = Math.max(1, (maxResults ?? DEFAULT_MAX_RESULTS) + 1);
      const rawLines = collectBoundedNonEmptyLines(output, lineLimit);
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

// --- Tool-local helpers ---

const UNSUPPORTED_PARAMS = [
  "contentPattern",
  "filesWithMatches",
  "filesWithoutMatches",
  "contextLines",
  "beforeContext",
  "afterContext",
  "maxMatchesPerFile",
] as const;

function isMateriallySet(value: unknown): boolean {
  if (value === undefined || value === null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  return true;
}

function normalizeOptionalType(value: string | undefined): { value?: string; error?: string } {
  if (value === undefined || value === null) return {};
  const trimmed = value.trim();
  if (!trimmed) return {};
  if (/\s/.test(trimmed)) {
    return { error: "ERROR: File type values must not contain whitespace." };
  }
  return { value: trimmed.toLowerCase() };
}

// --- Tool definition ---

export default tool({
  description: `Search for files by glob pattern using discovery-only ripgrep mode. Only include parameters you need — omit all others.

SIMPLE EXAMPLES (copy these patterns):

Basic search:      { pattern: "**/*.ts" }
Search in folder:  { pattern: "**/*.py", path: "adw" }
Limit results:     { pattern: "**/*", maxResults: 100 }
Compact output:    { pattern: "**/*.md", path: "docs", compactOutput: true }
File type include: { pattern: "**/*", fileType: "py" }
File type exclude: { pattern: "**/*", excludeFileType: "json" }

RULES:
- Discovery only: content-search parameters are rejected (use search_content for simple content search or ripgrep_advanced for advanced controls).
- Required 'pattern' must be a non-empty string after trim.
- Results are sorted by mtime (most recent first).
- No matches return a deterministic non-error message.
- Searches are constrained to repository boundaries.`,

  args: {
    pattern: tool.schema.string(),
    path: tool.schema.string().optional(),
    fileType: tool.schema.string().optional(),
    excludeFileType: tool.schema.string().optional(),
    globCaseInsensitive: tool.schema.boolean().optional(),
    compactOutput: tool.schema.boolean().optional(),
    maxResults: tool.schema.number().optional(),

    // Explicit unsupported inputs so we can fail closed with deterministic guidance.
    contentPattern: tool.schema.string().optional(),
    filesWithMatches: tool.schema.boolean().optional(),
    filesWithoutMatches: tool.schema.boolean().optional(),
    contextLines: tool.schema.number().optional(),
    beforeContext: tool.schema.number().optional(),
    afterContext: tool.schema.number().optional(),
    maxMatchesPerFile: tool.schema.number().optional(),
  },

  async execute(args) {
    for (const field of UNSUPPORTED_PARAMS) {
      if (isMateriallySet((args as Record<string, unknown>)[field])) {
        return (
          `ERROR: '${field}' is not supported by find_files.\n\n` +
          "Hint: find_files is discovery-only. Use search_content for simple content search or ripgrep_advanced for advanced matching controls."
        );
      }
    }

    const pattern = args.pattern?.trim();
    if (!pattern) {
      return "ERROR: 'pattern' parameter is required and cannot be empty.\n\nHint: Provide a glob pattern like '**/*.ts' or 'src/**/*.py'.";
    }

    const normalizedMaxResults = normalizeNumericParam(args.maxResults);
    const maxResults = normalizedMaxResults ?? DEFAULT_MAX_RESULTS;

    const normalizedFileType = normalizeOptionalType(args.fileType);
    if (normalizedFileType.error) {
      return normalizedFileType.error;
    }
    const normalizedExcludeFileType = normalizeOptionalType(args.excludeFileType);
    if (normalizedExcludeFileType.error) {
      return normalizedExcludeFileType.error;
    }

    const cwd = process.cwd();
    const resolvedSearchPath = args.path
      ? path.isAbsolute(args.path)
        ? path.normalize(args.path)
        : path.resolve(cwd, args.path)
      : cwd;

    const validatedPath = await resolveValidatedSearchPath(resolvedSearchPath, cwd);
    if (validatedPath.error) return validatedPath.error;
    const executedSearchPath = validatedPath.canonicalPath ?? resolvedSearchPath;

    const searchResult = await executeRipgrepSearch({
      pattern,
      searchPath: executedSearchPath,
      fileType: normalizedFileType.value,
      excludeFileType: normalizedExcludeFileType.value,
      globCaseInsensitive: args.globCaseInsensitive,
      maxResults,
      compactOutput: args.compactOutput,
    });

    if (searchResult.errorMessage) {
      return searchResult.errorMessage;
    }

    if (searchResult.files.length === 0) {
      const searchContext = args.path ? ` in '${args.path}'` : "";
      return `No files found matching pattern '${pattern}'${searchContext}.`;
    }

    const wasTruncated = searchResult.files.length > maxResults;
    const filesToProcess = searchResult.files.slice(0, maxResults);
    const filesWithMtime = await getFilesWithMtime(filesToProcess);
    if (filesToProcess.length > 0 && filesWithMtime.length === 0) {
      return "ERROR: Failed to read metadata for matched files.\n\nHint: Verify file permissions and path accessibility.";
    }
    filesWithMtime.sort((a, b) => b.mtime - a.mtime);

    const basePath = args.compactOutput ? resolvedSearchPath : cwd;
    let output = "";
    for (let i = 0; i < filesWithMtime.length; i++) {
      const filePath = filesWithMtime[i]?.path;
      if (!filePath) continue;
      const relativePath = path.relative(basePath, filePath) || filePath;
      output += i === 0 ? relativePath : `\n${relativePath}`;
    }

    if (wasTruncated) {
      output += `\n\n${buildTruncationWarning(maxResults, searchResult.files.length, "files")}`;
    }
    return output;
  },
});
