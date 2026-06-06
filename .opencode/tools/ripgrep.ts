/**
 * RipgrepTool - File and content search via ripgrep.
 *
 * Two modes: file discovery (`pattern` only) and content search (`contentPattern`).
 * Auto-retries file discovery with relaxed ignore/hidden flags when initial search
 * finds nothing. All searches are confined to the repository root.
 *
 * No-op normalization: empty strings, 0, false, negative numbers, and NaN are
 * silently treated as "not provided" for all optional parameters. Callers never
 * need to worry about accidentally passing these values.
 *
 * Performance: maxResults caps output (default 5000), stat() calls are batched
 * (100/batch), and empty results exit early.
 */

import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "path";

/** Maximum files to process (default). Prevents unbounded I/O for large searches. */
const DEFAULT_MAX_RESULTS = 5000;

/** Batch size for parallel stat operations. Prevents EMFILE errors. */
const STAT_BATCH_SIZE = 100;

/** Warning shown when context parameters are used without contentPattern. */
const CONTEXT_PARAMS_IGNORED_WARNING =
  "[WARNING: contextLines/beforeContext/afterContext are ignored unless contentPattern is set.]";

const FALLBACK_TYPE_ALIASES = new Set([
  "ts",
  "tsx",
  "js",
  "jsx",
  "py",
  "json",
  "md",
  "yml",
  "yaml",
]);

const FALLBACK_TYPE_LIST_WARNING =
  "[WARNING: Unable to load ripgrep type list; using built-in fallback allowlist: ts, tsx, js, jsx, py, json, md, yml, yaml.]";

let cachedTypeAliases: Set<string> | null = null;
let cachedTypeListWarning: string | null = null;

function normalizeFileTypeParam(
  value: string | undefined,
  paramName: "fileType" | "excludeFileType"
): { normalized?: string; error?: string } {
  if (value === undefined) return {};
  if (typeof value !== "string") {
    return { error: `ERROR: '${paramName}' must be a string.` };
  }
  const trimmed = value.trim();
  if (!trimmed) {
    // Treat empty/whitespace-only strings as "not provided" (sparse default)
    return {};
  }
  if (/\s/.test(trimmed)) {
    return { error: `ERROR: '${paramName}' must not contain whitespace.` };
  }
  return { normalized: trimmed.toLowerCase() };
}

function getRipgrepTypeAliases(): { aliases: Set<string>; warning?: string } {
  if (cachedTypeAliases) {
    return { aliases: cachedTypeAliases, warning: cachedTypeListWarning ?? undefined };
  }

  if (cachedTypeListWarning) {
    return { aliases: FALLBACK_TYPE_ALIASES, warning: cachedTypeListWarning };
  }

  try {
    const result = Bun.spawnSync(["rg", "--type-list"]);
    if (result.exitCode !== 0) {
      const stderr = result.stderr.toString().trim();
      cachedTypeListWarning = stderr
        ? `${FALLBACK_TYPE_LIST_WARNING}\n\nRipgrep error: ${stderr}`
        : FALLBACK_TYPE_LIST_WARNING;
      return { aliases: FALLBACK_TYPE_ALIASES, warning: cachedTypeListWarning };
    }

    const lines = result.stdout.toString().split("\n");
    const aliases = new Set<string>();
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const alias = trimmed.split(":")[0]?.trim();
      if (alias) {
        aliases.add(alias.toLowerCase());
      }
    }

    if (aliases.size === 0) {
      cachedTypeListWarning = FALLBACK_TYPE_LIST_WARNING;
      return { aliases: FALLBACK_TYPE_ALIASES, warning: cachedTypeListWarning };
    }

    cachedTypeAliases = aliases;
    return { aliases };
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    cachedTypeListWarning = `${FALLBACK_TYPE_LIST_WARNING}\n\nRipgrep error: ${errorMessage}`;
    return { aliases: FALLBACK_TYPE_ALIASES, warning: cachedTypeListWarning };
  }
}

const isTestEnvironment =
  process.env.NODE_ENV === "test" || process.env.OPENCODE_ENV === "test";
if (isTestEnvironment) {
  (globalThis as any).__OPENCODE_RIPGREP_TEST_HOOKS__ = {
    resetTypeAliasCache() {
      cachedTypeAliases = null;
      cachedTypeListWarning = null;
    },
  };
}

/** File entry with modification time for sorting */
interface FileWithMtime {
  path: string;
  mtime: number;
}

/**
 * Parameters for executing a ripgrep search.
 * Used by the executeRipgrepSearch helper to enable reusable search logic.
 */
interface SearchParams {
  /** Glob pattern to match files (file discovery) or filter files (content search). */
  pattern?: string;
  /** Content regex pattern to search within files (content search mode). */
  contentPattern?: string;
  /** Resolved absolute path to search directory */
  searchPath: string;
  /** Skip .gitignore rules */
  ignoreGitignore?: boolean;
  /** Include hidden files/directories */
  includeHidden?: boolean;
  /** Unrestricted search level (1-3) */
  unrestricted?: number;
  /** Include only files of this type */
  fileType?: string;
  /** Exclude files of this type */
  excludeFileType?: string;
  /** Enable case-insensitive glob matching */
  globCaseInsensitive?: boolean;
  /** Format output paths relative to the search directory */
  compactOutput?: boolean;
  /** Return only files that contain matches (content search only). */
  filesWithMatches?: boolean;
  /** Return only files that do not contain matches (content search only). */
  filesWithoutMatches?: boolean;
  /** Maximum number of results (files or content lines depending on mode). */
  maxResults?: number;
  /** Maximum number of matches per file (content search only). */
  maxMatchesPerFile?: number;
  /** Context lines before and after each match (content search only). */
  contextLines?: number;

  /** Context lines before each match (content search only). */
  beforeContext?: number;
  /** Context lines after each match (content search only). */
  afterContext?: number;
}

/**
 * Result from a ripgrep search execution.
 * Contains raw file list and status information for retry logic.
 */
interface SearchResult {
  /** List of file paths found by ripgrep */
  files: string[];
  /** Raw match lines for content search mode */
  rawLines?: string[];
  /** Ripgrep exit code (0=matches, 1=no matches, 2=error) */
  exitCode: number;
  /** Error message if search failed */
  errorMessage?: string;
}

/**
 * Normalize a numeric parameter: treat 0, negative, NaN, and non-integer values
 * as "not provided" (undefined). This prevents LLM-supplied default values like 0
 * from silently breaking searches (e.g., maxResults: 0 returning nothing).
 *
 * Only positive integers are meaningful for all numeric ripgrep parameters.
 */
function normalizeNumericParam(value: number | undefined): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) return undefined;
  return value;
}

/**
 * Validate that a value is a non-negative integer.
 * Returns an error message string if invalid, or undefined if valid.
 */
function validateNonNegativeInt(value: unknown, paramName: string): string | undefined {
  if (value === undefined) return undefined;
  const num = typeof value === "number" ? value : Number(String(value).trim());
  if (!Number.isInteger(num) || num < 0) {
    return `ERROR: Invalid ${paramName} value. It must be a non-negative integer.`;
  }
  return undefined;
}

/**
 * Check if a path is within the repository boundary.
 * Prevents searching outside the current working directory.
 */
function isWithinRepository(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));

  return (
    normalizedTarget === normalizedRoot ||
    normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

/**
 * Check repository boundary using canonical realpaths.
 *
 * This guards against symlink traversal where lexical path checks alone could
 * incorrectly treat an out-of-repo target as safe.
 */
function isWithinRepositoryRealpath(targetPath: string, repoRoot: string): boolean {
  const normalizedTarget = path.normalize(path.resolve(targetPath));
  const normalizedRoot = path.normalize(path.resolve(repoRoot));
  return (
    normalizedTarget === normalizedRoot || normalizedTarget.startsWith(normalizedRoot + path.sep)
  );
}

/**
 * Get file modification times in parallel with batching.
 * Processes files in batches to avoid EMFILE (too many open files).
 * Gracefully handles deleted files (returns null, filtered out).
 */
async function getFilesWithMtime(files: string[]): Promise<FileWithMtime[]> {
  const results: FileWithMtime[] = [];

  // Process in batches to avoid EMFILE (too many open files)
  for (let i = 0; i < files.length; i += STAT_BATCH_SIZE) {
    const batch = files.slice(i, i + STAT_BATCH_SIZE);
    const batchResults = await Promise.all(
      batch.map(async (filePath) => {
        try {
          const stats = await fs.promises.stat(filePath);
          return { path: filePath, mtime: stats.mtimeMs };
        } catch {
          // File may have been deleted between rg output and stat
          return null;
        }
      })
    );
    results.push(...batchResults.filter((r): r is FileWithMtime => r !== null));
  }

  return results;
}

/**
 * Execute a ripgrep search with the given parameters.
 *
 * Builds and runs the ripgrep command, returning raw results that the caller
 * can use to decide whether to retry with different parameters.
 *
 * @param params - Search parameters including pattern, path, and ignore flags
 * @returns SearchResult with files array, exit code, and optional error message
 */
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

  // Build ripgrep command
  const cmdArgs: string[] = isContentSearch
    ? ["rg", "-n", contentPattern]
    : ["rg", "--files"];

  if (isContentSearch) {
    if (filesWithMatches) {
      cmdArgs.push("-l");
    } else if (filesWithoutMatches) {
      cmdArgs.push("-L");
    }
  }

  // Add glob pattern for file discovery or content search filtering
  if (pattern && pattern.trim()) {
    cmdArgs.push("--glob", pattern);
  }

  // Handle unrestricted mode (takes precedence)
  if (unrestricted !== undefined) {
    // Add the appropriate -u flags
    cmdArgs.push("-" + "u".repeat(unrestricted));
  } else {
    // Handle individual flags
    if (ignoreGitignore) {
      cmdArgs.push("--no-ignore-vcs");
    }
    if (includeHidden) {
      cmdArgs.push("--hidden");
    }
  }

  // File type filtering
  if (fileType) {
    cmdArgs.push("-t", fileType);
  }
  if (excludeFileType) {
    cmdArgs.push("-T", excludeFileType);
  }
  if (globCaseInsensitive) {
    cmdArgs.push("--glob-case-insensitive");
  }

  if (isContentSearch) {
    // Validate context parameters are non-negative integers
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
    if (maxMatchesError) {
      return { files: [], exitCode: 2, errorMessage: maxMatchesError };
    }
  }

  // --max-count limits matches per file; only apply when there is no context
  // (context lines inflate the output count well beyond the match count).
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
      if (beforeContext !== undefined && beforeContext > 0) {
        cmdArgs.push("-B", String(beforeContext));
      }
      if (afterContext !== undefined && afterContext > 0) {
        cmdArgs.push("-A", String(afterContext));
      }
    } else if (contextLines !== undefined && contextLines > 0) {
      cmdArgs.push("-C", String(contextLines));
    }
  }

  // Add search path
  cmdArgs.push(searchPath);

  // Execute ripgrep
  try {
    const result = Bun.spawnSync(cmdArgs);
    const output = result.stdout.toString();
    const exitCode = result.exitCode;

    // Handle ripgrep exit codes
    // 0 = matches found
    // 1 = no matches (not an error)
    // 2 = error (invalid pattern, permission denied, etc.)
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
      const rawLines = output
        .trim()
        .split("\n")
        .filter((line) => line.trim().length > 0);
      const files = filesWithMatches || filesWithoutMatches ? rawLines : [];
      return { files, rawLines, exitCode };
    }

    // Parse output into file list
    const files = output
      .trim()
      .split("\n")
      .filter((line) => line.trim().length > 0);

    return { files, exitCode };
  } catch (err: unknown) {
    // Handle case where rg is not installed
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

export default tool({
  description: `Search for files and content using ripgrep. Only include parameters you need — omit all others.

SIMPLE EXAMPLES (copy these patterns):

  Find files:        { pattern: "**/*.ts" }
  Find files in dir: { pattern: "**/*.py", path: "src" }
  Search content:    { contentPattern: "TODO" }
  Content in dir:    { contentPattern: "import os", path: "adw" }
  Content + filter:  { pattern: "**/*.py", contentPattern: "import" }
  File type filter:  { contentPattern: "TODO", fileType: "py" }
  Files with match:  { contentPattern: "TODO", filesWithMatches: true }
  Search hidden/ignored: { pattern: "**/*.js", unrestricted: 2 }
  With context:      { contentPattern: "TODO", contextLines: 2 }

RULES:
- File discovery mode: set 'pattern' only. Returns file paths sorted by mtime.
- Content search mode: set 'contentPattern'. Returns file:line:content matches.
- Omit optional params entirely — do NOT pass empty strings or false/0 values.
- No-op safety: empty strings, 0, false, and negative numbers are silently ignored for all optional params.
- Auto-retry with ignore/hidden is automatic when file discovery finds nothing.

See .opencode/tools/ripgrep.md for full parameter reference and advanced usage.`,

  args: {
    pattern: tool.schema
      .string()
      .optional()
      .describe(`Glob pattern for file matching (e.g., '**/*.ts', 'src/**/*.js').
Required for file discovery; optional glob filter for content search.
Syntax: * (any non-/), ** (any including /), ? (single char), [abc] (set).`),

    contentPattern: tool.schema
      .string()
      .optional()
      .describe(`Regex pattern to search inside files. Triggers content search mode.
Output: file:line_number:content. Combine with 'pattern' to filter files first.`),

    filesWithMatches: tool.schema
      .boolean()
      .optional()
      .describe(`Return only file paths with matches (-l). Content search only. Mutually exclusive with filesWithoutMatches.`),

    filesWithoutMatches: tool.schema
      .boolean()
      .optional()
      .describe(`Return only file paths without matches (-L). Content search only. Mutually exclusive with filesWithMatches.`),

    path: tool.schema
      .string()
      .optional()
      .describe(`Directory to search in (default: cwd). Must be within the repository.`),

    ignoreGitignore: tool.schema
      .boolean()
      .optional()
      .describe(`Skip .gitignore rules (--no-ignore-vcs). Implied by unrestricted >= 1.`),

    includeHidden: tool.schema
      .boolean()
      .optional()
      .describe(`Include hidden files/dirs (--hidden). Implied by unrestricted >= 2.`),

    unrestricted: tool.schema
      .number()
      .optional()
      .describe(`Unrestricted search level (1-3). Overrides ignoreGitignore/includeHidden.
1: ignore .gitignore, 2: +hidden files, 3: +binary files.`),

    fileType: tool.schema
      .string()
      .optional()
      .describe(`Include only files of this type (-t TYPE). Examples: ts, js, py, json, md.`),

    excludeFileType: tool.schema
      .string()
      .optional()
      .describe(`Exclude files of this type (-T TYPE). Examples: ts, js, py, json, md.`),

    globCaseInsensitive: tool.schema
      .boolean()
      .optional()
      .describe(`Case-insensitive glob matching (--glob-case-insensitive).`),

    compactOutput: tool.schema
      .boolean()
      .optional()
      .describe(`Paths relative to search dir instead of cwd. Useful for deep directory searches.`),

    maxResults: tool.schema
      .number()
      .optional()
      .describe(`Max results returned (default: 5000). Caps files or content lines.
Overridden by maxMatchesPerFile for per-file --max-count in content search.`),

    maxMatchesPerFile: tool.schema
      .number()
      .optional()
      .describe(`Max matches per file in content search (--max-count). Overrides maxResults for this flag.
Ignored in file discovery, files-with-matches mode, and when context flags are active.`),

    contextLines: tool.schema
      .number()
      .optional()
      .describe(`Lines before and after each match (-C N). Content search only.
Overridden by beforeContext/afterContext when those are > 0.`),

    beforeContext: tool.schema
      .number()
      .optional()
      .describe(`Lines before each match (-B N). Content search only. Overrides contextLines.`),

    afterContext: tool.schema
      .number()
      .optional()
      .describe(`Lines after each match (-A N). Content search only. Overrides contextLines.`),
  },

  async execute(args): Promise<string> {
    const {
      pattern,
      contentPattern,
      path: searchPath,
      ignoreGitignore,
      includeHidden,
      unrestricted,
      fileType,
      excludeFileType,
      globCaseInsensitive,
      maxResults: rawMaxResults,
      maxMatchesPerFile: rawMaxMatchesPerFile,
      contextLines: rawContextLines,
      beforeContext: rawBeforeContext,
      afterContext: rawAfterContext,
      filesWithMatches,
      filesWithoutMatches,
      compactOutput = false,
    } = args;

    // Normalize numeric params: treat 0/negative/NaN as "not provided" (undefined).
    // This prevents LLM-supplied defaults like 0 from silently breaking searches
    // (e.g., maxResults: 0 → zero results returned instead of the 5000 default).
    const maxResults = normalizeNumericParam(rawMaxResults);
    const maxMatchesPerFile = normalizeNumericParam(rawMaxMatchesPerFile);
    const contextLines = normalizeNumericParam(rawContextLines);
    const beforeContext = normalizeNumericParam(rawBeforeContext);
    const afterContext = normalizeNumericParam(rawAfterContext);

    // Normalize unrestricted: treat 0/negative/NaN as "not provided",
    // consistent with all other numeric params above.
    const normalizedUnrestricted = normalizeNumericParam(unrestricted);

    // Normalize mode-driving optional strings:
    // - exact-empty string ("") => omitted (undefined)
    // - whitespace-only strings are preserved for explicit validation errors
    const normalizedPattern = pattern === "" ? undefined : pattern;
    const normalizedContentPattern = contentPattern === "" ? undefined : contentPattern;

    const cwd = process.cwd();
    const effectiveMaxResults = maxResults ?? DEFAULT_MAX_RESULTS;
    const isContentSearch = normalizedContentPattern !== undefined;
    const isFilesMode = Boolean(filesWithMatches || filesWithoutMatches);
    const contentPatternWarning =
      "[WARNING: filesWithMatches/filesWithoutMatches are ignored unless contentPattern is set.]";
    const shouldWarnAboutContentPattern = !isContentSearch && isFilesMode;

    const appendFilesModeWarning = (message: string) =>
      shouldWarnAboutContentPattern ? `${message}\n\n${contentPatternWarning}` : message;

    let typeListWarning: string | undefined;
    const appendTypeListWarning = (message: string) =>
      typeListWarning ? `${message}\n\n${typeListWarning}` : message;

    // === VALIDATION ===

    if (isContentSearch && (!normalizedContentPattern || !normalizedContentPattern.trim())) {
      return appendTypeListWarning("ERROR: 'contentPattern' cannot be empty.");
    }

    const { normalized: normalizedFileType, error: fileTypeError } = normalizeFileTypeParam(
      fileType,
      "fileType"
    );
    if (fileTypeError) {
      return appendTypeListWarning(fileTypeError);
    }

    const { normalized: normalizedExcludeFileType, error: excludeFileTypeError } =
      normalizeFileTypeParam(excludeFileType, "excludeFileType");
    if (excludeFileTypeError) {
      return appendTypeListWarning(excludeFileTypeError);
    }

    if (normalizedFileType || normalizedExcludeFileType) {
      const { aliases, warning } = getRipgrepTypeAliases();
      typeListWarning = warning;
      if (normalizedFileType && !aliases.has(normalizedFileType)) {
        return appendTypeListWarning(
          `ERROR: Unsupported fileType '${normalizedFileType}'.\n\nHint: Run 'rg --type-list' to view supported types.`
        );
      }

      if (normalizedExcludeFileType && !aliases.has(normalizedExcludeFileType)) {
        return appendTypeListWarning(
          `ERROR: Unsupported excludeFileType '${normalizedExcludeFileType}'.\n\nHint: Run 'rg --type-list' to view supported types.`
        );
      }
    }

    if (isFilesMode && !isContentSearch) {
      return appendTypeListWarning(
        `${contentPatternWarning}\n\nHint: Set contentPattern when using filesWithMatches/filesWithoutMatches.`
      );
    }

    if (filesWithMatches && filesWithoutMatches) {
      const errorMessage = "ERROR: 'filesWithMatches' and 'filesWithoutMatches' cannot both be true.";
      return appendFilesModeWarning(appendTypeListWarning(errorMessage));
    }

    // Validate pattern (required for file discovery)
    if (
      !isContentSearch &&
      (!normalizedPattern || typeof normalizedPattern !== "string" || !normalizedPattern.trim())
    ) {
      return appendFilesModeWarning(
        appendTypeListWarning(
          "ERROR: 'pattern' parameter is required and cannot be empty.\n\nHint: Provide a glob pattern like '**/*.ts' or 'src/**/*.js'"
        )
      );
    }

    // Validate and resolve search path
    const resolvedSearchPath = searchPath
      ? path.isAbsolute(searchPath)
        ? path.normalize(searchPath)
        : path.resolve(cwd, searchPath)
      : cwd;

    // Security: ensure search path is within repository (lexical fast-path)
    if (!isWithinRepository(resolvedSearchPath, cwd)) {
      const errorMessage = `ERROR: Search path is outside the repository: ${resolvedSearchPath}\n\nHint: All searches must stay within the repository root (${cwd}).`;
      return appendFilesModeWarning(appendTypeListWarning(errorMessage));
    }

    // Check if search path exists
    try {
      const stats = await fs.promises.stat(resolvedSearchPath);
      if (!stats.isDirectory()) {
        const errorMessage = `ERROR: Search path is not a directory: ${resolvedSearchPath}\n\nHint: Provide a directory path to search in.`;
        return appendFilesModeWarning(appendTypeListWarning(errorMessage));
      }
    } catch {
      const errorMessage = `ERROR: Search path does not exist: ${resolvedSearchPath}\n\nHint: Verify the path is correct.`;
      return appendFilesModeWarning(appendTypeListWarning(errorMessage));
    }

    // Security: enforce canonical boundary checks after existence validation.
    // This blocks symlink-escape paths that may pass lexical prefix checks.
    try {
      const [canonicalSearchPath, canonicalRepoRoot] = await Promise.all([
        fs.promises.realpath(resolvedSearchPath),
        fs.promises.realpath(cwd),
      ]);
      if (!isWithinRepositoryRealpath(canonicalSearchPath, canonicalRepoRoot)) {
        const errorMessage =
          `ERROR: Search path is outside the repository: ${resolvedSearchPath}\n\n` +
          `Hint: All searches must stay within the repository root (${cwd}).`;
        return appendFilesModeWarning(appendTypeListWarning(errorMessage));
      }
    } catch {
      const errorMessage =
        `ERROR: Unable to resolve canonical search path: ${resolvedSearchPath}\n\n` +
        "Hint: Verify the path exists and is accessible.";
      return appendFilesModeWarning(appendTypeListWarning(errorMessage));
    }

    // Validate unrestricted level (after normalization, so 0/negative are already undefined)
    if (normalizedUnrestricted !== undefined && normalizedUnrestricted > 3) {
      return appendTypeListWarning(
        "ERROR: 'unrestricted' must be 1, 2, or 3.\n\nLevels:\n- 1: Ignore .gitignore\n- 2: Also include hidden files\n- 3: Also include binary files"
      );
    }

    // === EXECUTE INITIAL SEARCH ===

    // Build search params from args
    const searchParams: SearchParams = {
      pattern: normalizedPattern,
      contentPattern: normalizedContentPattern,
      searchPath: resolvedSearchPath,
      ignoreGitignore,
      includeHidden,
      unrestricted: normalizedUnrestricted,
      fileType: normalizedFileType,
      excludeFileType: normalizedExcludeFileType,
      globCaseInsensitive,
      compactOutput,
      maxResults,
      maxMatchesPerFile,
      contextLines,
      beforeContext,
      afterContext,
      filesWithMatches,
      filesWithoutMatches,
    };

    // Execute initial search
    const initialResult = await executeRipgrepSearch(searchParams);

    // Handle errors from initial search
    if (initialResult.errorMessage) {
      return appendTypeListWarning(initialResult.errorMessage);
    }

    if (isContentSearch) {
      const pathOnlyMode = Boolean(filesWithMatches || filesWithoutMatches);
      const rawLines = initialResult.rawLines ?? [];
      if (rawLines.length === 0) {
        if (filesWithoutMatches) {
          return appendTypeListWarning(
            `No files found without matches for contentPattern '${normalizedContentPattern}'.`
          );
        }
        if (filesWithMatches) {
          return appendTypeListWarning(
            `No files found with matches for contentPattern '${normalizedContentPattern}'.`
          );
        }
        return appendTypeListWarning(
          `No matches found for contentPattern '${normalizedContentPattern}'.`
        );
      }
      const wasTruncated = rawLines.length > effectiveMaxResults;
      const linesToReturn = rawLines.slice(0, effectiveMaxResults);
      const basePath = compactOutput ? resolvedSearchPath : cwd;
      let result = pathOnlyMode
        ? linesToReturn
            .map((line) => {
              const relativePath = path.relative(basePath, line) || line;
              return relativePath;
            })
            .join("\n")
        : linesToReturn.join("\n");
      if (wasTruncated) {
        const unit = pathOnlyMode ? "files" : "lines";
        result += `\n\n[WARNING: Results truncated to ${effectiveMaxResults} ${unit} (${rawLines.length} total found). Use maxResults parameter to increase limit.]`;
      }
      if (shouldWarnAboutContentPattern) {
        result += `\n\n${contentPatternWarning}`;
      }
      return appendTypeListWarning(result);
    }

    const hasContextParams =
      (contextLines !== undefined && contextLines > 0) ||
      (beforeContext !== undefined && beforeContext > 0) ||
      (afterContext !== undefined && afterContext > 0);

    let files = initialResult.files;
    let didAutoRetry = false;


    // === AUTO-RETRY LOGIC ===

    // Check if auto-retry should trigger:
    // - Initial search returned 0 results
    // - User didn't explicitly set ignoreGitignore
    // - User didn't use unrestricted mode
    const shouldAutoRetry =
      files.length === 0 &&
      !ignoreGitignore && // User didn't explicitly enable
      !normalizedUnrestricted; // User didn't use unrestricted mode

    if (shouldAutoRetry) {
      // Retry with expanded search parameters
      const retryParams: SearchParams = {
        ...searchParams,
        ignoreGitignore: true,
        includeHidden: true,
      };

      const retryResult = await executeRipgrepSearch(retryParams);

      // Handle errors from retry search
    if (retryResult.errorMessage) {
      return appendTypeListWarning(retryResult.errorMessage);
    }

      // Use retry results if found
      if (retryResult.files.length > 0) {
        files = retryResult.files;
        didAutoRetry = true;
      } else {
        // Both searches returned nothing
        const searchContext = searchPath ? ` in '${searchPath}'` : "";
        let result =
          `No files found matching pattern '${normalizedPattern}'${searchContext}.\n\n` +
          "Note: Auto-retry was attempted with ignoreGitignore=true, includeHidden=true but still found no results.";
        if (hasContextParams) {
          result += `\n\n${CONTEXT_PARAMS_IGNORED_WARNING}`;
        }
        return appendTypeListWarning(result);
      }
    }

    // === PROCESS RESULTS ===

    // EARLY EXIT: No files found (and no auto-retry occurred)
    if (files.length === 0) {
      const searchContext = searchPath ? ` in '${searchPath}'` : "";
      let result = `No files found matching pattern '${pattern}'${searchContext}.`;
      if (hasContextParams) {
        result += `\n\n${CONTEXT_PARAMS_IGNORED_WARNING}`;
      }
      return appendTypeListWarning(result);
    }

    // Track if we truncated results
    const wasTruncated = files.length > effectiveMaxResults;
    const filesToProcess = files.slice(0, effectiveMaxResults);

    // Get file modification times with async parallel batching
    const filesWithMtime = await getFilesWithMtime(filesToProcess);

    // Sort by mtime (most recently modified first)
    filesWithMtime.sort((a, b) => b.mtime - a.mtime);

    // Format output as newline-separated paths
    const basePath = compactOutput ? resolvedSearchPath : cwd;
    const resultPaths = filesWithMtime.map((f) => path.relative(basePath, f.path) || f.path);

    // Build output with optional auto-retry indicator
    let result = "";

    if (didAutoRetry) {
      result += `⚠️ Auto-retry activated
   Reason: Initial search returned no results
   Original: ignoreGitignore=false, includeHidden=${includeHidden ?? false}
   Retry:    ignoreGitignore=true, includeHidden=true

`;
    }

    result += resultPaths.join("\n");

    // Add truncation warning if needed
    if (wasTruncated) {
      result += `\n\n[WARNING: Results truncated to ${effectiveMaxResults} files (${files.length} total found). Use maxResults parameter to increase limit.]`;
    }

    if (hasContextParams) {
      result += `\n\n${CONTEXT_PARAMS_IGNORED_WARNING}`;
    }

    if (shouldWarnAboutContentPattern) {
      result += `\n\n${contentPatternWarning}`;
    }

    return appendTypeListWarning(result);
  },
});
