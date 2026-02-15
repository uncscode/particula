/**
 * RipgrepTool - File Search with Ignore-Control Options and Auto-Retry
 *
 * Searches for files using ripgrep with extended ignore-control parameters.
 * Enables searches across all files regardless of .gitignore rules, solving
 * the common need to search in ignored directories like node_modules/, build/,
 * dist/, or .cache/.
 *
 * AUTO-RETRY FEATURE (F19-P3):
 * When an initial search returns zero results and the user hasn't explicitly
 * set ignore flags, the tool automatically retries with `ignoreGitignore: true`
 * and `includeHidden: true`. This eliminates the common manual retry workflow.
 * Auto-retry conditions:
 * - Initial search returned 0 results
 * - User did NOT explicitly set `ignoreGitignore: true`
 * - User did NOT use `unrestricted` mode
 * Output includes a clear indicator when auto-retry was activated.
 *
 * SECURITY: This tool enforces repository boundary restrictions using the
 * same isWithinRepository() pattern as move.ts. All searches must stay
 * within the current working directory.
 *
 * PERFORMANCE: Large searches are optimized with:
  * - maxResults parameter (default 5000) to bound I/O operations
  * - maxMatchesPerFile (content search) takes precedence over maxResults for --max-count

 * - Async parallel stat() calls with batching (100 files per batch)
 * - Early exit for empty results (skip stat loop entirely)
 *
 * @see adw-docs/dev-plans/features/F19-ripgrep-tool.md - Feature specification
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
  description: `Search for files using ripgrep with extended ignore-control options plus file type filters and glob case-insensitive matching.

This tool provides fine-grained control over which files are searched, including
 the ability to search in .gitignore-ignored directories like node_modules/ and to constrain results by ripgrep file types.

MODE SELECTION:
- File discovery mode (default): set 'pattern' only. Uses "rg --files --glob <pattern>" and sorts results by mtime.
- Content search mode: set 'contentPattern'. Uses "rg -n <contentPattern>" and returns raw match lines.
  Optionally combine with 'pattern' as a glob filter (adds --glob <pattern>).
- Files-with-matches mode: set contentPattern + filesWithMatches/filesWithoutMatches to return paths only (maps to -l/-L).

Quick guide:
- Use file discovery when you only need file paths (pattern required).
- Use content search when you need matching lines (contentPattern required).
- Use files-with-matches when you want only matching file paths (contentPattern + filesWithMatches/filesWithoutMatches).

Results are sorted by modification time (most recently modified first) in file discovery mode.
Content search results are returned in ripgrep order (file:line:content). Files-with-matches returns paths only.

AUTO-RETRY BEHAVIOR:
When a file discovery search returns no results and ignore flags weren't explicitly set, the tool
automatically retries with ignoreGitignore=true and includeHidden=true. This eliminates
the common "search returned nothing, try again with ignore disabled" workflow.
- Auto-retry triggers when: 0 results AND ignoreGitignore not set AND unrestricted not set
- Auto-retry does NOT trigger when: user explicitly set ignoreGitignore=true or unrestricted
- Output includes a clear indicator when auto-retry occurred
- Auto-retry does NOT run for content search mode

USAGE EXAMPLES:
- Basic search: { pattern: "**/*.ts" }
- Search ignored dirs: { pattern: "**/*.js", ignoreGitignore: true }
- Include hidden: { pattern: "**/.*rc", includeHidden: true }
- Full unrestricted: { pattern: "**/*", unrestricted: 3 }
- Limit results: { pattern: "**/*", unrestricted: 1, maxResults: 1000 }
- Include file type: { pattern: "**/*", fileType: "ts" }
- Exclude file type: { pattern: "**/*", excludeFileType: "json" }
- Case-insensitive glob: { pattern: "**/readme.md", globCaseInsensitive: true }
- Combine ignore control and file type: { pattern: "**/*", fileType: "js", ignoreGitignore: true }
- Compact output relative to search path: { pattern: "**/*.ts", path: "src/deep/nested", compactOutput: true }
- Basic content search: { contentPattern: "TODO" }
- Content search with glob filter: { pattern: "**/*.py", contentPattern: "import" }
- Content search regex: { contentPattern: "function\\s+\\w+" }
- Content search with word boundaries: { contentPattern: "\\bTODO\\b" }
- Content search with fixed string (escaped): { contentPattern: "console\\.log" }
- Content search with context: { contentPattern: "TODO", contextLines: 2 }
- Content search with directional context: { contentPattern: "TODO", beforeContext: 2, afterContext: 1 }
- Files with matches only: { contentPattern: "TODO", filesWithMatches: true }
- Files without matches only: { contentPattern: "TODO", filesWithoutMatches: true }
- Content search + glob + file type + compact output: { pattern: "**/*.ts", contentPattern: "TODO", fileType: "ts", compactOutput: true }
- Glob-case-insensitive filtering: { pattern: "**/readme.md", contentPattern: "TODO", globCaseInsensitive: true }

MIGRATION FROM GLOB/GREP:
- GlobTool file discovery: glob({ pattern: "**/*.ts", path: "src" })
  → ripgrep({ pattern: "**/*.ts", path: "src" })
- GrepTool content search: grep({ pattern: "TODO", include: "*.py" })
  → ripgrep({ contentPattern: "TODO", fileType: "py" })
- GrepTool regex + directory scope: grep({ pattern: "def\\s+\\w+", path: "adw" })
  → ripgrep({ contentPattern: "def\\s+\\w+", path: "adw" })

BEHAVIOR NOTES:
- File discovery uses only "pattern" (glob), and returns file paths sorted by mtime.
- Content search requires "contentPattern" (line results); "pattern" is optional and acts as a glob filter.
- Use "fileType" for language filters (e.g., "py", "ts") instead of GrepTool's include.
- Files-with-matches mode requires contentPattern + filesWithMatches/filesWithoutMatches.

PARAMETER MAPPING TO RIPGREP FLAGS:
| Parameter            | Ripgrep Flag             | Description                           |
|----------------------|--------------------------|---------------------------------------|
| pattern (discovery)  | --glob PATTERN           | File glob pattern (file discovery)    |
| pattern (content)    | --glob PATTERN           | File filter glob (content search)     |
| contentPattern       | (mode switch) + -n       | Triggers content search, adds line #s |
| filesWithMatches     | -l                       | Content search: return only file paths with matches |
| filesWithoutMatches  | -L                       | Content search: return only file paths without matches |
| ignoreGitignore      | --no-ignore-vcs          | Skip .gitignore rules                 |
| includeHidden        | --hidden                 | Include hidden files/dirs             |
| unrestricted: 1      | -u                       | Ignore .gitignore                     |
| unrestricted: 2      | -uu                      | Also hidden files                     |
| unrestricted: 3      | -uuu                     | Also binary files                     |
| fileType             | -t TYPE                  | Include only this file type           |
| excludeFileType      | -T TYPE                  | Exclude this file type                |
| globCaseInsensitive  | --glob-case-insensitive  | Case-insensitive glob matching        |
| maxResults (content) | --max-count              | Limits matches per file (fallback when maxMatchesPerFile unset; omitted when context flags active) |
| maxMatchesPerFile    | --max-count              | Limits matches per file (content search; overrides maxResults) |
| contextLines         | -C N                     | Context lines before and after matches (content search) |
| beforeContext        | -B N                     | Context lines before matches (content search) |
| afterContext         | -A N                     | Context lines after matches (content search) |
| compactOutput        | (output-only)            | Format paths relative to search path and files-with-matches output (no rg flag; ignored for raw match lines) |
 
 PARAMETER PRECEDENCE:

- contentPattern switches to content search; without it, content-only parameters are ignored
- filesWithMatches/filesWithoutMatches are mutually exclusive and only valid in content search
- beforeContext/afterContext override contextLines (skip -C when directional context is provided)
- maxMatchesPerFile takes precedence over maxResults for --max-count (content search only)
- --max-count is omitted when any context flags (-C/-B/-A) are present
- unrestricted takes precedence over ignoreGitignore and includeHidden
- unrestricted: 1 implies ignoreGitignore: true
- unrestricted: 2 implies both ignoreGitignore: true AND includeHidden: true
- unrestricted: 3 also includes binary files
- excludeFileType overrides include when both match the same type
- compactOutput only affects path-only outputs (file discovery and files-with-matches); it does not rewrite raw match lines

PARAMETER INTERACTIONS (SUMMARY):
| Mode                    | Requires              | Uses pattern/glob | Context flags | filesWithMatches/filesWithoutMatches | compactOutput |
|-------------------------|-----------------------|-------------------|---------------|--------------------------------------|---------------|
| File discovery          | pattern               | Yes (glob)        | Ignored       | Ignored                              | Applies       |
| Content search (lines)  | contentPattern        | Optional filter   | Applies       | Ignored                              | Ignored       |
| Files-with-matches      | contentPattern + flag | Optional filter   | Ignored       | Applies                              | Applies       |

PERFORMANCE:
- Results limited to maxResults (default 5000) to bound I/O
- Truncation warning shown when limit is reached
- Parallel async stat operations with batching for speed`,

  args: {
    pattern: tool.schema
      .string()
      .optional()
      .describe(`Glob pattern to match files (e.g., '**/*.ts', 'src/**/*.js').

Uses ripgrep's --glob option for pattern matching.
Patterns follow standard glob syntax:
- * matches any characters except /
- ** matches any characters including /
- ? matches any single character
- [abc] matches a, b, or c

In content search mode, 'pattern' is optional and used as a glob filter.

EXAMPLES:
- "**/*.ts" - all TypeScript files
- "src/**/*.{js,jsx}" - JS/JSX files in src/
- "**/package.json" - all package.json files`),

    contentPattern: tool.schema
      .string()
      .optional()
      .describe(`Regex pattern to search inside files (triggers content search mode).

When set, switches from file discovery (--files) to content search mode.
Output format: file:line_number:matched_line_content
Line numbers are always included (-n flag).

Can combine with 'pattern' parameter to filter files first:
- pattern: "**/*.py" + contentPattern: "import" → search imports in Python files only

EXAMPLES:
- contentPattern: "TODO" → find all TODO comments
- contentPattern: "function\\s+\\w+" → find function definitions
- contentPattern: "console\\.log" → find console.log calls
- contentPattern: "\\bTODO\\b" → find word-boundary matches`),

    filesWithMatches: tool.schema
      .boolean()
      .optional()
      .describe(`Return only file paths that contain matches (maps to -l). Content search only.

When true, output is a list of file paths with matches (no line content).
Ignored unless contentPattern is set. Mutually exclusive with filesWithoutMatches.
Context flags (-C/-B/-A) are ignored in this mode.`),

    filesWithoutMatches: tool.schema
      .boolean()
      .optional()
      .describe(`Return only file paths that do NOT contain matches (maps to -L). Content search only.

When true, output is a list of file paths without matches (no line content).
Ignored unless contentPattern is set. Mutually exclusive with filesWithMatches.
Context flags (-C/-B/-A) are ignored in this mode.`),

    path: tool.schema
      .string()
      .optional()
      .describe(`Directory to search in. Defaults to current working directory.

Must be within the repository boundary (security restriction).

EXAMPLES:
- "src" - search in src/ directory
- "." - search from current directory (default)
- "node_modules/lodash" - search specific dependency`),

    ignoreGitignore: tool.schema
      .boolean()
      .optional()
      .describe(`Skip .gitignore rules (maps to --no-ignore-vcs). Default: false.

When true, ripgrep will search files that would normally be ignored
by .gitignore rules. Useful for searching in node_modules/, build/, etc.

NOTE: unrestricted: 1+ implies this setting.`),

    includeHidden: tool.schema
      .boolean()
      .optional()
      .describe(`Include hidden files and directories (maps to --hidden). Default: false.

When true, ripgrep will include files and directories starting with a dot
(like .eslintrc, .prettierrc, .config/).

NOTE: unrestricted: 2+ implies this setting.`),

    unrestricted: tool.schema
      .number()
      .optional()
      .describe(`Unrestricted search level (1-3). Overrides other ignore flags.

Levels:
- 1 (-u): Ignore .gitignore rules (same as ignoreGitignore: true)
- 2 (-uu): Also include hidden files (same as both flags true)
- 3 (-uuu): Also include binary files

Higher levels are supersets of lower levels.
When set, ignores ignoreGitignore and includeHidden parameters.`),

    fileType: tool.schema
      .string()
      .optional()
      .describe(`Include only files of this type (maps to -t TYPE). Examples: ts, js, py, json, md.`),

    excludeFileType: tool.schema
      .string()
      .optional()
      .describe(`Exclude files of this type (maps to -T TYPE). Examples: ts, js, py, json, md.`),

    globCaseInsensitive: tool.schema
      .boolean()
      .optional()
      .describe(`Enable case-insensitive glob matching (maps to --glob-case-insensitive). Default: false.`),

    compactOutput: tool.schema
      .boolean()
      .optional()
      .describe(`Return paths relative to the search directory (resolved 'path' argument). Default: false.

When true, output paths are formatted relative to the search directory instead of the current working directory.
Applies to file discovery and files-with-matches output; raw content match lines are not rewritten.
No change when omitted or when path is '.' (same as cwd).
Useful when searching deep directories to reduce path prefixes (e.g., searching "src/deep/nested" yields "file.ts" instead of "src/deep/nested/file.ts").`),

    maxResults: tool.schema
      .number()
      .optional()
      .describe(`Maximum number of results to return. Default: 5000.

File discovery mode: Limits number of files returned.
Content search mode: Limits total output lines returned. Also passed as --max-count to ripgrep
when no context flags (-C/-B/-A) are active. When context is enabled, --max-count is omitted
to avoid truncating context lines; the output line cap still applies.

If maxMatchesPerFile is set, it overrides maxResults for the --max-count flag in content search.

Limits I/O operations for large searches. When searching unrestricted
in large directories (like node_modules/), thousands of files may match.

If results exceed this limit, a warning is appended to the output.
Increase for comprehensive searches; decrease for faster results.`),

    maxMatchesPerFile: tool.schema
      .number()
      .optional()
      .describe(`Maximum number of matches per file for content search (maps to --max-count).

Must be a non-negative integer. A value of 0 means unlimited matches per file.
When set, this takes precedence over maxResults for the --max-count flag.
Ignored for file discovery mode, for files-with-matches mode
(filesWithMatches/filesWithoutMatches), and when context flags (-C/-B/-A) are active.
In files-with-matches mode ripgrep returns paths only and --max-count is not used, so this
setting does not limit the amount of work performed or the number of paths returned.`),

    contextLines: tool.schema
      .number()
      .optional()
      .describe(`Show N lines before and after each match (maps to -C N). Content search only.

Ignored unless contentPattern is set. Must be a non-negative integer.
Use beforeContext/afterContext for directional context; those
override this setting when set to a value greater than 0.`),

    beforeContext: tool.schema
      .number()
      .optional()
      .describe(`Show N lines before each match (maps to -B N). Content search only.

Ignored unless contentPattern is set. Must be a non-negative integer.
When provided with a value greater than 0, contextLines is ignored.`),

    afterContext: tool.schema
      .number()
      .optional()
      .describe(`Show N lines after each match (maps to -A N). Content search only.

Ignored unless contentPattern is set. Must be a non-negative integer.
When provided with a value greater than 0, contextLines is ignored.`),
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
      maxResults,
      maxMatchesPerFile,
      contextLines,
      beforeContext,
      afterContext,
      filesWithMatches,
      filesWithoutMatches,
      compactOutput = false,
    } = args;

    const cwd = process.cwd();
    const effectiveMaxResults = maxResults ?? DEFAULT_MAX_RESULTS;
    const isContentSearch = contentPattern !== undefined;
    const isFilesMode = Boolean(filesWithMatches || filesWithoutMatches);
    const contentPatternWarning =
      "[WARNING: filesWithMatches/filesWithoutMatches are ignored unless contentPattern is set.]";
    const shouldWarnAboutContentPattern = !isContentSearch && isFilesMode;

    const appendFilesModeWarning = (message: string) =>
      shouldWarnAboutContentPattern ? `${message}\n\n${contentPatternWarning}` : message;

    // === VALIDATION ===

    if (isContentSearch && (!contentPattern || !contentPattern.trim())) {
      return "ERROR: 'contentPattern' cannot be empty.";
    }

    if (isFilesMode && !isContentSearch) {
      return `${contentPatternWarning}\n\nHint: Set contentPattern when using filesWithMatches/filesWithoutMatches.`;
    }

    if (filesWithMatches && filesWithoutMatches) {
      const errorMessage = "ERROR: 'filesWithMatches' and 'filesWithoutMatches' cannot both be true.";
      return appendFilesModeWarning(errorMessage);
    }

    // Validate pattern (required for file discovery)
    if (!isContentSearch && (!pattern || typeof pattern !== "string" || !pattern.trim())) {
      return appendFilesModeWarning(
        "ERROR: 'pattern' parameter is required and cannot be empty.\n\nHint: Provide a glob pattern like '**/*.ts' or 'src/**/*.js'"
      );
    }

    // Validate and resolve search path
    const resolvedSearchPath = searchPath
      ? path.isAbsolute(searchPath)
        ? path.normalize(searchPath)
        : path.resolve(cwd, searchPath)
      : cwd;

    // Security: ensure search path is within repository
    if (!isWithinRepository(resolvedSearchPath, cwd)) {
      const errorMessage = `ERROR: Search path is outside the repository: ${resolvedSearchPath}\n\nHint: All searches must stay within the repository root (${cwd}).`;
      return appendFilesModeWarning(errorMessage);
    }

    // Check if search path exists
    try {
      const stats = await fs.promises.stat(resolvedSearchPath);
      if (!stats.isDirectory()) {
        const errorMessage = `ERROR: Search path is not a directory: ${resolvedSearchPath}\n\nHint: Provide a directory path to search in.`;
        return appendFilesModeWarning(errorMessage);
      }
    } catch {
      const errorMessage = `ERROR: Search path does not exist: ${resolvedSearchPath}\n\nHint: Verify the path is correct.`;
      return appendFilesModeWarning(errorMessage);
    }

    // Validate unrestricted level
    if (unrestricted !== undefined && (unrestricted < 1 || unrestricted > 3)) {
      return "ERROR: 'unrestricted' must be 1, 2, or 3.\n\nLevels:\n- 1: Ignore .gitignore\n- 2: Also include hidden files\n- 3: Also include binary files";
    }

    // === EXECUTE INITIAL SEARCH ===

    // Build search params from args
    const searchParams: SearchParams = {
      pattern,
      contentPattern,
      searchPath: resolvedSearchPath,
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
      filesWithMatches,
      filesWithoutMatches,
    };

    // Execute initial search
    const initialResult = await executeRipgrepSearch(searchParams);

    // Handle errors from initial search
    if (initialResult.errorMessage) {
      return initialResult.errorMessage;
    }

    if (isContentSearch) {
      const pathOnlyMode = Boolean(filesWithMatches || filesWithoutMatches);
      const rawLines = initialResult.rawLines ?? [];
      if (rawLines.length === 0) {
        if (filesWithoutMatches) {
          return `No files found without matches for contentPattern '${contentPattern}'.`;
        }
        if (filesWithMatches) {
          return `No files found with matches for contentPattern '${contentPattern}'.`;
        }
        return `No matches found for contentPattern '${contentPattern}'.`;
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
      return result;
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
      !unrestricted; // User didn't use unrestricted mode

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
        return retryResult.errorMessage;
      }

      // Use retry results if found
      if (retryResult.files.length > 0) {
        files = retryResult.files;
        didAutoRetry = true;
      } else {
        // Both searches returned nothing
        const searchContext = searchPath ? ` in '${searchPath}'` : "";
        let result =
          `No files found matching pattern '${pattern}'${searchContext}.\n\n` +
          "Note: Auto-retry was attempted with ignoreGitignore=true, includeHidden=true but still found no results.";
        if (hasContextParams) {
          result += `\n\n${CONTEXT_PARAMS_IGNORED_WARNING}`;
        }
        return result;
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
      return result;
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

    return result;
  },
});
