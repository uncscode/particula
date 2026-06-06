import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "node:path";

// --- Inlined from lib/ripgrep_shared.ts ---

const DEFAULT_MAX_RESULTS = 5000;

const MAX_STDOUT_CAPTURE_BYTES = 4 * 1024 * 1024;

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

interface SearchResult {
  files: string[];
  rawLines?: string[];
  exitCode: number;
  errorMessage?: string;
  outputClipped?: boolean;
}

interface ValidatedSearchPathResult {
  canonicalPath?: string;
  error?: string;
}

function normalizeNumericParam(value: number | undefined): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) return undefined;
  return value;
}

function validateNonNegativeInt(value: unknown, paramName: string): string | undefined {
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

function buildTruncationWarning(
  limit: number,
  total: number,
  unit: "files" | "lines",
  options?: { approximateTotal?: boolean }
): string {
  const qualifier = options?.approximateTotal ? `at least ${total}` : `${total}`;
  return `[WARNING: Results truncated to ${limit} ${unit} (${qualifier} total found). Use maxResults parameter to increase limit.]`;
}

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
  description: `Search file content using advanced ripgrep controls. Use this only for advanced/low-level search behavior.

SIMPLE EXAMPLES (copy these patterns):

Context lines:          { contentPattern: "TODO", contextLines: 2 }
Before/after context:   { contentPattern: "ERROR", beforeContext: 2, afterContext: 1 }
Files with matches:     { contentPattern: "import", filesWithMatches: true }
Files without matches:  { contentPattern: "TODO", filesWithoutMatches: true, path: "adw" }
Unrestricted search:    { contentPattern: "secret", unrestricted: 2 }

RULES:
- Required 'contentPattern' must be a non-empty string after trim.
- Advanced controls are supported (context/files-with/files-without/unrestricted/hidden/ignored).
- Directional context (-A/-B) takes precedence over contextLines (-C).
- filesWithMatches and filesWithoutMatches are mutually exclusive.
- Searches are constrained to repository boundaries.`,

  args: {
    contentPattern: tool.schema.string(),
    pattern: tool.schema.string().optional(),
    path: tool.schema.string().optional(),
    fileType: tool.schema.string().optional(),
    excludeFileType: tool.schema.string().optional(),
    globCaseInsensitive: tool.schema.boolean().optional(),
    compactOutput: tool.schema.boolean().optional(),
    maxResults: tool.schema.number().optional(),
    maxMatchesPerFile: tool.schema.number().optional(),
    contextLines: tool.schema.number().optional(),
    beforeContext: tool.schema.number().optional(),
    afterContext: tool.schema.number().optional(),
    filesWithMatches: tool.schema.boolean().optional(),
    filesWithoutMatches: tool.schema.boolean().optional(),
    unrestricted: tool.schema.number().optional(),
    ignoreGitignore: tool.schema.boolean().optional(),
    includeHidden: tool.schema.boolean().optional(),
  },

  async execute(args) {
    if (typeof args.contentPattern !== "string") {
      return "ERROR: 'contentPattern' must be a string.\n\nHint: Provide a regex string such as 'TODO' or 'import\\s+os'.";
    }

    const contentPattern = args.contentPattern.trim();
    if (!contentPattern) {
      return "ERROR: 'contentPattern' parameter is required and cannot be empty.\n\nHint: Provide a non-empty regex string such as 'TODO'.";
    }

    if (args.filesWithMatches && args.filesWithoutMatches) {
      return (
        "ERROR: 'filesWithMatches' and 'filesWithoutMatches' cannot both be true.\n\n" +
        "Hint: Choose exactly one file-mode toggle."
      );
    }

    const normalizedFileType = normalizeOptionalType(args.fileType);
    if (normalizedFileType.error) return normalizedFileType.error;
    const normalizedExcludeFileType = normalizeOptionalType(args.excludeFileType);
    if (normalizedExcludeFileType.error) return normalizedExcludeFileType.error;

    for (const [value, name] of [
      [args.contextLines, "contextLines"],
      [args.beforeContext, "beforeContext"],
      [args.afterContext, "afterContext"],
      [args.maxMatchesPerFile, "maxMatchesPerFile"],
      [args.unrestricted, "unrestricted"],
    ] as const) {
      const err = validateNonNegativeInt(value, name);
      if (err) return err;
    }

    const normalizedUnrestricted = normalizeNumericParam(args.unrestricted);
    if (normalizedUnrestricted !== undefined && (normalizedUnrestricted < 1 || normalizedUnrestricted > 3)) {
      return "ERROR: Invalid unrestricted value. It must be an integer between 1 and 3.";
    }

    if (args.maxResults !== undefined) {
      const maxResultsError = validateNonNegativeInt(args.maxResults, "maxResults");
      if (maxResultsError) return maxResultsError;
    }

    const normalizedMaxResults = normalizeNumericParam(args.maxResults);
    const normalizedMaxMatchesPerFile = normalizeNumericParam(args.maxMatchesPerFile);
    const maxResults = normalizedMaxResults ?? DEFAULT_MAX_RESULTS;

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
      contentPattern,
      pattern: args.pattern,
      searchPath: executedSearchPath,
      fileType: normalizedFileType.value,
      excludeFileType: normalizedExcludeFileType.value,
      globCaseInsensitive: args.globCaseInsensitive,
      compactOutput: args.compactOutput,
      maxResults,
      maxMatchesPerFile: normalizedMaxMatchesPerFile,
      contextLines: args.contextLines,
      beforeContext: args.beforeContext,
      afterContext: args.afterContext,
      filesWithMatches: args.filesWithMatches,
      filesWithoutMatches: args.filesWithoutMatches,
      unrestricted: normalizedUnrestricted,
      ignoreGitignore: args.ignoreGitignore,
      includeHidden: args.includeHidden,
    });

    if (searchResult.errorMessage) return searchResult.errorMessage;

    const lines = searchResult.rawLines ?? [];
    if (lines.length === 0) {
      const searchContext = args.path ? ` in '${args.path}'` : "";
      return `No matches found for contentPattern '${contentPattern}'${searchContext}.`;
    }

    const wasTruncated = lines.length > maxResults;
    const limited = lines.slice(0, maxResults);
    let output = limited.join("\n");
    if (wasTruncated) {
      output += `\n\n${buildTruncationWarning(maxResults, lines.length, "lines", { approximateTotal: true })}`;
    }
    if (searchResult.outputClipped) {
      output += "\n\n[WARNING: Ripgrep stdout was clipped for safety. Refine your query (path/pattern/fileType) or lower maxResults.]";
    }
    return output;
  },
});
