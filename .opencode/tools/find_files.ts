import { tool } from "@opencode-ai/plugin";
import * as fs from "fs";
import * as path from "node:path";
import {
  buildTruncationWarning,
  DEFAULT_MAX_RESULTS,
  executeRipgrepSearch,
  normalizeNumericParam,
  resolveValidatedSearchPath,
} from "./lib/ripgrep_shared";

/** Batch size for parallel stat operations. Prevents EMFILE errors. */
const STAT_BATCH_SIZE = 100;

/** File entry with modification time for sorting. */
interface FileWithMtime {
  path: string;
  mtime: number;
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

const TOKEN_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;

interface ParsedFindFilesOptions {
  fileType?: string;
  excludeFileType?: string;
  globCaseInsensitive?: boolean;
  compactOutput?: boolean;
  maxResults?: number;
}

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

function parseFindFilesOptions(rawOptions: unknown):
  | { ok: true; options: ParsedFindFilesOptions }
  | { ok: false; error: string } {
  if (rawOptions === undefined || rawOptions === null) {
    return { ok: true, options: {} };
  }
  if (typeof rawOptions !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalizedOptions = rawOptions.trim();
  if (!normalizedOptions) {
    return { ok: true, options: {} };
  }

  const parsed: ParsedFindFilesOptions = {};

  for (const token of normalizedOptions.split(/\s+/)) {
    const separatorCount = token.split("=").length - 1;
    if (separatorCount > 1) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }

    if (separatorCount === 0) {
      if (!TOKEN_NAME_PATTERN.test(token)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': token names must use lowercase-kebab-case.` };
      }
      if (token === "glob-case-insensitive") {
        if (parsed.globCaseInsensitive) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'glob-case-insensitive' token is not allowed.` };
        }
        parsed.globCaseInsensitive = true;
        continue;
      }
      if (token === "compact-output") {
        if (parsed.compactOutput) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'compact-output' token is not allowed.` };
        }
        parsed.compactOutput = true;
        continue;
      }
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    const [tokenName, rawValue] = token.split("=");
    if (!TOKEN_NAME_PATTERN.test(tokenName)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token names must use lowercase-kebab-case.` };
    }
    if (!rawValue) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token value must not be empty.` };
    }

    if (tokenName === "glob-case-insensitive" || tokenName === "compact-output") {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token must be provided without '=value'.` };
    }

    if (tokenName === "file-type") {
      if (parsed.fileType !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'file-type' token is not allowed.` };
      }
      const normalized = normalizeOptionalType(rawValue);
      if (normalized.error) return { ok: false, error: normalized.error };
      parsed.fileType = normalized.value;
      continue;
    }

    if (tokenName === "exclude-file-type") {
      if (parsed.excludeFileType !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'exclude-file-type' token is not allowed.` };
      }
      const normalized = normalizeOptionalType(rawValue);
      if (normalized.error) return { ok: false, error: normalized.error };
      parsed.excludeFileType = normalized.value;
      continue;
    }

    if (tokenName === "max-results") {
      if (!/^\d+$/.test(rawValue)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': max-results must be a non-negative integer.` };
      }
      if (parsed.maxResults !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'max-results' token is not allowed.` };
      }
      parsed.maxResults = Number(rawValue);
      continue;
    }

    return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for this wrapper.` };
  }

  return { ok: true, options: parsed };
}

// --- Tool definition ---

export default tool({
  description: `Search for files by glob pattern using discovery-only ripgrep mode. Only include parameters you need — omit all others.

SIMPLE EXAMPLES (copy these patterns):

Basic search:      { pattern: "**/*.ts" }
Search in folder:  { pattern: "**/*.py", path: "adw" }
Limit results:     { pattern: "**/*", options: "max-results=100" }
Compact output:    { pattern: "**/*.md", path: "docs", options: "compact-output" }
File type include: { pattern: "**/*", options: "file-type=py" }
File type exclude: { pattern: "**/*", options: "exclude-file-type=json" }

RULES:
- Discovery only: content-search parameters are rejected (use search_content for simple content search or ripgrep_advanced for advanced controls).
- Required 'pattern' must be a non-empty string after trim.
- Results are sorted by mtime (most recent first).
- No matches return a deterministic non-error message.
- Searches are constrained to repository boundaries.`,

  args: {
    pattern: tool.schema.string(),
    path: tool.schema.string().optional(),
    options: tool.schema.string().optional(),

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

    const parsedOptions = parseFindFilesOptions((args as Record<string, unknown>).options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const normalizedMaxResults = normalizeNumericParam(parsedOptions.options.maxResults);
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
      pattern,
      searchPath: executedSearchPath,
      fileType: parsedOptions.options.fileType,
      excludeFileType: parsedOptions.options.excludeFileType,
      globCaseInsensitive: parsedOptions.options.globCaseInsensitive,
      compactOutput: parsedOptions.options.compactOutput,
      maxResults,
    });

    if (searchResult.errorMessage) {
      return searchResult.errorMessage;
    }

    if (searchResult.files.length === 0) {
      const searchContext = args.path ? ` in '${args.path}'` : "";
      return `No files found matching pattern '${pattern}'${searchContext}.`;
    }

    const wasTruncated = searchResult.files.length > maxResults;
    const filesToProcess = searchResult.files;
    const filesWithMtime = await getFilesWithMtime(filesToProcess);
    if (filesToProcess.length > 0 && filesWithMtime.length === 0) {
      return "ERROR: Failed to read metadata for matched files.\n\nHint: Verify file permissions and path accessibility.";
    }
    filesWithMtime.sort((a, b) => b.mtime - a.mtime);
    const limitedFiles = filesWithMtime.slice(0, maxResults);

    const basePath = parsedOptions.options.compactOutput
      ? validatedPath.compactOutputBase ?? path.dirname(executedSearchPath)
      : cwd;
    let output = "";
    for (let i = 0; i < limitedFiles.length; i++) {
      const filePath = limitedFiles[i]?.path;
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
