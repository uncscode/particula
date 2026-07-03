import { tool } from "@opencode-ai/plugin";
import * as path from "node:path";
import {
  buildTruncationWarning,
  DEFAULT_MAX_RESULTS,
  executeRipgrepSearch,
  normalizeNumericParam,
  resolveValidatedSearchPath,
  validateNonNegativeInt,
} from "./lib/ripgrep_shared";

// --- Tool-local helpers ---

const TOKEN_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;

interface ParsedRipgrepAdvancedOptions {
  pattern?: string;
  fileType?: string;
  excludeFileType?: string;
  globCaseInsensitive?: boolean;
  compactOutput?: boolean;
  maxResults?: number;
  maxMatchesPerFile?: number;
  contextLines?: number;
  beforeContext?: number;
  afterContext?: number;
  filesWithMatches?: boolean;
  filesWithoutMatches?: boolean;
  unrestricted?: number;
  ignoreGitignore?: boolean;
  includeHidden?: boolean;
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

function parseNonNegativeIntegerToken(token: string, rawValue: string, label: string): number | string {
  if (!/^\d+$/.test(rawValue)) {
    return `ERROR: Invalid options token '${token}': ${label} must be a non-negative integer.`;
  }
  return Number(rawValue);
}

function parseRipgrepAdvancedOptions(rawOptions: unknown):
  | { ok: true; options: ParsedRipgrepAdvancedOptions }
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

  const parsed: ParsedRipgrepAdvancedOptions = {};

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
      if (token === "files-with-matches") {
        if (parsed.filesWithMatches) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'files-with-matches' token is not allowed.` };
        }
        parsed.filesWithMatches = true;
        continue;
      }
      if (token === "files-without-matches") {
        if (parsed.filesWithoutMatches) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'files-without-matches' token is not allowed.` };
        }
        parsed.filesWithoutMatches = true;
        continue;
      }
      if (token === "ignore-gitignore") {
        if (parsed.ignoreGitignore) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'ignore-gitignore' token is not allowed.` };
        }
        parsed.ignoreGitignore = true;
        continue;
      }
      if (token === "include-hidden") {
        if (parsed.includeHidden) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'include-hidden' token is not allowed.` };
        }
        parsed.includeHidden = true;
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

    if (["glob-case-insensitive", "compact-output", "files-with-matches", "files-without-matches", "ignore-gitignore", "include-hidden"].includes(tokenName)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token must be provided without '=value'.` };
    }

    if (tokenName === "pattern") {
      if (parsed.pattern !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'pattern' token is not allowed.` };
      }
      parsed.pattern = rawValue;
      continue;
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

    const parsedNumber = (() => {
      switch (tokenName) {
        case "max-results":
          return parseNonNegativeIntegerToken(token, rawValue, "max-results");
        case "max-matches-per-file":
          return parseNonNegativeIntegerToken(token, rawValue, "max-matches-per-file");
        case "context-lines":
          return parseNonNegativeIntegerToken(token, rawValue, "context-lines");
        case "before-context":
          return parseNonNegativeIntegerToken(token, rawValue, "before-context");
        case "after-context":
          return parseNonNegativeIntegerToken(token, rawValue, "after-context");
        case "unrestricted":
          return parseNonNegativeIntegerToken(token, rawValue, "unrestricted");
        default:
          return undefined;
      }
    })();

    if (typeof parsedNumber === "string") {
      return { ok: false, error: parsedNumber };
    }

    if (tokenName === "max-results") {
      if (parsed.maxResults !== undefined) return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'max-results' token is not allowed.` };
      parsed.maxResults = parsedNumber;
      continue;
    }
    if (tokenName === "max-matches-per-file") {
      if (parsed.maxMatchesPerFile !== undefined) return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'max-matches-per-file' token is not allowed.` };
      parsed.maxMatchesPerFile = parsedNumber;
      continue;
    }
    if (tokenName === "context-lines") {
      if (parsed.contextLines !== undefined) return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'context-lines' token is not allowed.` };
      parsed.contextLines = parsedNumber;
      continue;
    }
    if (tokenName === "before-context") {
      if (parsed.beforeContext !== undefined) return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'before-context' token is not allowed.` };
      parsed.beforeContext = parsedNumber;
      continue;
    }
    if (tokenName === "after-context") {
      if (parsed.afterContext !== undefined) return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'after-context' token is not allowed.` };
      parsed.afterContext = parsedNumber;
      continue;
    }
    if (tokenName === "unrestricted") {
      if (parsed.unrestricted !== undefined) return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'unrestricted' token is not allowed.` };
      parsed.unrestricted = parsedNumber;
      continue;
    }

    return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for this wrapper.` };
  }

  return { ok: true, options: parsed };
}

// --- Tool definition ---

export default tool({
  description: `Search file content using advanced ripgrep controls. Use this only for advanced/low-level search behavior.

SIMPLE EXAMPLES (copy these patterns):

Context lines:          { contentPattern: "TODO", options: "context-lines=2" }
Before/after context:   { contentPattern: "ERROR", options: "before-context=2 after-context=1" }
Files with matches:     { contentPattern: "import", options: "files-with-matches" }
Files without matches:  { contentPattern: "TODO", path: "adw", options: "files-without-matches" }
Unrestricted search:    { contentPattern: "secret", options: "unrestricted=2" }

RULES:
- Required 'contentPattern' must be a non-empty string after trim.
- Advanced controls are supported (context/files-with/files-without/unrestricted/hidden/ignored).
- Directional context (-A/-B) takes precedence over contextLines (-C).
- filesWithMatches and filesWithoutMatches are mutually exclusive.
- Searches are constrained to repository boundaries.`,

  args: {
    contentPattern: tool.schema.string(),
    path: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },

  async execute(args) {
    if (typeof args.contentPattern !== "string") {
      return "ERROR: 'contentPattern' must be a string.\n\nHint: Provide a regex string such as 'TODO' or 'import\\s+os'.";
    }

    const contentPattern = args.contentPattern.trim();
    if (!contentPattern) {
      return "ERROR: 'contentPattern' parameter is required and cannot be empty.\n\nHint: Provide a non-empty regex string such as 'TODO'.";
    }

    const parsedOptions = parseRipgrepAdvancedOptions((args as Record<string, unknown>).options);
    if (!parsedOptions.ok) return parsedOptions.error;

    if (parsedOptions.options.filesWithMatches && parsedOptions.options.filesWithoutMatches) {
      return (
        "ERROR: 'filesWithMatches' and 'filesWithoutMatches' cannot both be true.\n\n" +
        "Hint: Choose exactly one file-mode toggle."
      );
    }

    for (const [value, name] of [
      [parsedOptions.options.contextLines, "contextLines"],
      [parsedOptions.options.beforeContext, "beforeContext"],
      [parsedOptions.options.afterContext, "afterContext"],
      [parsedOptions.options.maxMatchesPerFile, "maxMatchesPerFile"],
      [parsedOptions.options.unrestricted, "unrestricted"],
    ] as const) {
      const err = validateNonNegativeInt(value, name);
      if (err) return err;
    }

    const normalizedUnrestricted = normalizeNumericParam(parsedOptions.options.unrestricted);
    if (normalizedUnrestricted !== undefined && (normalizedUnrestricted < 1 || normalizedUnrestricted > 3)) {
      return "ERROR: Invalid unrestricted value. It must be an integer between 1 and 3.";
    }

    if (parsedOptions.options.maxResults !== undefined) {
      const maxResultsError = validateNonNegativeInt(parsedOptions.options.maxResults, "maxResults");
      if (maxResultsError) return maxResultsError;
    }

    const normalizedMaxResults = normalizeNumericParam(parsedOptions.options.maxResults);
    const normalizedMaxMatchesPerFile = normalizeNumericParam(parsedOptions.options.maxMatchesPerFile);
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
      pattern: parsedOptions.options.pattern,
      searchPath: executedSearchPath,
      fileType: parsedOptions.options.fileType,
      excludeFileType: parsedOptions.options.excludeFileType,
      globCaseInsensitive: parsedOptions.options.globCaseInsensitive,
      compactOutput: parsedOptions.options.compactOutput,
      compactOutputBase: parsedOptions.options.compactOutput
        ? validatedPath.compactOutputBase ?? path.dirname(executedSearchPath)
        : undefined,
      targetKind: validatedPath.targetKind,
      maxResults,
      maxMatchesPerFile: normalizedMaxMatchesPerFile,
      contextLines: parsedOptions.options.contextLines,
      beforeContext: parsedOptions.options.beforeContext,
      afterContext: parsedOptions.options.afterContext,
      filesWithMatches: parsedOptions.options.filesWithMatches,
      filesWithoutMatches: parsedOptions.options.filesWithoutMatches,
      unrestricted: normalizedUnrestricted,
      ignoreGitignore: parsedOptions.options.ignoreGitignore,
      includeHidden: parsedOptions.options.includeHidden,
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
