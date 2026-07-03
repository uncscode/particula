import { tool } from "@opencode-ai/plugin";
import * as path from "node:path";
import {
  buildTruncationWarning,
  DEFAULT_MAX_RESULTS,
  executeRipgrepSearch,
  normalizeNumericParam,
  resolveValidatedSearchPath,
} from "./lib/ripgrep_shared";

// --- Tool-local helpers ---

const ADVANCED_ONLY_PARAMS = [
  "contextLines",
  "beforeContext",
  "afterContext",
  "filesWithMatches",
  "filesWithoutMatches",
  "unrestricted",
  "ignoreGitignore",
  "includeHidden",
] as const;

const TOKEN_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;

interface ParsedSearchContentOptions {
  pattern?: string;
  fileType?: string;
  excludeFileType?: string;
  globCaseInsensitive?: boolean;
  compactOutput?: boolean;
  maxResults?: number;
  maxMatchesPerFile?: number;
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

function parseSearchContentOptions(rawOptions: unknown):
  | { ok: true; options: ParsedSearchContentOptions }
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

  const parsed: ParsedSearchContentOptions = {};

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

    if (tokenName === "max-matches-per-file") {
      if (!/^\d+$/.test(rawValue)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': max-matches-per-file must be a non-negative integer.` };
      }
      if (parsed.maxMatchesPerFile !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'max-matches-per-file' token is not allowed.` };
      }
      parsed.maxMatchesPerFile = Number(rawValue);
      continue;
    }

    return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for this wrapper.` };
  }

  return { ok: true, options: parsed };
}

// --- Tool definition ---

export default tool({
  description: `Search for file content using ripgrep with a constrained schema. Only include parameters you need — omit all others.

SIMPLE EXAMPLES (copy these patterns):

Minimal search:   { contentPattern: "TODO" }
Filtered search:  { contentPattern: "import", path: "adw", options: "file-type=py" }
Bounded output:   { contentPattern: "ERROR:", options: "max-results=200 max-matches-per-file=3" }

RULES:
- Required 'contentPattern' must be a non-empty string after trim.
- Scope is simple content search only; advanced controls are rejected.
- Use ripgrep_advanced for advanced controls.
- Searches are constrained to repository boundaries.
- No matches return a deterministic non-error message.`,

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

    for (const field of ADVANCED_ONLY_PARAMS) {
      if (isMateriallySet((args as Record<string, unknown>)[field])) {
        return (
          `ERROR: '${field}' is not supported by search_content.\n\n` +
          "Hint: Use ripgrep_advanced in this repository for advanced controls."
        );
      }
    }

    const parsedOptions = parseSearchContentOptions((args as Record<string, unknown>).options);
    if (!parsedOptions.ok) return parsedOptions.error;

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
