import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

export type OutputMode = "summary" | "full" | "json";

type ParsedOptionsResult<T> =
  | { ok: true; options: T }
  | { ok: false; error: string };

type ParsedCppLintOptions = {
  outputMode?: OutputMode;
  linters?: string[];
};

type ParsedCppCoverageSummaryOptions = {
  outputMode?: OutputMode;
};

type ParsedCppCoverageAdvancedOptions = {
  outputMode?: OutputMode;
  tool?: string;
};

type RedactionRule = {
  pattern: RegExp;
  redact: (...groups: string[]) => string;
};

const MAX_DIAGNOSTIC_LENGTH = 4_000;
const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;
const REDACTION_RULES: RedactionRule[] = [
  {
    pattern: /\b(auth|token|key|password|secret)=([^\s,]+)/gi,
    redact: (key) => `${key}=[REDACTED]`,
  },
  {
    pattern: /(https?:\/\/[^:\s]+:)([^@\s/]+)(@[^\s]+)/gi,
    redact: (prefix, _secret, suffix) => `${prefix}[REDACTED]${suffix}`,
  },
  {
    pattern: /(bearer\s+)([^\s]+)/gi,
    redact: (prefix) => `${prefix}[REDACTED]`,
  },
];

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

export const SUPPORTED_CPP_LINTERS = ["clang-format", "clang-tidy", "cppcheck"] as const;
export const SUPPORTED_CPP_LINTER_SET = new Set<string>(SUPPORTED_CPP_LINTERS);
export const SUPPORTED_CPP_COVERAGE_TOOLS = ["gcov", "llvm-cov"] as const;
export const SUPPORTED_CPP_COVERAGE_TOOL_SET = new Set<string>(SUPPORTED_CPP_COVERAGE_TOOLS);

type ValidatedRepoPathResult =
  | { ok: true; path: string }
  | { ok: false; error: string };

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

function tokenizeOptions(options: string): { ok: true; tokens: string[] } | { ok: false; error: string } {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;

  for (let index = 0; index < options.length; index += 1) {
    const char = options[index];
    if (quote) {
      current += char;
      if (char === quote) {
        quote = undefined;
      }
      continue;
    }

    if (char === "'" || char === '"') {
      quote = char;
      current += char;
      continue;
    }

    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }

    current += char;
  }

  if (quote) {
    return { ok: false, error: "ERROR: Invalid options string: unterminated quoted value." };
  }
  if (current) {
    tokens.push(current);
  }

  return { ok: true, tokens };
}

function stripOptionalQuotes(value: string): string {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === '"' || first === "'") && last === first) {
      return value.slice(1, -1);
    }
  }

  return value;
}

function parseOptionsPrelude<T>(
  rawOptions: unknown,
): ParsedOptionsResult<{ parsed: T; tokens: string[] }> {
  if (rawOptions === undefined || rawOptions === null) {
    return { ok: true, options: { parsed: {} as T, tokens: [] } };
  }
  if (typeof rawOptions !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalized = rawOptions.trim();
  if (!normalized) {
    return { ok: true, options: { parsed: {} as T, tokens: [] } };
  }

  const tokenized = tokenizeOptions(normalized);
  if (!tokenized.ok) {
    return tokenized;
  }

  return { ok: true, options: { parsed: {} as T, tokens: tokenized.tokens } };
}

function parseOutputToken(
  token: string,
  rawValue: string,
  parsed: { outputMode?: OutputMode },
): string | undefined {
  const value = stripOptionalQuotes(rawValue).trim();
  if (!value) {
    return `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`;
  }
  if (!OUTPUT_MODES.has(value as OutputMode)) {
    return `ERROR: Invalid options token '${token}': output must be one of summary, full, json.`;
  }
  if (parsed.outputMode !== undefined) {
    return `ERROR: Invalid options token '${token}': duplicate token.`;
  }
  parsed.outputMode = value as OutputMode;
  return undefined;
}

export function parseCppLintOptions(rawOptions: unknown): ParsedOptionsResult<ParsedCppLintOptions> {
  const prelude = parseOptionsPrelude<ParsedCppLintOptions>(rawOptions);
  if (!prelude.ok) {
    return prelude;
  }

  const { parsed, tokens } = prelude.options;
  for (const token of tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }
    if (separatorIndex === -1) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    const name = token.slice(0, separatorIndex);
    const rawValue = token.slice(separatorIndex + 1);
    if (!rawValue) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    if (name === "output") {
      const error = parseOutputToken(token, rawValue, parsed);
      if (error) {
        return { ok: false, error };
      }
      continue;
    }

    if (name === "linters") {
      if (parsed.linters !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      const linters = stripOptionalQuotes(rawValue)
        .split(",")
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
      if (linters.length === 0) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': linters must contain at least one supported linter.`,
        };
      }
      const invalidLinters = linters.filter((linter) => !SUPPORTED_CPP_LINTER_SET.has(linter));
      if (invalidLinters.length > 0) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': unsupported linter '${invalidLinters[0]}'.`,
        };
      }
      parsed.linters = linters;
      continue;
    }

    return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
  }

  return { ok: true, options: parsed };
}

export function parseCppCoverageSummaryOptions(
  rawOptions: unknown,
): ParsedOptionsResult<ParsedCppCoverageSummaryOptions> {
  const prelude = parseOptionsPrelude<ParsedCppCoverageSummaryOptions>(rawOptions);
  if (!prelude.ok) {
    return prelude;
  }

  const { parsed, tokens } = prelude.options;
  for (const token of tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }
    if (separatorIndex === -1) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    const name = token.slice(0, separatorIndex);
    const rawValue = token.slice(separatorIndex + 1);
    if (name !== "output") {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
    }

    const error = parseOutputToken(token, rawValue, parsed);
    if (error) {
      return { ok: false, error };
    }
  }

  return { ok: true, options: parsed };
}

export function parseCppCoverageAdvancedOptions(
  rawOptions: unknown,
): ParsedOptionsResult<ParsedCppCoverageAdvancedOptions> {
  const prelude = parseOptionsPrelude<ParsedCppCoverageAdvancedOptions>(rawOptions);
  if (!prelude.ok) {
    return prelude;
  }

  const { parsed, tokens } = prelude.options;
  for (const token of tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }
    if (separatorIndex === -1) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    const name = token.slice(0, separatorIndex);
    const rawValue = token.slice(separatorIndex + 1);
    if (!rawValue) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    if (name === "output") {
      const error = parseOutputToken(token, rawValue, parsed);
      if (error) {
        return { ok: false, error };
      }
      continue;
    }

    if (name === "tool") {
      const value = stripOptionalQuotes(rawValue).trim();
      if (!value) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
      }
      if (!SUPPORTED_CPP_COVERAGE_TOOL_SET.has(value)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': tool must be one of gcov, llvm-cov.` };
      }
      if (parsed.tool !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.tool = value;
      continue;
    }

    return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
  }

  return { ok: true, options: parsed };
}

function escapeControlChars(value: string): string {
  return value.replace(/[\x00-\x1f\x7f]/g, (char) => {
    const hex = char.charCodeAt(0).toString(16).padStart(2, "0");
    return `\\x${hex}`;
  });
}

function redactSecrets(value: string): string {
  let redacted = value;
  for (const rule of REDACTION_RULES) {
    redacted = redacted.replace(rule.pattern, (...args: string[]) => {
      const groups = args.slice(1, -2);
      return rule.redact(...groups);
    });
  }
  return redacted;
}

export function sanitizeDiagnosticValue(value: string): string {
  const sanitized = redactSecrets(escapeControlChars(value));
  if (sanitized.length <= MAX_DIAGNOSTIC_LENGTH) {
    return sanitized;
  }
  return `${sanitized.slice(0, MAX_DIAGNOSTIC_LENGTH)}... [truncated]`;
}

function isPathWithinRepo(repoRoot: string, resolvedPath: string): boolean {
  const rel = path.relative(repoRoot, resolvedPath);
  return !(rel.startsWith("..") || path.isAbsolute(rel));
}

function buildOutsideRepoError(label: string, pathValue: string, resolvedPath: string): string {
  return `ERROR: ${label} path resolves outside repository root: ${pathValue} (canonical: ${resolvedPath})`;
}

function getRepoRoot(): string {
  return realpathSync(process.cwd());
}

function ensureDirectoryStat(pathValue: string, label: string): string | undefined {
  if (!isStatDirectory(statSync(pathValue))) {
    return `ERROR: ${label} path is not a directory: ${pathValue}`;
  }
  return undefined;
}

export function rejectLegacyDirectFields(
  args: Record<string, unknown>,
  wrapperName: string,
  fields: readonly string[],
): string | undefined {
  for (const field of fields) {
    if (Object.hasOwn(args, field)) {
      return `ERROR: ${wrapperName} does not accept direct field '${field}'. Use bounded 'options' tokens instead.`;
    }
  }

  return undefined;
}

export function resolveExistingDirectoryWithinRepo(
  pathValue: string,
  label: string,
): ValidatedRepoPathResult {
  try {
    if (!existsSync(pathValue)) {
      return { ok: false, error: `ERROR: ${label} path does not exist: ${pathValue}` };
    }
    const directoryError = ensureDirectoryStat(pathValue, label);
    if (directoryError) {
      return { ok: false, error: directoryError };
    }

    const repoRoot = getRepoRoot();
    const resolvedPath = realpathSync(pathValue);
    if (!isPathWithinRepo(repoRoot, resolvedPath)) {
      return { ok: false, error: buildOutsideRepoError(label, pathValue, resolvedPath) };
    }

    return { ok: true, path: resolvedPath };
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, error: `ERROR: invalid ${label} path: ${pathValue} (${message})` };
  }
}

export function validateExistingDirectoryWithinRepo(pathValue: string, label: string): string | undefined {
  const result = resolveExistingDirectoryWithinRepo(pathValue, label);
  return result.ok ? undefined : result.error;
}

export function resolveDirectoryPathWithinRepo(pathValue: string, label: string): ValidatedRepoPathResult {
  try {
    const repoRoot = getRepoRoot();
    const resolvedPath = path.resolve(pathValue);

    if (existsSync(resolvedPath)) {
      const resolvedExistingPath = realpathSync(resolvedPath);
      if (!isPathWithinRepo(repoRoot, resolvedExistingPath)) {
        return { ok: false, error: buildOutsideRepoError(label, pathValue, resolvedExistingPath) };
      }
      const directoryError = ensureDirectoryStat(resolvedPath, label);
      if (directoryError) {
        return { ok: false, error: directoryError };
      }
      return { ok: true, path: resolvedExistingPath };
    }

    let nearestExistingPath = resolvedPath;
    while (!existsSync(nearestExistingPath)) {
      const parentPath = path.dirname(nearestExistingPath);
      if (parentPath === nearestExistingPath) {
        return { ok: false, error: `ERROR: invalid ${label} path: ${pathValue} (unable to resolve parent directory)` };
      }
      nearestExistingPath = parentPath;
    }

    const resolvedExistingAncestor = realpathSync(nearestExistingPath);
    if (!isPathWithinRepo(repoRoot, resolvedExistingAncestor)) {
      return {
        ok: false,
        error: buildOutsideRepoError(label, pathValue, resolvedExistingAncestor),
      };
    }

    const directoryError = ensureDirectoryStat(nearestExistingPath, label);
    if (directoryError) {
      return { ok: false, error: directoryError };
    }

    const relativeSuffix = path.relative(nearestExistingPath, resolvedPath);
    const canonicalPath = path.join(resolvedExistingAncestor, relativeSuffix);
    if (!isPathWithinRepo(repoRoot, canonicalPath)) {
      return { ok: false, error: buildOutsideRepoError(label, pathValue, canonicalPath) };
    }

    return { ok: true, path: canonicalPath };
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, error: `ERROR: invalid ${label} path: ${pathValue} (${message})` };
  }
}

export function validateDirectoryPathWithinRepo(pathValue: string, label: string): string | undefined {
  const result = resolveDirectoryPathWithinRepo(pathValue, label);
  return result.ok ? undefined : result.error;
}

export function buildCppLintDiagnostics(params: {
  sourceDir: string;
  buildDir?: string;
  linters: string[];
  timeout: number;
  command: (string | number)[];
  autoFix?: boolean;
}): string {
  const lines = [
    `- sourceDir=${sanitizeDiagnosticValue(params.sourceDir)}`,
    `- buildDir=${sanitizeDiagnosticValue(params.buildDir ?? "(none)")}`,
    `- linters=${sanitizeDiagnosticValue(params.linters.join(","))}`,
  ];
  if (params.autoFix !== undefined) {
    lines.push(`- autoFix=${String(params.autoFix)}`);
  }
  lines.push(`- timeout=${String(params.timeout)}`);
  lines.push(
    `- command=${sanitizeDiagnosticValue(params.command.map((part) => String(part)).join(" "))}`,
  );

  return [
    "Diagnostics:",
    ...lines,
    "",
    "Suggestions:",
    "- Verify sourceDir/buildDir paths.",
    "- Ensure required linters are installed and on PATH.",
    "- Confirm backing script exists and is executable.",
  ].join("\n");
}
