/**
 * MkDocs Validate-Only Tool
 */

import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

// --- Inlined from lib/diagnostics.ts ---

const CONTROL_CHARS_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WHITESPACE_COLLAPSE_PATTERN = /\s+/g;

const REDACTION_PATTERNS: RegExp[] = [
  /\bgh[pousr]_[A-Za-z0-9]{10,}\b/g,
  /\b(?:token|api[_-]?key|secret|password)\s*[:=]\s*([^\s"']+)/gi,
  /\bAuthorization\s*:\s*Bearer\s+([^\s"']+)/gi,
];

const REDACTION_MARKER = "[REDACTED]";

function redactSensitiveFragments(value: string): string {
  let redacted = value;
  redacted = redacted.replace(REDACTION_PATTERNS[0], REDACTION_MARKER);
  redacted = redacted.replace(REDACTION_PATTERNS[1], (match) => {
    const splitIndex = match.indexOf(":") >= 0 ? match.indexOf(":") : match.indexOf("=");
    if (splitIndex < 0) {
      return REDACTION_MARKER;
    }
    return `${match.slice(0, splitIndex + 1)} ${REDACTION_MARKER}`;
  });
  redacted = redacted.replace(REDACTION_PATTERNS[2], `Authorization: Bearer ${REDACTION_MARKER}`);
  return redacted;
}

function sanitizeDiagnosticText(value: string): string {
  if (!value) {
    return "";
  }
  const noControlChars = value.replace(CONTROL_CHARS_PATTERN, " ");
  const redacted = redactSensitiveFragments(noControlChars);
  const normalized = redacted.replace(WHITESPACE_COLLAPSE_PATTERN, " ").trim();
  if (!normalized) {
    return "";
  }
  return normalized;
}

// --- Inlined from lib/cpp_lint_wrapper_shared.ts (isStatDirectory only) ---

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

// --- Inlined from lib/build_mkdocs_shared.ts ---

const MISSING_SCRIPT_HINT =
  "Encountered an ENOENT error. Ensure python3 is installed and on your PATH, mkdocs is " +
  "installed, and the backing script .opencode/tools/build_mkdocs.py exists.";

const MAX_DIAGNOSTIC_CHARS = 4000;
const DIAGNOSTIC_TRUNCATION_MARKER = "... [truncated]";
type OutputMode = "summary" | "full" | "json";
type ParsedMkdocsOptions = {
  outputMode?: OutputMode;
  strict?: true;
  clean?: boolean;
};
const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

const sanitizeAndClipDiagnostic = (value: string): string => {
  const sanitized = sanitizeDiagnosticText(value);
  if (sanitized.length <= MAX_DIAGNOSTIC_CHARS) {
    return sanitized;
  }
  return `${sanitized.slice(0, MAX_DIAGNOSTIC_CHARS)}\n${DIAGNOSTIC_TRUNCATION_MARKER}`;
};

function validateTimeout(timeout: number): string | null {
  if (!Number.isFinite(timeout) || timeout <= 0) {
    return `ERROR: Timeout must be a finite positive number (received ${timeout}).`;
  }
  return null;
}

function extractTimeoutDiagnostic(stderr: string, message: string): string | null {
  const timeoutMatch =
    stderr.match(/timed out after (\d+(?:\.\d+)?) seconds/i) ??
    message.match(/timed out after (\d+(?:\.\d+)?) seconds/i);
  if (!timeoutMatch) {
    return null;
  }
  const seconds = timeoutMatch[1] ?? "unknown";
  return (
    "ERROR: MkDocs validation timed out\n\n" +
    `diagnostic: mkdocs validation exceeded the wrapper timeout after ${seconds} seconds\n` +
    `hint: build_mkdocs_validate defaults to 120 seconds; pass a larger direct timeout for slow validations.`
  );
}

function formatExecutionError(error: any): string {
  const stdout = sanitizeAndClipDiagnostic(error?.stdout?.toString?.() || "");
  const stderr = sanitizeAndClipDiagnostic(error?.stderr?.toString?.() || "");
  const message = error?.message || "Unknown error";
  const combinedLower = `${stderr} ${message}`.toLowerCase();
  const timeoutDiagnostic = extractTimeoutDiagnostic(stderr, message);

  if (timeoutDiagnostic) {
    return timeoutDiagnostic;
  }

  if (stdout.trim()) {
    return stdout;
  }

  if (stderr.trim()) {
    const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
    return `ERROR: MkDocs build failed\n\n${stderr}${hint}`;
  }

  if (combinedLower.includes("enoent")) {
    return `ERROR: Failed to run mkdocs build: ${message}\n${MISSING_SCRIPT_HINT}`;
  }

  return `ERROR: Failed to run mkdocs build: ${message}`;
}

function validatePathWithinRepoRoot(
  value: string | undefined,
  parameterName: "cwd" | "configFile",
  cwdValue?: string,
): string | undefined {
  if (!value) {
    return undefined;
  }

  try {
    if (parameterName === "cwd") {
      if (!existsSync(value)) {
        return `ERROR: cwd path does not exist: ${value}`;
      }
      if (!isStatDirectory(statSync(value))) {
        return `ERROR: cwd path is not a directory: ${value}`;
      }
    }

    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = parameterName === "configFile" && cwdValue && !path.isAbsolute(value)
      ? realpathSync(path.resolve(realpathSync(cwdValue), value))
      : realpathSync(value);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: ${parameterName} path resolves outside repository root: ${value} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid ${parameterName} path: ${value} (${message})`;
  }

  return undefined;
}

async function executeMkdocsWrapper(cmdParts: (string | number)[]): Promise<string> {
  try {
    const result = await Bun.$`${cmdParts}`.text();
    return result || "mkdocs build completed but returned no output.";
  } catch (error: any) {
    return formatExecutionError(error);
  }
}

function parseMkdocsOptions(rawOptions: unknown):
  | { ok: true; options: ParsedMkdocsOptions }
  | { ok: false; error: string } {
  if (rawOptions === undefined || rawOptions === null) {
    return { ok: true, options: {} };
  }
  if (typeof rawOptions !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalized = rawOptions.trim();
  if (!normalized) {
    return { ok: true, options: {} };
  }

  const parsed: ParsedMkdocsOptions = {};
  for (const token of normalized.split(/\s+/)) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }

    if (separatorIndex === -1) {
      if (token !== "strict") {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
        };
      }
      if (parsed.strict) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.strict = true;
      continue;
    }

    const name = token.slice(0, separatorIndex);
    const value = token.slice(separatorIndex + 1).trim();
    if (!value) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    if (name === "output") {
      if (!OUTPUT_MODES.has(value as OutputMode)) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': output must be one of summary, full, json.`,
        };
      }
      if (parsed.outputMode !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.outputMode = value as OutputMode;
      continue;
    }

    if (name === "clean") {
      if (parsed.clean !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      if (value !== "true" && value !== "false") {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': clean must be true or false.`,
        };
      }
      parsed.clean = value === "true";
      continue;
    }

    if (name === "strict") {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token does not accept a value.`,
      };
    }

    return {
      ok: false,
      error: `ERROR: Invalid options token '${token}': token is not supported.`,
    };
  }

  return { ok: true, options: parsed };
}

// --- Tool definition ---

export default tool({
  description: `Validate docs with mkdocs without persisting build artifacts.

EXAMPLES:
- Default validate: build_mkdocs_validate({})
- Strict validate: build_mkdocs_validate({ options: 'strict' })
- Custom config: build_mkdocs_validate({ configFile: 'docs/mkdocs.yml' })

  IMPORTANT:
  - Always runs validate-only mode (temporary output dir).
  - Uses python3 to run the backing script.
  - Default timeout is 120 seconds.
  - The backing Python script supports longer runtimes, but this wrapper intentionally keeps a shorter validation default.
  - Passing 'strict' escalates MkDocs warnings into failure behavior.`,
  args: {
    timeout: tool.schema.number().optional(),
    cwd: tool.schema.string().optional(),
    configFile: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    const parsedOptions = parseMkdocsOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const timeout = args.timeout ?? 120;
    const timeoutError = validateTimeout(timeout);
    if (timeoutError) {
      return timeoutError;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/build_mkdocs.py`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
      "--validate-only",
    ];

    const cwdRaw = typeof args.cwd === "string" ? args.cwd.trim() : "";
    const configRaw = typeof args.configFile === "string" ? args.configFile.trim() : "";

    const cwdError = validatePathWithinRepoRoot(cwdRaw || undefined, "cwd");
    if (cwdError) {
      return cwdError;
    }
    const configError = validatePathWithinRepoRoot(configRaw || undefined, "configFile", cwdRaw || undefined);
    if (configError) {
      return configError;
    }

    if (cwdRaw) {
      cmdParts.push(`--cwd=${cwdRaw}`);
    }
    if (parsedOptions.options.strict === true) {
      cmdParts.push("--strict");
    }
    if (parsedOptions.options.clean === false) {
      cmdParts.push("--no-clean");
    }
    if (configRaw) {
      cmdParts.push(`--config-file=${configRaw}`);
    }

    return executeMkdocsWrapper(cmdParts);
  },
});
