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

function sanitizeDiagnosticText(value: string, maxChars = 4000): string {
  if (!value) {
    return "";
  }
  const noControlChars = value.replace(CONTROL_CHARS_PATTERN, " ");
  const redacted = redactSensitiveFragments(noControlChars);
  const normalized = redacted.replace(WHITESPACE_COLLAPSE_PATTERN, " ").trim();
  if (!normalized) {
    return "";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)} ...(truncated)`;
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

function formatExecutionError(error: any): string {
  const stdout = sanitizeAndClipDiagnostic(error?.stdout?.toString?.() || "");
  const stderr = sanitizeAndClipDiagnostic(error?.stderr?.toString?.() || "");
  const message = error?.message || "Unknown error";
  const combinedLower = `${stderr} ${message}`.toLowerCase();

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
    const resolvedPath = realpathSync(value);
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

// --- Tool definition ---

export default tool({
  description: `Validate docs with mkdocs without persisting build artifacts.

EXAMPLES:
- Default validate: build_mkdocs_validate({})
- Strict validate: build_mkdocs_validate({ strict: true })
- Custom config: build_mkdocs_validate({ configFile: 'docs/mkdocs.yml' })

IMPORTANT:
- Always runs validate-only mode (temporary output dir).
- Uses python3 to run the backing script.
- Default timeout is 120 seconds.`,
  args: {
    outputMode: tool.schema.enum(["summary", "full", "json"]).optional(),
    timeout: tool.schema.number().optional(),
    cwd: tool.schema.string().optional(),
    strict: tool.schema.boolean().optional(),
    clean: tool.schema.boolean().optional(),
    configFile: tool.schema.string().optional(),
  },
  async execute(args) {
    const outputMode = (args.outputMode as string | undefined) || "summary";
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
    const configError = validatePathWithinRepoRoot(configRaw || undefined, "configFile");
    if (configError) {
      return configError;
    }

    if (cwdRaw) {
      cmdParts.push(`--cwd=${cwdRaw}`);
    }
    if (args.strict === true) {
      cmdParts.push("--strict");
    }
    if (args.clean === false) {
      cmdParts.push("--no-clean");
    }
    if (configRaw) {
      cmdParts.push(`--config-file=${configRaw}`);
    }

    return executeMkdocsWrapper(cmdParts);
  },
});
