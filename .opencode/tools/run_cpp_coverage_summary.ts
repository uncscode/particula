import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

// --- Inlined from lib/cpp_lint_wrapper_shared.ts ---

const MAX_DIAGNOSTIC_LENGTH = 4_000;

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

type RedactionRule = {
  pattern: RegExp;
  redact: (...groups: string[]) => string;
};

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

function sanitizeDiagnosticValue(value: string): string {
  const sanitized = redactSecrets(escapeControlChars(value));
  if (sanitized.length <= MAX_DIAGNOSTIC_LENGTH) {
    return sanitized;
  }
  return `${sanitized.slice(0, MAX_DIAGNOSTIC_LENGTH)}... [truncated]`;
}

function validatePathWithinRepo(pathValue: string, label: "sourceDir" | "buildDir"): string | undefined {
  try {
    if (!existsSync(pathValue)) {
      return `ERROR: ${label} path does not exist: ${pathValue}`;
    }
    if (!isStatDirectory(statSync(pathValue))) {
      return `ERROR: ${label} path is not a directory: ${pathValue}`;
    }

    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = realpathSync(pathValue);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: ${label} path resolves outside repository root: ${pathValue} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid ${label} path: ${pathValue} (${message})`;
  }

  return undefined;
}

// --- Tool-local helpers ---

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cpp_coverage.py.";
const MIN_TIMEOUT_SECONDS = 1;
const MAX_TIMEOUT_SECONDS = 3_600;
const MIN_THRESHOLD = 0;
const MAX_THRESHOLD = 100;
const FORBIDDEN_ADVANCED_KEYS = ["tool", "filter", "html", "extraArgs"] as const;

function validateThreshold(threshold: unknown): string | undefined {
  if (threshold === undefined) {
    return undefined;
  }
  if (typeof threshold !== "number" || !Number.isFinite(threshold)) {
    return "ERROR: threshold must be a finite number between 0 and 100.";
  }
  if (threshold < MIN_THRESHOLD || threshold > MAX_THRESHOLD) {
    return "ERROR: threshold must be between 0 and 100.";
  }
  return undefined;
}

function validateTimeout(timeout: unknown): string | undefined {
  if (timeout === undefined) {
    return undefined;
  }
  if (!Number.isInteger(timeout)) {
    return "ERROR: timeout must be an integer between 1 and 3600 seconds.";
  }
  if (timeout < MIN_TIMEOUT_SECONDS || timeout > MAX_TIMEOUT_SECONDS) {
    return "ERROR: timeout must be between 1 and 3600 seconds.";
  }
  return undefined;
}

// --- Tool definition ---

export default tool({
  description: `Run routine C++ coverage summary checks (build dir + threshold only).

Use this summary wrapper for standard coverage reporting.
Advanced options (tool/filter/html/extraArgs) are intentionally blocked here; use run_cpp_coverage_advanced for those controls.`,
  args: {
    outputMode: tool.schema.enum(["summary", "full", "json"]).optional(),
    buildDir: tool.schema.string(),
    threshold: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    tool: tool.schema.string().optional(),
    filter: tool.schema.string().optional(),
    html: tool.schema.string().optional(),
    extraArgs: tool.schema.array(tool.schema.string()).optional(),
  },
  async execute(args) {
    for (const key of FORBIDDEN_ADVANCED_KEYS) {
      if (Object.hasOwn(args, key)) {
        return `ERROR: run_cpp_coverage_summary does not accept advanced option '${key}'. Use run_cpp_coverage_advanced.`;
      }
    }

    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : "";
    if (!buildDir) {
      return "ERROR: buildDir is required. Provide a build directory containing coverage artifacts.";
    }
    const pathError = validatePathWithinRepo(buildDir, "buildDir");
    if (pathError) {
      return pathError;
    }

    const thresholdError = validateThreshold(args.threshold);
    if (thresholdError) {
      return thresholdError;
    }
    const timeoutError = validateTimeout(args.timeout);
    if (timeoutError) {
      return timeoutError;
    }

    const outputMode = args.outputMode || "summary";
    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_cpp_coverage.py`,
      `--build-dir=${buildDir}`,
      `--output=${outputMode}`,
    ];

    if (args.threshold !== undefined) {
      cmdParts.push("--threshold", Number(args.threshold));
    }
    if (args.timeout !== undefined) {
      cmdParts.push("--timeout", Number(args.timeout));
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ coverage summary completed but returned no output.";
    } catch (error: any) {
      const stdout = sanitizeDiagnosticValue(error?.stdout?.toString?.() || "");
      const stderr = sanitizeDiagnosticValue(error?.stderr?.toString?.() || "");
      const message = sanitizeDiagnosticValue(error?.message || "Unknown error");
      const combinedLower = `${stderr} ${stdout} ${message}`.toLowerCase();

      if (stderr.trim() || stdout.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        return [
          "ERROR: C++ coverage summary failed",
          "",
          "STDERR:",
          stderr || "(empty)",
          "",
          "STDOUT:",
          stdout || "(empty)",
        ].join("\n") + hint;
      }

      const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
      return `ERROR: Failed to run C++ coverage summary: ${message}${hint}`;
    }
  },
});
