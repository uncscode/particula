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

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cpp_linters.py (dependency #1365).";
const DEFAULT_TIMEOUT_SECONDS = 300;
const MIN_TIMEOUT_SECONDS = 1;
const MAX_TIMEOUT_SECONDS = 3_600;
const SUPPORTED_LINTERS = ["clang-format", "clang-tidy", "cppcheck"] as const;
const SUPPORTED_LINTER_SET = new Set<string>(SUPPORTED_LINTERS);

function buildDiagnostics(params: {
  sourceDir: string;
  buildDir?: string;
  linters: string[];
  timeout: number;
  command: (string | number)[];
}): string {
  const lines = [
    `- sourceDir=${sanitizeDiagnosticValue(params.sourceDir)}`,
    `- buildDir=${sanitizeDiagnosticValue(params.buildDir ?? "(none)")}`,
    `- linters=${sanitizeDiagnosticValue(params.linters.join(","))}`,
    `- timeout=${String(params.timeout)}`,
    `- command=${sanitizeDiagnosticValue(params.command.map((part) => String(part)).join(" "))}`,
  ];
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

// --- Tool definition ---

export default tool({
  description: `Run C++ lint checks (clang-format, clang-tidy, cppcheck) without applying fixes.

EXAMPLES:
- Check all linters: run_cpp_lint_check({ sourceDir: 'example_cpp_dev' })
- Format-only check: run_cpp_lint_check({ sourceDir: 'src', linters: ['clang-format'] })
- clang-tidy check: run_cpp_lint_check({ sourceDir: 'src', buildDir: 'build', linters: ['clang-tidy'] })

IMPORTANT:
- Non-mutating by design: this wrapper never appends --auto-fix
- clang-tidy requires compile_commands.json in buildDir`,
  args: {
    outputMode: tool.schema.enum(["summary", "full", "json"]).optional(),
    sourceDir: tool.schema.string(),
    buildDir: tool.schema.string().optional(),
    linters: tool.schema.array(tool.schema.string()).optional(),
    timeout: tool.schema.number().optional(),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const sourceDir = args.sourceDir as string | undefined;
    const buildDir = args.buildDir as string | undefined;
    const linters = (args.linters as string[] | undefined) ?? [...SUPPORTED_LINTERS];
    const timeout = args.timeout ?? DEFAULT_TIMEOUT_SECONDS;

    const trimmedSourceDir = typeof sourceDir === "string" ? sourceDir.trim() : "";
    if (!trimmedSourceDir) {
      return "ERROR: sourceDir is required. Provide the directory containing C++ source files to lint.";
    }
    const sourceDirPathError = validatePathWithinRepo(trimmedSourceDir, "sourceDir");
    if (sourceDirPathError) {
      return sourceDirPathError;
    }
    if (typeof buildDir === "string" && buildDir.trim()) {
      const buildDirPathError = validatePathWithinRepo(buildDir.trim(), "buildDir");
      if (buildDirPathError) {
        return buildDirPathError;
      }
    }
    if (!Number.isInteger(timeout)) {
      return `ERROR: Timeout must be an integer in seconds (received ${timeout}).`;
    }
    if (timeout < MIN_TIMEOUT_SECONDS || timeout > MAX_TIMEOUT_SECONDS) {
      return `ERROR: Timeout must be between ${MIN_TIMEOUT_SECONDS} and ${MAX_TIMEOUT_SECONDS} seconds (received ${timeout}).`;
    }
    if (!Array.isArray(linters) || linters.length === 0) {
      return "ERROR: linters must be a non-empty array. Valid values: clang-format, clang-tidy, cppcheck.";
    }
    const invalidLinters = linters.filter((linter) => !SUPPORTED_LINTER_SET.has(linter));
    if (invalidLinters.length > 0) {
      return `ERROR: Unsupported linter(s): ${invalidLinters.join(", ")}. Valid values: ${SUPPORTED_LINTERS.join(", ")}.`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_cpp_linters.py`,
      `--source-dir=${trimmedSourceDir}`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
      `--linters=${linters.join(",")}`,
    ];
    const trimmedBuildDir = typeof buildDir === "string" ? buildDir.trim() : "";
    if (trimmedBuildDir) {
      cmdParts.push(`--build-dir=${trimmedBuildDir}`);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ lint check completed but returned no output.";
    } catch (error: any) {
      const stdout = sanitizeDiagnosticValue(error?.stdout?.toString?.() || "");
      const stderr = sanitizeDiagnosticValue(error?.stderr?.toString?.() || "");
      const message = sanitizeDiagnosticValue(error?.message || "Unknown error");
      const combinedLower = `${stderr} ${stdout} ${message}`.toLowerCase();

      if (stderr.trim() || stdout.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        const outputBlock = [
          "ERROR: C++ lint check failed",
          "",
          "STDERR:",
          stderr || "(empty)",
          "",
          "STDOUT:",
          stdout || "(empty)",
        ].join("\n");
        return `${outputBlock}${hint}`;
      }

      const diagnostics = buildDiagnostics({
        sourceDir: trimmedSourceDir,
        buildDir: trimmedBuildDir || undefined,
        linters,
        timeout,
        command: cmdParts,
      });
      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run C++ lint check: ${message}\n${MISSING_SCRIPT_HINT}\n\n${diagnostics}`;
      }
      return `ERROR: Failed to run C++ lint check: ${message}\n\n${diagnostics}`;
    }
  },
});
