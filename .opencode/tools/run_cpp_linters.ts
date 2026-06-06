/**
 * C++ Linter Runner Tool
 *
 * Wraps the Python backing script run_cpp_linters.py to execute clang-format,
 * clang-tidy, and cppcheck with optional auto-fixing and structured output.
 * Mirrors run_ctest.ts/run_linters.ts patterns for consistent OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cpp_linters.py (dependency #1365).";
const DEFAULT_TIMEOUT_SECONDS = 300;
const MIN_TIMEOUT_SECONDS = 1;
const MAX_TIMEOUT_SECONDS = 3_600;
const SUPPORTED_LINTERS = ["clang-format", "clang-tidy", "cppcheck"] as const;
const SUPPORTED_LINTER_SET = new Set<string>(SUPPORTED_LINTERS);

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
  return redactSecrets(escapeControlChars(value));
}

function buildDiagnostics(params: {
  sourceDir: string;
  buildDir?: string;
  linters: string[];
  autoFix: boolean;
  timeout: number;
  command: (string | number)[];
}): string {
  const lines = [
    `- sourceDir=${sanitizeDiagnosticValue(params.sourceDir)}`,
    `- buildDir=${sanitizeDiagnosticValue(params.buildDir ?? "(none)")}`,
    `- linters=${sanitizeDiagnosticValue(params.linters.join(","))}`,
    `- autoFix=${String(params.autoFix)}`,
    `- timeout=${String(params.timeout)}`,
    `- command=${sanitizeDiagnosticValue(params.command.map((part) => String(part)).join(" "))}`,
  ];

  return ["Diagnostics:", ...lines, "", "Suggestions:", "- Verify sourceDir/buildDir paths.", "- Ensure required linters are installed and on PATH.", "- Confirm backing script exists and is executable."].join("\n");
}

export default tool({
  description: `Run C++ linters (clang-format, clang-tidy, cppcheck) with optional auto-fix.

EXAMPLES:
- Check all linters: run_cpp_linters({ sourceDir: 'example_cpp_dev' })
- Format only: run_cpp_linters({ sourceDir: 'src', linters: ['clang-format'] })
- Auto-fix formatting: run_cpp_linters({ sourceDir: 'src', autoFix: true })
- clang-tidy (requires compile_commands.json): run_cpp_linters({ sourceDir: 'src', buildDir: 'build', linters: ['clang-tidy'] })
- JSON output: run_cpp_linters({ sourceDir: 'src', outputMode: 'json' })

IMPORTANT:
- clang-tidy requires compile_commands.json in buildDir (cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
- Linters must be installed and on PATH; missing linters are skipped with a warning
- autoFix applies clang-format -i and clang-tidy --fix when available`,
  args: {
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe(
        "Output mode: 'summary' (default, human-readable), 'full' (complete linter output), 'json' (structured data).",
      ),
    sourceDir: tool.schema
      .string()
      .describe(
        "Directory containing C++ sources to lint (required). Example: 'src', 'example_cpp_dev/src'.",
      ),
    buildDir: tool.schema
      .string()
      .optional()
      .describe(
        "Build directory containing compile_commands.json (required for clang-tidy). Example: 'build', 'example_cpp_dev/build'.",
      ),
    linters: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe(
        "Specific linters to run: ['clang-format', 'clang-tidy', 'cppcheck']. Default: all available linters.",
      ),
    autoFix: tool.schema
      .boolean()
      .optional()
      .describe(
        "Automatically fix issues where possible (default: false). Applies clang-format -i and clang-tidy --fix when supported.",
      ),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds for all linters (default: 300). Must be positive."),
  },
  async execute(args) {
    const outputMode = args.outputMode || "summary";
    const sourceDir = args.sourceDir as string | undefined;
    const buildDir = args.buildDir as string | undefined;
    const linters = (args.linters as string[] | undefined) ?? [...SUPPORTED_LINTERS];
    const autoFix = args.autoFix || false;
    const timeout = args.timeout ?? DEFAULT_TIMEOUT_SECONDS;

    if (!sourceDir) {
      return "ERROR: sourceDir is required. Provide the directory containing C++ source files to lint.";
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
      `--source-dir=${sourceDir}`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
      `--linters=${linters.join(",")}`,
    ];

    if (buildDir) {
      cmdParts.push(`--build-dir=${buildDir}`);
    }

    if (autoFix) {
      cmdParts.push("--auto-fix");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ linters completed but returned no output.";
    } catch (error: any) {
      const stdout = error?.stdout?.toString?.() || "";
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";
      const combinedLower = `${stderr} ${message}`.toLowerCase();

      if (stderr.trim() || stdout.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        const outputBlock = [
          "ERROR: C++ linters failed",
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
        sourceDir,
        buildDir,
        linters,
        autoFix,
        timeout,
        command: cmdParts,
      });

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run C++ linters: ${message}\n${MISSING_SCRIPT_HINT}\n\n${diagnostics}`;
      }

      return `ERROR: Failed to run C++ linters: ${message}\n\n${diagnostics}`;
    }
  },
});
