/**
 * ast-grep Refactor Tool
 *
 * Wraps the ast-grep CLI for AST-aware find-and-rewrite with dry-run defaults.
 * Mirrors the OpenCode tool facade pattern used by run_cpp_linters.ts.
 */

import { tool } from "@opencode-ai/plugin";

const ALLOWED_LANGUAGES = [
  "python",
  "typescript",
  "javascript",
  "cpp",
  "c",
  "go",
  "java",
  "rust",
  "csharp",
  "kotlin",
  "swift",
  "ruby",
  "php",
] as const;

const MISSING_BINARY_HINT = "Install ast-grep-cli (pip install ast-grep-cli) and ensure ast-grep is on PATH.";

const DESCRIPTION = `Refactor code using ast-grep (AST-aware find-and-rewrite) with dry-run by default.

PATTERN SYNTAX (Meta Variables):
- $VAR: single AST node (expression, identifier, etc.) — like regex '.'
- $$$ARGS: zero or more AST nodes (arguments, statements, parameters) — like regex '.*'
- $_NAME: one node, non-capturing (each occurrence can differ)

EXAMPLES:
- Rename function (Python):
  ast-grep run -p 'old_name($$$ARGS)' -r 'new_name($$$ARGS)' -l python
  ast-grep run -p 'def old_name($$$PARAMS):' -r 'def new_name($$$PARAMS):' -l python
- Rename function (TypeScript):
  ast-grep run -p 'oldFunc($$$ARGS)' -r 'newFunc($$$ARGS)' -l typescript
  ast-grep run -p 'function oldFunc($$$PARAMS) { $$$BODY }' -r 'function newFunc($$$PARAMS) { $$$BODY }' -l typescript
- Rewrite API pattern:
  ast-grep run -p 'self.assertEqual($A, $B)' -r 'assert $A == $B' -l python

PATTERNS WILL NOT MATCH:
- Code inside comments
- Code inside string literals
- Partial identifiers (old_name won't match old_name_extended)

CLI FLAGS:
- --pattern / -p: AST pattern to match (required)
- --rewrite / -r: Replacement pattern using the same meta variables
- --lang / -l: Language (python, typescript, javascript, cpp, c, go, java, rust, etc.)
- --update-all / -U: Apply changes (default is dry-run preview)

EXAMPLES (TOOL USAGE):
- Dry-run (default): refactor_astgrep({ pattern: 'oldFunc($$$ARGS)', rewrite: 'newFunc($$$ARGS)', lang: 'typescript' })
- Apply changes: refactor_astgrep({ pattern: 'old_name($$$ARGS)', rewrite: 'new_name($$$ARGS)', lang: 'python', dryRun: false })`;

export default tool({
  description: DESCRIPTION,
  args: {
    pattern: tool.schema.string().describe("AST pattern to match (required)."),
    rewrite: tool.schema.string().describe("Replacement pattern using the same meta variables (required)."),
    lang: tool.schema
      .enum(ALLOWED_LANGUAGES)
      .describe("Language to parse (required). Examples: python, typescript, javascript, cpp, c, go, java, rust."),
    path: tool.schema
      .string()
      .optional()
      .describe("File or directory to search (default: '.')."),
    dryRun: tool.schema
      .boolean()
      .optional()
      .describe("Preview changes without applying (default: true). Set false to apply rewrites."),
  },
  async execute(args) {
    const pattern = args.pattern as string | undefined;
    const rewrite = args.rewrite as string | undefined;
    const lang = args.lang as (typeof ALLOWED_LANGUAGES)[number] | string | undefined;
    const path = (args.path as string | undefined) ?? ".";
    const dryRun = args.dryRun !== false;

    if (!pattern) {
      return "ERROR: pattern is required. Provide the AST pattern to match.";
    }

    if (!rewrite) {
      return "ERROR: rewrite is required. Provide the replacement pattern.";
    }

    if (!lang) {
      return `ERROR: lang is required. Choose one of: ${ALLOWED_LANGUAGES.join(", ")}.`;
    }

    if (!ALLOWED_LANGUAGES.includes(lang as any)) {
      return `ERROR: lang must be one of ${ALLOWED_LANGUAGES.join(", ")} (received ${lang}).`;
    }

    const cmdParts: (string | number)[] = [
      "ast-grep",
      "run",
      "-p",
      pattern,
      "-r",
      rewrite,
      "-l",
      lang,
      path,
    ];

    if (!dryRun) {
      cmdParts.push("--update-all");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      if (result && result.trim()) {
        return result;
      }

      if (dryRun) {
        return `No matches found for pattern: ${pattern}`;
      }

      return "No files modified (no matches).";
    } catch (error: any) {
      const stdout = error?.stdout?.toString?.() || "";
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";
      const exitCode = typeof error?.exitCode === "number" ? error.exitCode : "unknown";
      const stderrText = stderr.trim() ? stderr.trim() : "(empty)";
      const combinedLower = `${stderr} ${message}`.toLowerCase();

      if (stdout.trim()) {
        return stdout;
      }

      const missingBinary = combinedLower.includes("enoent") || combinedLower.includes("not found");
      const invalidPattern = combinedLower.includes("parse") || combinedLower.includes("pattern") || combinedLower.includes("rewrite");

      if (missingBinary) {
        return [
          "ERROR [MISSING_BINARY]: ast-grep CLI not found.",
          `Exit code: ${exitCode}`,
          `Stderr: ${stderrText}`,
          MISSING_BINARY_HINT,
        ].join("\n");
      }

      if (invalidPattern) {
        return [
          "ERROR [INVALID_PATTERN]: Invalid pattern/rewrite.",
          `Exit code: ${exitCode}`,
          `Stderr: ${stderrText}`,
        ].join("\n");
      }

      return [
        "ERROR [EXECUTION]: refactor_astgrep failed.",
        `Exit code: ${exitCode}`,
        `Stderr: ${stderrText}`,
      ].join("\n");
    }
  },
});
