import { tool } from "@opencode-ai/plugin";
import {
  ALLOWED_LANGUAGES,
  buildAstgrepCommand,
  classifyAstgrepFailure,
  MISSING_BINARY_HINT,
  normalizeAstgrepArgs,
  resolveAstgrepTarget,
  runAstgrepCommand,
  sanitizeDiagnostic,
  sanitizePreviewOutput,
  selectDiagnostic,
} from "./lib/refactor_astgrep_shared";

export type AstgrepPreviewResult =
  | { kind: "match"; preview: string; previewLineCount: number; truncated: boolean }
  | { kind: "no_match" }
  | { kind: "parse_error"; diagnostic: string; hint?: string }
  | { kind: "unavailable"; diagnostic: string; hint?: string }
  | { kind: "runtime_error"; diagnostic: string; hint?: string };
const failure = (
  kind: "parse_error" | "unavailable" | "runtime_error",
  diagnostic: string,
  hint?: string,
): AstgrepPreviewResult => ({
  kind,
  diagnostic: sanitizeDiagnostic(diagnostic).replace(/^ERROR:\s*/i, ""),
  ...(hint ? { hint: sanitizeDiagnostic(hint) } : {}),
});

export default tool({
  description: "Preview AST-aware refactors without mutation. Returns a bounded typed result object.",
  args: { pattern: tool.schema.string(), rewrite: tool.schema.string(), lang: tool.schema.enum(ALLOWED_LANGUAGES), path: tool.schema.string().optional() },
  async execute(args): Promise<AstgrepPreviewResult> {
    const normalized = normalizeAstgrepArgs(args as Record<string, unknown>);
    if (typeof normalized === "string") return failure("runtime_error", normalized);
    const target = await resolveAstgrepTarget(normalized.path);
    if (!target.ok) return failure("runtime_error", target.diagnostic);
    try {
      const output = await runAstgrepCommand(buildAstgrepCommand(normalized, target.target));
      if (!output.trim()) return { kind: "no_match" };
       const allLines = output.split(/\r?\n/).filter((line) => line.trim());
       const lines = allLines.slice(0, 200);
      const rawPreview = lines.join("\n");
      const preview = sanitizePreviewOutput(rawPreview, 16_000);
      return {
        kind: "match",
        preview,
        previewLineCount: lines.length,
        truncated:
           lines.length < allLines.length || rawPreview.length > 16_000,
      };
    } catch (error) {
      const classification = classifyAstgrepFailure(error);
      return classification === "missing_binary" ? failure("unavailable", selectDiagnostic(error), MISSING_BINARY_HINT) : classification === "parse_input" ? failure("parse_error", selectDiagnostic(error), "Fix the ast-grep pattern/rewrite input and retry.") : failure("runtime_error", selectDiagnostic(error));
    }
  },
});
