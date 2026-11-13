import { tool } from "@opencode-ai/plugin";
import { ALLOWED_LANGUAGES, buildAstgrepCommand, classifyAstgrepFailure, MISSING_BINARY_HINT, normalizeAstgrepArgs, openAstgrepApplyTarget, runAstgrepCommand, selectDiagnostic } from "./lib/refactor_astgrep_shared";

export default tool({
  description: "Apply AST-aware refactors using ast-grep. This mutating wrapper always includes --update-all.",
  args: { pattern: tool.schema.string(), rewrite: tool.schema.string(), lang: tool.schema.enum(ALLOWED_LANGUAGES), path: tool.schema.string().optional() },
  async execute(args) {
    const normalized = normalizeAstgrepArgs(args as Record<string, unknown>);
    if (typeof normalized === "string") return normalized;
    const target = await openAstgrepApplyTarget(normalized.path);
    if (!target.ok) return `ERROR: Failed to execute 'ast-grep apply'.\ndiagnostic: ${target.diagnostic}`;
    try { const output = await runAstgrepCommand(buildAstgrepCommand(normalized, target.target, true)); return output.trim() || "No files modified (no matches)."; }
    catch (error) {
      const type = classifyAstgrepFailure(error);
      const hint = type === "missing_binary" ? MISSING_BINARY_HINT : type === "parse_input" ? "Check ast-grep pattern/rewrite syntax and retry with a valid AST pattern." : "Apply mode may have partially modified files. Inspect `git diff`, restore affected paths, then retry.";
      return `ERROR: Failed to execute 'ast-grep apply'.\nclassification: ${type === "missing_binary" ? "missing_binary" : type === "parse_input" ? "invalid_pattern" : "execution"}\ndiagnostic: ${selectDiagnostic(error)}\nhint: ${hint}`;
    } finally { await target.close(); }
  },
});
