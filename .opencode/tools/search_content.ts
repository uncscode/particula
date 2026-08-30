import { tool } from "@opencode-ai/plugin";
import * as path from "node:path";
import { buildTruncationWarning, executeRipgrepSearch, parseRipgrepSearchRequest, resolveValidatedSearchPath } from "./lib/ripgrep_shared";

export default tool({
  description:
    "Search file content using bounded ripgrep options. Matching is literal by default; use match-mode=regex to opt in." +
    " The optional path is the sole filesystem target selector: absolute values are normalized, relative values resolve from process.cwd()," +
    " and omission selects process.cwd(). The canonical target must remain under the repository rooted at that cwd." +
    " Workflow-local searches require the process already be launched in the resolved workflow worktree; then use its absolute worktree_path" +
    " or a child path. An absolute sibling-worktree path is not an authority switch and is rejected when outside the cwd-rooted repository.",
  args: { contentPattern: tool.schema.string(), path: tool.schema.string().optional(), options: tool.schema.string().optional() },
  async execute(args) {
    const directArgs = args as Record<string, unknown>;
    const advancedOnlyFields = [
      "contextLines", "beforeContext", "afterContext", "filesWithMatches",
      "filesWithoutMatches", "unrestricted", "ignoreGitignore", "includeHidden",
    ];
    const unsupportedField = advancedOnlyFields.find((field) => Object.hasOwn(directArgs, field));
    if (unsupportedField) {
      return `ERROR: '${unsupportedField}' is only supported by ripgrep_advanced; use its options field instead.`;
    }
    const parsed = parseRipgrepSearchRequest(directArgs.contentPattern, directArgs.options, false);
    if (!parsed.ok) return parsed.error;
    const cwd = process.cwd();
    const input = args.path ? (path.isAbsolute(args.path) ? path.normalize(args.path) : path.resolve(cwd, args.path)) : cwd;
    const target = await resolveValidatedSearchPath(input, cwd);
    if (target.error) return target.error;
    const request = parsed.request;
    const result = await executeRipgrepSearch({ ...request, searchPath: target.canonicalPath!, targetKind: target.targetKind, compactOutputBase: request.compactOutput ? target.compactOutputBase : undefined });
    if (result.errorMessage) return result.errorMessage;
    const lines = result.rawLines ?? [];
    if (!lines.length) return `No matches found for contentPattern '${request.contentPattern}'${args.path ? ` in '${args.path}'` : ""}.`;
    const output = lines.slice(0, request.maxResults).join("\n");
    const resultLimitWarning = lines.length > request.maxResults
      ? `\n\n${buildTruncationWarning(request.maxResults, lines.length, "lines", { approximateTotal: true })}`
      : "";
    const safetyWarning = result.outputClipped
      ? "\n\n[WARNING: Ripgrep stdout was clipped for safety. Narrow the search path or pattern and try again.]"
      : "";
    return `${output}${resultLimitWarning}${safetyWarning}`;
  },
});
