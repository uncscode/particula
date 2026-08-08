import { tool } from "@opencode-ai/plugin";
import * as path from "node:path";
import { buildTruncationWarning, executeRipgrepSearch, parseRipgrepSearchRequest, resolveValidatedSearchPath } from "./lib/ripgrep_shared";

const DIRECT_CONTROL_FIELDS = [
  "pattern", "fileType", "excludeFileType", "globCaseInsensitive", "compactOutput",
  "maxResults", "maxMatchesPerFile", "matchMode", "contextLines", "beforeContext",
  "afterContext", "filesWithMatches", "filesWithoutMatches", "unrestricted",
  "ignoreGitignore", "includeHidden",
] as const;

export default tool({
  description: "Search file content using bounded advanced ripgrep controls. Matching is literal by default; use match-mode=regex to opt in.",
  args: { contentPattern: tool.schema.string(), path: tool.schema.string().optional(), options: tool.schema.string().optional() },
  async execute(args) {
    const rawArgs = args as Record<string, unknown>;
    const directControl = DIRECT_CONTROL_FIELDS.find((field) => Object.hasOwn(rawArgs, field));
    if (directControl) {
      return `ERROR: Direct search control field '${directControl}' is not allowed. Use the bounded 'options' string instead.`;
    }
    const parsed = parseRipgrepSearchRequest(rawArgs.contentPattern, rawArgs.options, true);
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
