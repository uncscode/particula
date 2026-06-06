/**
 * Git Operations Tool for OpenCode Integration
 *
 * Exposes ADW git commands (commit, push, status, diff, add, restore) so
 * tool-only agents can run git inside isolated worktrees. Mirrors
 * adw/commands/git_cli.py semantics and follows existing OpenCode tool
 * patterns (see adw_spec.ts and adw.ts).
 */

import { tool } from "@opencode-ai/plugin";

// Strip VIRTUAL_ENV so uv resolves the active project environment in worktrees.
if (process.env.VIRTUAL_ENV !== undefined) {
  delete process.env.VIRTUAL_ENV;
}

export default tool({
  description: `Execute ADW git commands within isolated worktrees. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

  Commit:    { command: "commit", summary: "Add feature", stage_all: true, worktree_path: "./trees/abc" }
  Push:      { command: "push", branch: "feature-123", worktree_path: "./trees/abc" }
  Status:    { command: "status", porcelain: true, worktree_path: "./trees/abc" }
  Diff:      { command: "diff", stat: true, worktree_path: "./trees/abc" }
  Diff base: { command: "diff", base: "main", stat: true, worktree_path: "./trees/abc" }
  Log:       { command: "log", max_count: 5, oneline: true, worktree_path: "./trees/abc" }
  Add all:   { command: "add", stage_all: true, worktree_path: "./trees/abc" }
  Checkout:  { command: "checkout", branch: "feat-123", create: true, source: "origin/develop" }
  Merge:     { command: "merge", source: "main", target: "develop", worktree_path: "./trees/abc" }

RULES:
- Always set worktree_path when inside ADW-managed trees.
- Empty strings, empty arrays, false booleans, and noisy zero numeric defaults are treated as omitted.
- Sparse explicit false is still honored for force:false and abort_on_conflict:false.
- commit requires summary; push/checkout/rebase require branch; merge requires source.
- push-force-with-lease and checkout --create block main/master.
- Set help: true to view CLI help for any command.

See .opencode/tools/git_operations.md for full parameter reference and all commands.`
,

  args: {
    command: tool.schema
      .enum([
        "commit",
        "push",
        "checkout",
        "status",
        "diff",
        "log",
        "show",
        "add",
        "restore",
        "merge",
        "rebase",
        "fetch",
        "sync",
        "accumulate",
        "abort",
        "continue",
        "reset",
        "push-force-with-lease",
        "worktree-list",
        "worktree-remove",
        "worktree-prune",
      ])
      .describe(`Git command to execute. Set help: true to see CLI usage for any command.`),

    summary: tool.schema
      .string()
      .optional()
      .describe(`Commit message summary line (first line of commit message).

Required for: commit
Mapped flag: --summary

Examples:
• "Add git operations tool"
• "Fix lint warnings in auth module"`),

    description: tool.schema
      .string()
      .optional()
      .describe(`Extended commit message body (appears after summary line).

Applies to: commit
Mapped flag: --description

Example: { command: "commit", summary: "Add feature", description: "Implements the new auth flow with OAuth2 support" }`),

    adw_id: tool.schema
      .string()
      .optional()
      .describe(`ADW workflow ID appended to commit body as "ADW-ID: <id>" for traceability.

Applies to: commit
Mapped flag: --adw-id

Example: { command: "commit", summary: "Fix bug", adw_id: "a17d3f8a" }`),

    worktree_path: tool.schema
      .string()
      .optional()
      .describe(`Target worktree directory for git operations.

Recommended for all commands to ensure operations run in the correct isolated worktree.

Applies to: all commands
Mapped flag: --worktree-path

Example: "./trees/a17d3f8a"`),

    stage_all: tool.schema
      .boolean()
      .optional()
      .describe(`Stage all tracked and untracked changes.

Applies to: commit, add
• For commit: stages all changes before committing (maps to --stage-all)
• For add: stages all changes with git add -A (maps to --all)

Note: For add command, mutually exclusive with files parameter.

Examples:
• { command: "commit", summary: "Update deps", stage_all: true }
• { command: "add", stage_all: true }`),

    force: tool.schema
      .boolean()
      .optional()
      .describe(`Force removal of the worktree even when dirty.

Applies to: worktree-remove
Mapped flag: --force (default true)

Example: { command: "worktree-remove", adw_id: "abc12345", force: true }`),

    max_retries: tool.schema
      .number()
      .optional()
      .describe(`Maximum commit retry attempts when pre-commit hooks modify files.

When hooks (e.g., formatters, linters) modify staged files, the commit is
retried with the updated files. Default: 3

Applies to: commit
Mapped flag: --max-retries

Example: { command: "commit", summary: "Add feature", max_retries: 5 }`),

    no_verify: tool.schema
      .boolean()
      .optional()
      .describe(`Bypass pre-commit hooks for this commit.

Applies to: commit
Mapped flag: --no-verify

Use only when explicitly required.`),

    source: tool.schema
      .string()
      .optional()
      .describe(`Source branch/ref for merge, sync, or checkout create flow.

Required for: merge
Optional for: sync (default origin)
Optional for: checkout (requires create=true)
Mapped flags: merge uses positional <source>; sync/checkout use --source

Example: { command: "merge", source: "main" }`),

    create: tool.schema
      .boolean()
      .optional()
      .describe(`Create branch without changing the current checkout.

Applies to: checkout
Mapped flag: --create
Default: false

Example: { command: "checkout", branch: "feature-123", create: true }`),

    target: tool.schema
      .string()
      .optional()
      .describe(`Target branch for merge, sync, or diff (defaults to current branch or HEAD).

Applies to: merge (maps to --into), sync, diff
Mapped flag: --into (merge), --target (sync, diff)

Example: { command: "merge", source: "main", target: "develop" }`),

    no_ff: tool.schema
      .boolean()
      .optional()
      .describe(`Create a merge commit even when fast-forward is possible.

Applies to: merge
Mapped flag: --no-ff
Default: false

Example: { command: "merge", source: "main", no_ff: true }`),

    abort_on_conflict: tool.schema
      .boolean()
      .optional()
      .describe(`Abort merge/rebase when conflicts are detected.

Applies to: merge, rebase
Mapped flag: --abort-on-conflict (default true); when false adds --no-abort-on-conflict

Example: { command: "merge", source: "main", abort_on_conflict: false }`),

    branch: tool.schema
      .string()
      .optional()
      .describe(`Branch name for push, checkout, rebase target, fetch branch, sync target, reset branch refs, or force-with-lease.

Applies to: push, checkout, rebase, push-force-with-lease, fetch, sync, reset
Required for: push, checkout, rebase, push-force-with-lease
Optional for: fetch, sync, reset
Mapped flags/positionals:
  • push, checkout, push-force-with-lease: --branch <branch>
  • rebase: <branch> positional (rebase target)
  • fetch: fetches the specified <branch> from --remote
  • sync: syncs the specified <branch> with --remote
  • reset: branch whose refs are reset (tool-specific behavior)

For push, the named local branch is pushed directly and does not need to be
the current checkout.

Example: { command: "rebase", branch: "develop" }`),

    onto: tool.schema
      .string()
      .optional()
      .describe(`Base to reapply commits onto during rebase.

Applies to: rebase
Mapped flag: --onto

Example: { command: "rebase", branch: "feature", onto: "main" }`),

    remote: tool.schema
      .string()
      .optional()
      .describe(`Remote name to fetch from.

Applies to: fetch, sync
Default: origin when omitted
Mapped flag: --remote

Example: { command: "fetch", remote: "upstream" }`),

    slice_branch: tool.schema
      .string()
      .optional()
      .describe(`Slice branch to accumulate from.

Applies to: accumulate
Required for: accumulate
Mapped flag: --slice-branch

Example: { command: "accumulate", slice_branch: "issue-123-adw-abc" }`),

    tracking_branch: tool.schema
      .string()
      .optional()
      .describe(`Tracking branch to accumulate into.

Applies to: accumulate
Required for: accumulate
Mapped flag: --tracking-branch

Example: { command: "accumulate", tracking_branch: "feature/epic-x" }`),

    recover_missing_worktree: tool.schema
      .boolean()
      .optional()
      .describe(`Recover from a missing worktree path by using a clean existing worktree already checked out on the tracking branch.

Applies to: accumulate
Default: false
Mapped flag: --recover-missing-worktree

Example: { command: "accumulate", recover_missing_worktree: true }`),

    prune: tool.schema
      .boolean()
      .optional()
      .describe(`Prune tracking branches that no longer exist on the remote.

Applies to: fetch
Mapped flag: --prune
Default: false

Example: { command: "fetch", prune: true }`),

    ref: tool.schema
      .string()
      .optional()
      .describe(`Git ref to reset to or inspect.

Required for: reset, show
Optional for: log (default: current branch)
Mapped flag: --ref

Example: { command: "reset", ref: "HEAD~1" }`),

    hard: tool.schema
      .boolean()
      .optional()
      .describe(`Use --hard when resetting to discard working tree changes.

Applies to: reset
Mapped flag: --hard
Default: false

Example: { command: "reset", ref: "HEAD~1", hard: true }`),

    porcelain: tool.schema
      .boolean()
      .optional()
      .describe(`Return machine-readable porcelain output for status command.

Applies to: status
Mapped flag: --porcelain

Example: { command: "status", porcelain: true }`),

    stat: tool.schema
      .boolean()
      .optional()
      .describe(`Include file change statistics (insertions/deletions per file) in diff or show output.

Applies to: diff, show
Mapped flag: --stat

Example: { command: "diff", stat: true }`),

    base: tool.schema
      .string()
      .optional()
      .describe(`Base revision for three-dot diff comparison (base...HEAD).

Shows all changes on current branch since it diverged from the base revision.
Useful for reviewing PR changes against the target branch.

Applies to: diff
Mapped flag: --base

Examples:
• "main" - Compare against local main branch
• "origin/main" - Compare against remote main branch

Usage: { command: "diff", base: "main", stat: true }`),

    max_count: tool.schema
      .number()
      .optional()
      .describe(`Maximum number of commits to show in log output.

Applies to: log
Mapped flag: --max-count
Default: 10

Example: { command: "log", max_count: 5 }`),

    oneline: tool.schema
      .boolean()
      .optional()
      .describe(`Use --oneline format for log output.

Applies to: log
Mapped flag: --oneline
Default: false

Example: { command: "log", oneline: true }`),

    path: tool.schema
      .string()
      .optional()
      .describe(`File path to show at a ref.

Applies to: show
Mapped flag: --path

Example: { command: "show", ref: "main", path: "src/app.py" }`),

    files: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe(`Specific files to stage (add) or restore.

Applies to: add, restore

For add command:
• Mutually exclusive with stage_all (cannot use both)
• Required when stage_all is not set

For restore command:
• Optional; when omitted, restores all changes

Examples:
• { command: "add", files: ["src/main.py", "src/utils.py"] }
• { command: "restore", files: ["src/main.py"] }`),

    staged: tool.schema
      .boolean()
      .optional()
      .describe(`Unstage files from the index instead of discarding working tree changes.

Applies to: restore
Mapped flag: --staged

Example: { command: "restore", staged: true, files: ["src/main.py"] }`),

    help: tool.schema
      .boolean()
      .optional()
      .describe(`Show CLI help for the specified command.

When true, parameter validation is skipped and the tool returns the CLI
help text for the chosen command.

Example: { command: "diff", help: true }`),
  },

  async execute(args) {
    const cmdParts = ["uv", "run", "adw", "git"];
    const ERROR_SNIPPET_LIMIT = 500;
    const TRUNCATION_MARKER = "... [truncated]";

    const rawArgs = args as Record<string, unknown>;
    const optionalKeys = [
      "summary",
      "description",
      "adw_id",
      "worktree_path",
      "stage_all",
      "force",
      "max_retries",
      "no_verify",
      "source",
      "create",
      "target",
      "no_ff",
      "abort_on_conflict",
      "branch",
      "onto",
      "remote",
      "slice_branch",
      "tracking_branch",
      "recover_missing_worktree",
      "prune",
      "ref",
      "hard",
      "porcelain",
      "stat",
      "base",
      "max_count",
      "oneline",
      "path",
      "files",
      "staged",
      "help",
    ];

    const isBlankString = (value: unknown): boolean =>
      typeof value === "string" && value.trim() === "";

    const isEmptyArray = (value: unknown): boolean => Array.isArray(value) && value.length === 0;

    const isInertDefaultValue = (value: unknown): boolean =>
      value === false || value === 0 || isBlankString(value) || isEmptyArray(value);

    const isMeaningfulValue = (value: unknown): boolean =>
      value !== undefined && value !== null && !isInertDefaultValue(value);

    const inertDefaultCount = optionalKeys.filter(
      (key) => key in rawArgs && isInertDefaultValue(rawArgs[key]),
    ).length;
    const meaningfulValueCount = optionalKeys.filter(
      (key) => key in rawArgs && isMeaningfulValue(rawArgs[key]),
    ).length;
    const isNoisyDefaultCall = inertDefaultCount >= 8 && meaningfulValueCount <= 6;

    const normalizeOptionalString = (value: unknown): string | undefined => {
      if (value === undefined || value === null) {
        return undefined;
      }
      const text = String(value).trim();
      return text || undefined;
    };

    const normalizeOptionalBoolean = (value: unknown): boolean | unknown | undefined => {
      if (value === undefined || value === null || value === false) {
        return undefined;
      }
      if (value === true) {
        return true;
      }
      return value;
    };

    const normalizeFalseOverrideBoolean = (value: unknown): boolean | unknown | undefined => {
      if (value === undefined || value === null) {
        return undefined;
      }
      if (value === false && isNoisyDefaultCall) {
        return undefined;
      }
      if (value === true || value === false) {
        return value;
      }
      return value;
    };

    const normalizeOptionalNumber = (value: unknown): number | unknown | undefined => {
      if (value === undefined || value === null) {
        return undefined;
      }
      if (value === 0 && isNoisyDefaultCall) {
        return undefined;
      }
      return value;
    };

    const command = normalizeOptionalString(rawArgs.command);
    const summary = normalizeOptionalString(rawArgs.summary);
    const description = normalizeOptionalString(rawArgs.description);
    const adw_id = normalizeOptionalString(rawArgs.adw_id);
    const worktree_path = normalizeOptionalString(rawArgs.worktree_path);
    const stage_all = normalizeOptionalBoolean(rawArgs.stage_all);
    const force = normalizeFalseOverrideBoolean(rawArgs.force);
    const max_retries = normalizeOptionalNumber(rawArgs.max_retries);
    const no_verify = normalizeOptionalBoolean(rawArgs.no_verify);
    const source = normalizeOptionalString(rawArgs.source);
    const create = normalizeOptionalBoolean(rawArgs.create);
    const target = normalizeOptionalString(rawArgs.target);
    const no_ff = normalizeOptionalBoolean(rawArgs.no_ff);
    const abort_on_conflict = normalizeFalseOverrideBoolean(rawArgs.abort_on_conflict);
    const branch = normalizeOptionalString(rawArgs.branch);
    const onto = normalizeOptionalString(rawArgs.onto);
    const remote = normalizeOptionalString(rawArgs.remote);
    const slice_branch = normalizeOptionalString(rawArgs.slice_branch);
    const tracking_branch = normalizeOptionalString(rawArgs.tracking_branch);
    const recover_missing_worktree = normalizeOptionalBoolean(rawArgs.recover_missing_worktree);
    const prune = normalizeOptionalBoolean(rawArgs.prune);
    const ref = normalizeOptionalString(rawArgs.ref);
    const hard = normalizeOptionalBoolean(rawArgs.hard);
    const porcelain = normalizeOptionalBoolean(rawArgs.porcelain);
    const stat = normalizeOptionalBoolean(rawArgs.stat);
    const base = normalizeOptionalString(rawArgs.base);
    const max_count = normalizeOptionalNumber(rawArgs.max_count);
    const oneline = normalizeOptionalBoolean(rawArgs.oneline);
    const path = normalizeOptionalString(rawArgs.path);
    const files = Array.isArray(rawArgs.files)
      ? rawArgs.files.map((filePath) => String(filePath).trim()).filter(Boolean)
      : undefined;
    const staged = normalizeOptionalBoolean(rawArgs.staged);
    const help = normalizeOptionalBoolean(rawArgs.help);

    const normalizeSnippet = (value: unknown): string => {
      const raw = typeof value === "string" ? value : String(value ?? "");
      return raw.replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").trim();
    };

    const clipSnippet = (value: unknown): string => {
      const text = normalizeSnippet(value);
      if (!text) {
        return "";
      }
      return text.length > ERROR_SNIPPET_LIMIT
        ? `${text.slice(0, ERROR_SNIPPET_LIMIT)}${TRUNCATION_MARKER}`
        : text;
    };

    const getCommitHint = (stderr: string, stdout: string): string => {
      const combined = `${stderr}\n${stdout}`.toLowerCase();

      if (
        combined.includes("index.lock") ||
        combined.includes(".lock") ||
        combined.includes("unable to create") ||
        combined.includes("another git process") ||
        combined.includes("could not lock") ||
        combined.includes("lock file")
      ) {
        return "Git lock contention detected; remove stale lock files or wait for other git processes to finish.";
      }

      if (combined.includes("pre-commit") || combined.includes("hook")) {
        return "Commit was rejected by git hooks; review hook output and fix reported issues before retrying.";
      }

      if (combined.includes("nothing to commit") || combined.includes("nothing added")) {
        return "Working tree is clean; stage or modify files before committing.";
      }

      return "Inspect stderr/stdout details and rerun with corrected inputs or repository state.";
    };

    const buildCommitError = (error: any): string => {
      const stderr = error?.stderr ? error.stderr.toString() : "";
      const stdout = error?.stdout ? error.stdout.toString() : "";
      const fallbackMessage = typeof error?.message === "string" ? error.message : "";
      const exitCode =
        typeof error?.exitCode === "number"
          ? error.exitCode
          : typeof error?.code === "number"
            ? error.code
            : "unknown";

      const lines = [
        "ERROR: Failed to execute 'adw git commit'",
        "command: commit",
        `exit_code: ${exitCode}`,
      ];

      if (worktree_path && String(worktree_path).trim()) {
        lines.push(`worktree_path: ${String(worktree_path).trim()}`);
      }

      if (stderr) {
        lines.push(`stderr: ${clipSnippet(stderr)}`);
      }
      if (stdout) {
        lines.push(`stdout: ${clipSnippet(stdout)}`);
      }
      if (!stderr && !stdout && fallbackMessage) {
        lines.push(`message: ${clipSnippet(fallbackMessage)}`);
      }

      lines.push(`hint: ${getCommitHint(stderr || fallbackMessage, stdout)}`);
      return lines.join("\n");
    };

    const normalizeRef = (value?: string): string => {
      if (!value) {
        return "";
      }
      const trimmed = value.trim();
      if (trimmed.toLowerCase().startsWith("refs/heads/")) {
        return trimmed.slice("refs/heads/".length).trim();
      }
      return trimmed;
    };

    const normalizeOptionalField = (
      field: string,
      value: unknown,
    ): { value?: string; error?: string } => {
      if (value === undefined || value === null) {
        return {};
      }
      const text = String(value).trim();
      if (!text) {
        return {};
      }
      if (text.startsWith("-")) {
        return { error: `ERROR: Invalid ${field}: ${value}.` };
      }
      return { value: text };
    };

    const isProtectedBranchRef = (value?: string): boolean => {
      const normalized = normalizeRef(value).toLowerCase();
      return normalized === "main" || normalized === "master";
    };

    const isValidRefToken = (value: string): boolean => {
      if (!value) {
        return false;
      }
      if (value.startsWith("-")) {
        return false;
      }
      if (
        value.includes("..") ||
        value.includes("@{") ||
        value.includes("//") ||
        value.startsWith("/") ||
        value.endsWith("/") ||
        value.endsWith(".") ||
        value.endsWith(".lock") ||
        value.includes("\x00") ||
        /[ :?*\[\]\\]/.test(value)
      ) {
        return false;
      }
      return /^[A-Za-z0-9._/\-~^]+$/.test(value);
    };

    const appendCommandParts = (skipValidation = false): string | undefined => {
      switch (command) {
        case "commit": {
          cmdParts.push("commit");
          if (!skipValidation && (!summary || !summary.trim())) {
            return "ERROR: 'commit' command requires non-empty 'summary'.";
          }
          if (!skipValidation && no_verify !== undefined && typeof no_verify !== "boolean") {
            return "ERROR: 'no_verify' must be a boolean when provided.";
          }

          if (summary && summary.trim()) {
            cmdParts.push("--summary", summary);
          }
          if (description) {
            cmdParts.push("--description", description);
          }
          if (adw_id) {
            cmdParts.push("--adw-id", adw_id);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          if (stage_all) {
            cmdParts.push("--stage-all");
          }
          if (no_verify === true) {
            cmdParts.push("--no-verify");
          }
          const retries = max_retries ?? 3;
          if (!skipValidation && (!Number.isInteger(retries) || retries < 0 || retries > 10)) {
            return "ERROR: 'max_retries' must be an integer between 0 and 10.";
          }
          cmdParts.push("--max-retries", retries.toString());
          return undefined;
        }

        case "push": {
          cmdParts.push("push");
          const normalizedBranch = normalizeRef(branch);
          if (!skipValidation && !normalizedBranch) {
            return "ERROR: 'push' command requires 'branch'.";
          }
          if (!skipValidation && normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "checkout": {
          cmdParts.push("checkout");
          const normalizedBranch = normalizeRef(branch);
          const normalizedSource = normalizeRef(source);

          if (!skipValidation && !normalizedBranch) {
            return "ERROR: 'checkout' command requires 'branch'.";
          }
          if (!skipValidation && normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (!skipValidation && rawArgs.source !== undefined && !normalizedSource) {
            return "ERROR: 'checkout' command requires non-empty 'source' when provided.";
          }
          if (!skipValidation && normalizedSource && create !== true) {
            return "ERROR: 'checkout' command requires 'create' when 'source' is provided.";
          }
          if (!skipValidation && create === true && isProtectedBranchRef(normalizedBranch)) {
            return "ERROR: checkout --create to protected branch is blocked.";
          }
          if (!skipValidation && normalizedSource && !isValidRefToken(normalizedSource)) {
            return `ERROR: Invalid source: ${source}.`;
          }

          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (normalizedSource) {
            cmdParts.push("--source", normalizedSource);
          }
          if (create) {
            cmdParts.push("--create");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "merge": {
          cmdParts.push("merge");
          const normalizedSource = normalizeRef(source);
          if (!skipValidation && !normalizedSource) {
            return "ERROR: 'merge' command requires 'source'.";
          }
          if (source !== undefined && !skipValidation && !normalizedSource) {
            return "ERROR: 'merge' command requires non-empty 'source'.";
          }
          if (!skipValidation && normalizedSource && !isValidRefToken(normalizedSource)) {
            return `ERROR: Invalid source: ${source}.`;
          }
          if (normalizedSource) {
            cmdParts.push(normalizedSource);
          }
          const normalizedTarget = normalizeOptionalField("target", target);
          if (!skipValidation && normalizedTarget.error) {
            return normalizedTarget.error;
          }
          if (normalizedTarget.value) {
            cmdParts.push("--into", normalizedTarget.value);
          }
          if (no_ff) {
            cmdParts.push("--no-ff");
          }
          if (abort_on_conflict === false) {
            cmdParts.push("--no-abort-on-conflict");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "rebase": {
          cmdParts.push("rebase");
          const normalizedBranch = normalizeRef(branch);
          if (!skipValidation && !normalizedBranch) {
            return "ERROR: 'rebase' command requires 'branch'.";
          }
          if (!skipValidation && normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (normalizedBranch) {
            cmdParts.push(normalizedBranch);
          }
          const normalizedOnto = normalizeOptionalField("onto", onto);
          if (!skipValidation && normalizedOnto.error) {
            return normalizedOnto.error;
          }
          if (normalizedOnto.value) {
            cmdParts.push("--onto", normalizedOnto.value);
          }
          if (abort_on_conflict === false) {
            cmdParts.push("--no-abort-on-conflict");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "fetch": {
          cmdParts.push("fetch");
          const normalizedRemote = normalizeOptionalField("remote", remote);
          const normalizedBranch = normalizeRef(branch);
          if (!skipValidation && normalizedRemote.error) {
            return normalizedRemote.error;
          }
          if (!skipValidation && branch !== undefined && !normalizedBranch) {
            return "ERROR: 'branch' must be non-empty when provided.";
          }
          if (!skipValidation && normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (normalizedRemote.value) {
            cmdParts.push("--remote", normalizedRemote.value);
          } else {
            cmdParts.push("--remote", "origin");
          }
          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (prune) {
            cmdParts.push("--prune");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "sync": {
          cmdParts.push("sync");
          const normalizedSource = normalizeOptionalField("source", source);
          if (!skipValidation && normalizedSource.error) {
            return normalizedSource.error;
          }
          if (normalizedSource.value) {
            cmdParts.push("--source", normalizedSource.value);
          }
          const normalizedTarget = normalizeOptionalField("target", target);
          if (!skipValidation && normalizedTarget.error) {
            return normalizedTarget.error;
          }
          if (normalizedTarget.value) {
            cmdParts.push("--target", normalizedTarget.value);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "accumulate": {
          cmdParts.push("accumulate");
          const normalizedSliceBranch = normalizeRef(slice_branch);
          const normalizedTrackingBranch = normalizeRef(tracking_branch);
          if (!skipValidation && !normalizedSliceBranch) {
            return "ERROR: 'accumulate' command requires 'slice_branch'.";
          }
          if (!skipValidation && !normalizedTrackingBranch) {
            return "ERROR: 'accumulate' command requires 'tracking_branch'.";
          }
          if (!skipValidation && normalizedSliceBranch && !isValidRefToken(normalizedSliceBranch)) {
            return `ERROR: Invalid slice_branch: ${slice_branch}.`;
          }
          if (!skipValidation && normalizedTrackingBranch && !isValidRefToken(normalizedTrackingBranch)) {
            return `ERROR: Invalid tracking_branch: ${tracking_branch}.`;
          }
          if (normalizedSliceBranch) {
            cmdParts.push("--slice-branch", normalizedSliceBranch);
          }
          if (normalizedTrackingBranch) {
            cmdParts.push("--tracking-branch", normalizedTrackingBranch);
          }
          cmdParts.push("--json");
          if (recover_missing_worktree) {
            cmdParts.push("--recover-missing-worktree");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "abort": {
          cmdParts.push("abort");
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "continue": {
          cmdParts.push("continue");
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "reset": {
          cmdParts.push("reset");
          if (!skipValidation && (!ref || !ref.trim())) {
            return "ERROR: 'reset' command requires 'ref'.";
          }
          if (ref && ref.trim()) {
            cmdParts.push("--ref", ref);
          }
          if (hard) {
            cmdParts.push("--hard");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "push-force-with-lease": {
          cmdParts.push("push-force-with-lease");
          const normalizedBranch = normalizeRef(branch);
          if (!skipValidation && !normalizedBranch) {
            return "ERROR: 'push-force-with-lease' command requires 'branch'.";
          }
          if (!skipValidation && normalizedBranch && !isValidRefToken(normalizedBranch)) {
            return `ERROR: Invalid branch: ${branch}.`;
          }
          if (!skipValidation && isProtectedBranchRef(normalizedBranch)) {
            return "ERROR: push-force-with-lease to protected branch is blocked.";
          }
          if (normalizedBranch) {
            cmdParts.push("--branch", normalizedBranch);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "status": {
          cmdParts.push("status");
          if (porcelain) {
            cmdParts.push("--porcelain");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "diff": {
          cmdParts.push("diff");
          if (stat) {
            cmdParts.push("--stat");
          }
          const normalizedBase = normalizeOptionalField("base", base);
          if (!skipValidation && normalizedBase.error) {
            return normalizedBase.error;
          }
          if (normalizedBase.value) {
            cmdParts.push("--base", normalizedBase.value);
          }
          const normalizedTarget = normalizeOptionalField("target", target);
          if (!skipValidation && normalizedTarget.error) {
            return normalizedTarget.error;
          }
          if (normalizedTarget.value) {
            cmdParts.push("--target", normalizedTarget.value);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "log": {
          cmdParts.push("log");
          if (ref) {
            cmdParts.push("--ref", ref);
          }
          const maxCount = max_count ?? 10;
          if (!skipValidation && (!Number.isInteger(maxCount) || maxCount < 1 || maxCount > 1000)) {
            return "ERROR: 'max_count' must be an integer between 1 and 1000.";
          }
          cmdParts.push("--max-count", maxCount.toString());
          if (oneline) {
            cmdParts.push("--oneline");
          }
          if (stat) {
            cmdParts.push("--stat");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "show": {
          cmdParts.push("show");
          if (!skipValidation && (!ref || !ref.trim())) {
            return "ERROR: 'show' command requires 'ref'.";
          }
          const normalizedRef = normalizeOptionalField("ref", ref);
          if (!skipValidation && normalizedRef.error) {
            return normalizedRef.error;
          }
          if (normalizedRef.value) {
            cmdParts.push("--ref", normalizedRef.value);
          }
          const normalizedPath = normalizeOptionalField("path", path);
          if (!skipValidation && normalizedPath.error) {
            return normalizedPath.error;
          }
          if (normalizedPath.value) {
            cmdParts.push("--path", normalizedPath.value);
          }
          if (stat) {
            cmdParts.push("--stat");
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "add": {
          cmdParts.push("add");
          const hasStageAll = Boolean(stage_all);
          const hasFiles = Boolean(files && files.length > 0);

          if (!skipValidation && hasStageAll && hasFiles) {
            return "ERROR: 'add' command cannot combine 'stage_all' with 'files'.";
          }
          if (!skipValidation && !hasStageAll && !hasFiles) {
            return "ERROR: 'add' command requires either 'stage_all' or 'files'.";
          }

          if (hasStageAll) {
            cmdParts.push("--all");
          }
          if (hasFiles && files) {
            files.forEach((filePath) => {
              cmdParts.push("--files", filePath);
            });
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "restore": {
          cmdParts.push("restore");
          if (staged) {
            cmdParts.push("--staged");
          }
          if (files && files.length > 0) {
            files.forEach((filePath) => {
              cmdParts.push("--files", filePath);
            });
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "worktree-list": {
          cmdParts.push("worktree", "list");
          return undefined;
        }

        case "worktree-prune": {
          cmdParts.push("worktree", "prune");
          return undefined;
        }

        case "worktree-remove": {
          cmdParts.push("worktree", "remove");
          if (!skipValidation && (!adw_id || !adw_id.trim())) {
            return "ERROR: 'worktree-remove' command requires 'adw_id'.";
          }
          if (adw_id && adw_id.trim()) {
            cmdParts.push(adw_id);
          }
          if (force !== false) {
            cmdParts.push("--force");
          }
          return undefined;
        }

        default:
          return `ERROR: Unsupported command '${command}'.`;
      }
    };

    if (help) {
      const validationMessage = appendCommandParts(true);
      if (validationMessage && validationMessage.startsWith("ERROR")) {
        return validationMessage;
      }

      cmdParts.push("--help");
      try {
        const result = await Bun.$`${cmdParts}`.text();
        return `Git Command: ${command} (help)\n\n${result}`;
      } catch (error: any) {
        const errorOutput = error.stdout ? error.stdout.toString() : "";
        const errorMsg = error.stderr ? error.stderr.toString() : error.message;
        return `Git Command Failed: ${command}\n${errorMsg}${errorOutput ? `\n\nOutput:\n${errorOutput}` : ""}`;
      }
    }

    const validationMessage = appendCommandParts();
    if (validationMessage && validationMessage.startsWith("ERROR")) {
      return validationMessage;
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();

      if (command === "accumulate") {
        return result.trim();
      }


      return `Git Command: ${command}\n\n${result}`;
    } catch (error: any) {
      const errorOutput = error.stdout ? error.stdout.toString() : "";
      const errorMsg = error.stderr ? error.stderr.toString() : "";
      const fallbackMessage = typeof error?.message === "string" ? error.message : "";

      if (command === "commit") {
        return buildCommitError(error);
      }

      if (command === "accumulate" && errorOutput) {
        return errorOutput.trim();
      }

      const clippedStderr = clipSnippet(errorMsg);
      const clippedStdout = clipSnippet(errorOutput);
      const clippedFallback = clipSnippet(fallbackMessage);

      if (clippedStderr) {
        return `Git Command Failed: ${command}\n${clippedStderr}`;
      }

      if (clippedStdout) {
        return `Git Command Failed:\n${clippedStdout}`;
      }

      if (clippedFallback) {
        return `Git Command Failed: ${command}\n${clippedFallback}`;
      }

      return `Git Command Failed: ${command}\nCommand: ${cmdParts.join(" ")}\nUnknown error occurred during execution. No output or error message was captured.`;
    }
  },
});
