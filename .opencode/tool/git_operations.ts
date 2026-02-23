/**
 * Git Operations Tool for OpenCode Integration
 *
 * Exposes ADW git commands (commit, push, status, diff, add, restore) so
 * tool-only agents can run git inside isolated worktrees. Mirrors
 * adw/commands/git_cli.py semantics and follows existing OpenCode tool
 * patterns (see adw_spec.ts and adw.ts).
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: `Execute ADW git commands within isolated worktrees.

This tool wraps 'adw git <command>' for commit, push, status, diff, add, restore,
merge, rebase, fetch, sync, abort, reset, worktree lifecycle, and safe
push-force-with-lease. Use the worktree_path parameter to target a specific ADW
worktree and optionally attach adw_id to commits for traceability. Set help:
true to bypass validation and view CLI help for any command.

AVAILABLE COMMANDS:
• commit: Create commit with optional description and ADW ID
  Usage: { command: "commit", summary: "Add feature", worktree_path: "./trees/abc", stage_all: true }
• push: Push branch to origin
  Usage: { command: "push", branch: "feature-123", worktree_path: "./trees/abc" }
• status: Show git status (porcelain optional)
  Usage: { command: "status", porcelain: true, worktree_path: "./trees/abc" }
• diff: Show git diff with optional stat summary or base comparison
  Usage: { command: "diff", worktree_path: "./trees/abc" }
  Usage: { command: "diff", stat: true, worktree_path: "./trees/abc" }
  Usage: { command: "diff", base: "main", stat: true, worktree_path: "./trees/abc" }
  Usage: { command: "diff", base: "main", target: "feature", stat: true, worktree_path: "./trees/abc" }
• log: Show git log output with optional formatting
  Usage: { command: "log", ref: "main", max_count: 5, oneline: true, worktree_path: "./trees/abc" }
• show: Show git object content (commit or file at ref)
  Usage: { command: "show", ref: "main", stat: true, worktree_path: "./trees/abc" }
  Usage: { command: "show", ref: "main", path: "src/app.py", worktree_path: "./trees/abc" }
• add: Stage changes (mutually exclusive stage_all vs files)
  Usage: { command: "add", stage_all: true, worktree_path: "./trees/abc" }
  Usage: { command: "add", files: ["file1.py", "file2.py"], worktree_path: "./trees/abc" }
• restore: Restore changes (optional staged flag and files list)
  Usage: { command: "restore", staged: true, files: ["file.py"], worktree_path: "./trees/abc" }
• merge: Merge a branch into current or target branch (requires source)
  Usage: { command: "merge", source: "main", target: "develop", no_ff: true, worktree_path: "./trees/abc" }
• rebase: Rebase current branch onto specified branch (requires branch)
  Usage: { command: "rebase", branch: "develop", onto: "main", worktree_path: "./trees/abc" }
• fetch: Fetch from remote repository
  Usage: { command: "fetch", remote: "upstream", prune: true, worktree_path: "./trees/abc" }
• sync: Sync branch by fetching/merging from remote
  Usage: { command: "sync", source: "upstream", target: "develop", worktree_path: "./trees/abc" }
• abort: Abort in-progress merge or rebase
  Usage: { command: "abort", worktree_path: "./trees/abc" }
• continue: Continue an in-progress merge or rebase after resolving conflicts
  Usage: { command: "continue", worktree_path: "./trees/abc" }
• reset: Reset to specified ref (requires ref)
  Usage: { command: "reset", ref: "HEAD~1", hard: true, worktree_path: "./trees/abc" }
• push-force-with-lease: Safe force push for rebased branches (requires branch; blocks main/master)
  Usage: { command: "push-force-with-lease", branch: "feature-123", worktree_path: "./trees/abc" }
• worktree-list: List registered git worktrees
  Usage: { command: "worktree-list" }
• worktree-prune: Prune stale git worktree references
  Usage: { command: "worktree-prune" }
• worktree-remove: Remove a worktree by ADW ID (supports --force)
  Usage: { command: "worktree-remove", adw_id: "abc12345" }
  Usage: { command: "worktree-remove", adw_id: "abc12345", force: true }

WORKTREE ISOLATION & ADW ID TRACEABILITY:
- Always set worktree_path when running inside ADW-managed trees
- Optional adw_id is appended to commit body for auditing
- add.stage_all maps to the CLI --all flag; commit.stage_all maps to --stage-all

HELP:
Set help: true to view command help without validation or required parameters.
  Example: { command: "commit", help: true }

REQUIRED PARAMETERS BY COMMAND:
• commit: summary (description, adw_id, worktree_path, stage_all, max_retries optional)
• push: branch (worktree_path optional)
• merge: source required; target optional (--into), no_ff optional (--no-ff), abort_on_conflict optional (default true)
• rebase: branch required; onto optional (--onto), abort_on_conflict optional (default true)
• fetch: remote optional (default origin), branch optional, prune optional, worktree_path optional
• sync: source optional (default origin), target optional, worktree_path optional
• abort: no required params; worktree_path optional
• continue: no required params; worktree_path optional
• reset: ref required; hard optional (default false); worktree_path optional
• push-force-with-lease: branch required; blocks main/master; worktree_path optional
• status: none (porcelain, worktree_path optional)
• diff: none (stat, base, target, worktree_path optional)
• log: none (ref, max_count, oneline, stat, worktree_path optional)
• show: ref required (path, stat, worktree_path optional)
• add: exactly one of stage_all or files (worktree_path optional)
• restore: none required (staged, files, worktree_path optional)
• worktree-list: none required
• worktree-prune: none required
• worktree-remove: adw_id required (force optional)

Set help: true to bypass validation and return CLI help.`
,

  args: {
    command: tool.schema
      .enum([
        "commit",
        "push",
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
        "abort",
        "continue",
        "reset",
        "push-force-with-lease",
        "worktree-list",
        "worktree-remove",
        "worktree-prune",
      ])
      .describe(`Git command to execute. Use help: true to see CLI usage.
 
REQUIRED PARAMETERS BY COMMAND:

• commit: summary (description, adw_id, worktree_path, stage_all, max_retries optional)
• push: branch (worktree_path optional)
• merge: source required; target optional (--into), no_ff optional (--no-ff), abort_on_conflict optional (default true)
• rebase: branch required; onto optional (--onto), abort_on_conflict optional (default true)
• fetch: remote optional (default origin), branch optional, prune optional, worktree_path optional
• sync: source optional (default origin), target optional, worktree_path optional
• abort: no required params; worktree_path optional
• reset: ref required; hard optional (default false); worktree_path optional
• push-force-with-lease: branch required; blocks main/master; worktree_path optional
• status: none (porcelain, worktree_path optional)
• diff: none (stat, base, target, worktree_path optional)
• log: none (ref, max_count, oneline, stat, worktree_path optional)
• show: ref required (path, stat, worktree_path optional)
• add: exactly one of stage_all or files (worktree_path optional)
• restore: none required (staged, files, worktree_path optional)
• worktree-list: none required
• worktree-prune: none required
• worktree-remove: adw_id required (force optional)

Set help: true to bypass validation and return CLI help.`),

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

    branch: tool.schema
      .string()
      .optional()
      .describe(`Branch name to push to origin.

Required for: push
Mapped flag: --branch

Example: { command: "push", branch: "feature-123" }`),

    source: tool.schema
      .string()
      .optional()
      .describe(`Source branch/ref for merge or sync.

Required for: merge
Optional for: sync (default origin)
Mapped flags: merge uses positional <source>; sync uses --source

Example: { command: "merge", source: "main" }`),

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
      .describe(`Branch name for push, rebase target, fetch branch, sync target, reset branch refs, or force-with-lease.

Applies to: push, rebase, push-force-with-lease, fetch, sync, reset
Required for: push, rebase, push-force-with-lease
Optional for: fetch, sync, reset
Mapped flags/positionals:
  • push, push-force-with-lease: --branch <branch>
  • rebase: <branch> positional (rebase target)
  • fetch: fetches the specified <branch> from --remote
  • sync: syncs the specified <branch> with --remote
  • reset: branch whose refs are reset (tool-specific behavior)

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
    const {
      command,
      summary,
      description,
      adw_id,
      worktree_path,
      stage_all,
      force,
      max_retries,
      source,
      target,
      no_ff,
      abort_on_conflict,
      branch,
      onto,
      remote,
      prune,
      ref,
      hard,
      porcelain,
      stat,
      base,
      max_count,
      oneline,
      path,
      files,
      staged,
      help,
    } = args;

    const cmdParts = ["uv", "run", "adw", "git"];

    const appendCommandParts = (skipValidation = false): string | undefined => {
      switch (command) {
        case "commit": {
          cmdParts.push("commit");
          if (!skipValidation && (!summary || !summary.trim())) {
            return "ERROR: 'commit' command requires non-empty 'summary'.";
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
          const retries = max_retries ?? 3;
          cmdParts.push("--max-retries", retries.toString());
          return undefined;
        }

        case "push": {
          cmdParts.push("push");
          if (!skipValidation && !branch) {
            return "ERROR: 'push' command requires 'branch'.";
          }
          if (branch) {
            cmdParts.push("--branch", branch);
          }
          if (worktree_path) {
            cmdParts.push("--worktree-path", worktree_path);
          }
          return undefined;
        }

        case "merge": {
          cmdParts.push("merge");
          if (!skipValidation && (!source || !source.trim())) {
            return "ERROR: 'merge' command requires 'source'.";
          }
          if (source && source.trim()) {
            cmdParts.push(source);
          }
          if (target) {
            cmdParts.push("--into", target);
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
          if (!skipValidation && (!branch || !branch.trim())) {
            return "ERROR: 'rebase' command requires 'branch'.";
          }
          if (branch && branch.trim()) {
            cmdParts.push(branch);
          }
          if (onto) {
            cmdParts.push("--onto", onto);
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
          if (remote) {
            cmdParts.push("--remote", remote);
          } else {
            cmdParts.push("--remote", "origin");
          }
          if (branch) {
            cmdParts.push("--branch", branch);
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
          if (source) {
            cmdParts.push("--source", source);
          }
          if (target) {
            cmdParts.push("--target", target);
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
          if (!skipValidation && (!branch || !branch.trim())) {
            return "ERROR: 'push-force-with-lease' command requires 'branch'.";
          }
          if (!skipValidation && branch && ["main", "master"].includes(branch)) {
            return "ERROR: push-force-with-lease to protected branch is blocked.";
          }
          if (branch && branch.trim()) {
            cmdParts.push("--branch", branch);
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
          if (base) {
            cmdParts.push("--base", base);
          }
          if (target) {
            cmdParts.push("--target", target);
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
          if (ref && ref.trim()) {
            cmdParts.push("--ref", ref);
          }
          if (path) {
            cmdParts.push("--path", path);
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


      if (result.includes("ERROR:") || result.includes("Error:")) {
        return `Git Command Failed:\n${result}`;
      }

      return `Git Command: ${command}\n\n${result}`;
    } catch (error: any) {
      const errorOutput = error.stdout ? error.stdout.toString() : "";
      const errorMsg = error.stderr ? error.stderr.toString() : error.message;

      if (errorOutput) {
        return `Git Command Failed:\n${errorOutput}`;
      }

      if (errorMsg) {
        return `Git Command Failed: ${command}\n${errorMsg}`;
      }

      return `Git Command Failed: ${command}\nCommand: ${cmdParts.join(" ")}\nUnknown error occurred during execution. No output or error message was captured.`;
    }
  },
});
