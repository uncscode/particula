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

This tool wraps 'adw git <command>' for commit, push, status, diff, add, and
restore. Use the worktree_path parameter to target a specific ADW worktree and
optionally attach adw_id to commits for traceability. Set help: true to bypass
validation and view CLI help for any command.

AVAILABLE COMMANDS:
• commit: Create commit with optional description and ADW ID
  Usage: { command: "commit", summary: "Add feature", worktree_path: "./trees/abc", stage_all: true }
• push: Push branch to origin
  Usage: { command: "push", branch: "feature-123", worktree_path: "./trees/abc" }
• status: Show git status (porcelain optional)
  Usage: { command: "status", porcelain: true, worktree_path: "./trees/abc" }
• diff: Show git diff (stat optional)
  Usage: { command: "diff", stat: true, worktree_path: "./trees/abc" }
• add: Stage changes (mutually exclusive stage_all vs files)
  Usage: { command: "add", stage_all: true, worktree_path: "./trees/abc" }
  Usage: { command: "add", files: ["file1.py", "file2.py"], worktree_path: "./trees/abc" }
• restore: Restore changes (optional staged flag and files list)
  Usage: { command: "restore", staged: true, files: ["file.py"], worktree_path: "./trees/abc" }

WORKTREE ISOLATION & ADW ID TRACEABILITY:
- Always set worktree_path when running inside ADW-managed trees
- Optional adw_id is appended to commit body for auditing
- add.stage_all maps to the CLI --all flag; commit.stage_all maps to --stage-all

HELP:
Set help: true to view command help without validation or required parameters.
  Example: { command: "commit", help: true }`,

  args: {
    command: tool.schema
      .enum(["commit", "push", "status", "diff", "add", "restore"])
      .describe(`Git command to execute. Use help: true to see CLI usage.

REQUIRED PARAMETERS BY COMMAND:
• commit: summary (description, adw_id, worktree_path, stage_all, max_retries optional)
• push: branch (worktree_path optional)
• status: none (porcelain, worktree_path optional)
• diff: none (stat, worktree_path optional)
• add: exactly one of stage_all or files (worktree_path optional)
• restore: none required (staged, files, worktree_path optional)

Set help: true to bypass validation and return CLI help.`),

    summary: tool.schema
      .string()
      .optional()
      .describe(`Commit summary line. Required for command: "commit".

Examples:
• "Add git operations tool"
• "Fix lint warnings"

Mapped flag: --summary`),

    description: tool.schema
      .string()
      .optional()
      .describe(`Optional commit body description for "commit" command.

Mapped flag: --description
Example: "Add tool-only git operations wrapper"`),

    adw_id: tool.schema
      .string()
      .optional()
      .describe(`Optional ADW workflow ID appended to commit body for traceability.

Mapped flag: --adw-id
Example: "a17d3f8a"`),

    worktree_path: tool.schema
      .string()
      .optional()
      .describe(`Target worktree directory. Recommended for all commands to ensure
isolation.

Mapped flag: --worktree-path
Example: "./trees/a17d3f8a"`),

    stage_all: tool.schema
      .boolean()
      .optional()
      .describe(`Stage all changes.

Applies to:
• commit: stages all changes before commit when true (maps to --stage-all)
• add: mutually exclusive with files; when true, maps to --all

Examples:
• { command: "commit", summary: "msg", stage_all: true }
• { command: "add", stage_all: true }`),

    max_retries: tool.schema
      .number()
      .optional()
      .describe(`Maximum retries for commit when hooks modify files. Default: 3.

Applies to: commit
Mapped flag: --max-retries
Example: 5`),

    branch: tool.schema
      .string()
      .optional()
      .describe(`Branch name to push. Required for command: "push".

Mapped flag: --branch
Example: "feature-123"`),

    porcelain: tool.schema
      .boolean()
      .optional()
      .describe(`Return porcelain output for status.

Applies to: status
Mapped flag: --porcelain`),

    stat: tool.schema
      .boolean()
      .optional()
      .describe(`Include --stat summary for diff.

Applies to: diff
Mapped flag: --stat`),

    files: tool.schema
      .array(tool.schema.string())
      .optional()
      .describe(`Specific files for add or restore commands.

Rules for add:
- Mutually exclusive with stage_all
- Required when stage_all is false
- Each file is passed as --files <file>

Rules for restore:
- Optional; when omitted, restores all changes
- Each file is passed as --files <file>`),

    staged: tool.schema
      .boolean()
      .optional()
      .describe(`Restore from index instead of working tree when true.

Applies to: restore
Mapped flag: --staged`),

    help: tool.schema
      .boolean()
      .optional()
      .describe(`Show CLI help for the chosen command.

When true, validation is skipped and the tool executes with --help to return
command usage details.`),
  },

  async execute(args) {
    const {
      command,
      summary,
      description,
      adw_id,
      worktree_path,
      stage_all,
      max_retries,
      branch,
      porcelain,
      stat,
      files,
      staged,
      help,
    } = args;

    const cmdParts = ["uv", "run", "adw", "git", command];

    if (help) {
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

    switch (command) {
      case "commit": {
        if (!summary || !summary.trim()) {
          return "ERROR: 'commit' command requires non-empty 'summary'.";
        }

        cmdParts.push("--summary", summary);
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
        break;
      }

      case "push": {
        if (!branch) {
          return "ERROR: 'push' command requires 'branch'.";
        }

        cmdParts.push("--branch", branch);
        if (worktree_path) {
          cmdParts.push("--worktree-path", worktree_path);
        }
        break;
      }

      case "status": {
        if (porcelain) {
          cmdParts.push("--porcelain");
        }
        if (worktree_path) {
          cmdParts.push("--worktree-path", worktree_path);
        }
        break;
      }

      case "diff": {
        if (stat) {
          cmdParts.push("--stat");
        }
        if (worktree_path) {
          cmdParts.push("--worktree-path", worktree_path);
        }
        break;
      }

      case "add": {
        const hasStageAll = Boolean(stage_all);
        const hasFiles = Boolean(files && files.length > 0);

        if (hasStageAll && hasFiles) {
          return "ERROR: 'add' command cannot combine 'stage_all' with 'files'.";
        }
        if (!hasStageAll && !hasFiles) {
          return "ERROR: 'add' command requires either 'stage_all' or 'files'.";
        }

        if (hasStageAll) {
          cmdParts.push("--all");
        }
        if (hasFiles) {
          files.forEach((filePath) => {
            cmdParts.push("--files", filePath);
          });
        }
        if (worktree_path) {
          cmdParts.push("--worktree-path", worktree_path);
        }
        break;
      }

      case "restore": {
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
        break;
      }

      default:
        return `ERROR: Unsupported command '${command}'.`;
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
