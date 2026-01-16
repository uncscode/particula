---
description: "OpenCode Commands"
---

# OpenCode Commands

This directory contains slash commands for the ADW repository. Most workflow-related
commands have been replaced by JSON workflows in `.opencode/workflow/`.

## Available Commands

- **create-workflow.md** - Create custom ADW workflow with interactive builder
- **epic-to-issues.md** - Generate type:generate issues for all features in an epic
- **remove_trees.md** - List and remove git worktrees in `trees/`
- **tools.md** - List built-in development tools
- **version.md** - Display version information

## Using Commands

Commands are available with the `/` prefix:

```bash
/remove_trees
/version
```

## Replaced by JSON Workflows

The following commands are now handled by JSON workflows in `.opencode/workflow/`:

| Old Command | New Workflow |
|-------------|--------------|
| `/implement` | `adw workflow build` |
| `/patch` | `adw workflow patch` |
| `/document` | `adw workflow document` |
| `/review` | `adw workflow review` |
| `/feature` | `adw workflow complete` |

## Replaced by CLI Commands

| Old Command | New CLI |
|-------------|---------|
| `/interpret_issue` | `adw interpret-issue` |
| `/create_issue` | `adw create-issue` |

See `.opencode/workflow/` for available JSON workflow definitions.
