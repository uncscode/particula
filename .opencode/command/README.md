---
description: "OpenCode Commands"
---

# OpenCode Commands

This directory contains slash commands for the ADW repository. Workflow-related
commands have been replaced by JSON workflows in `.opencode/workflow/`.

## Available Commands

### Project Setup

- **prime.md** - Initialize and understand the codebase structure
- **install.md** - Installation instructions for development setup
- **install_worktree.md** - Set up a git worktree for isolated development
- **tools.md** - List available development tools

### Git Operations

- **commit.md** - Git commit helper with conventional commit format
- **pull_request.md** - Create pull requests with proper formatting
- **cleanup_worktrees.md** - Clean up git worktrees
- **sync.md** - Sync operations for ADW state

### Review Commands

- **feature_review.md** - Feature review checklist
- **maintenance_review.md** - Maintenance review checklist

### Utilities

- **conditional_docs.md** - Guide for which documentation to read
- **create-workflow.md** - Create custom JSON workflows
- **version.md** - Display version information

## Replaced by JSON Workflows

The following commands are now handled by JSON workflows in `.opencode/workflow/`:

| Old Command | New Workflow |
|-------------|--------------|
| `/implement` | `adw workflow execute-plan` |
| `/patch` | `adw workflow patch` |
| `/document` | `adw workflow document` |
| `/review` | Part of `adw workflow complete` |
| `/lint` | `linter` agent via workflows |
| `/docstring` | `docstring` agent via workflows |
| `/feature` | `adw workflow complete` |

## Replaced by CLI Commands

| Old Command | New CLI |
|-------------|---------|
| `/interpret_issue` | `adw interpret-issue` |
| `/create_issue` | `adw create-issue` |
| `/health_check` | `adw health` |

## Using Commands

Commands are available with the `/` prefix:

```bash
# Project setup
/prime
/install
/install_worktree

# Git operations
/commit
/pull_request
/cleanup_worktrees

# Reviews
/feature_review
/maintenance_review

# Utilities
/conditional_docs
/version
```
