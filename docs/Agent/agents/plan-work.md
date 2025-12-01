# Plan-work Agent - Usage Guide

## Overview

Creates implementation plans from GitHub issues. First phase of ADW workflows.

## Invocation

**Automatic:**
```bash
uv run adw plan <issue-number>
uv run adw complete <issue-number>  # includes plan phase
```

## What It Does

1. Creates workspace using `create_workspace` tool
2. Reads issue from `adw_state.json` via `adw_spec`
3. Researches codebase for context
4. Generates ordered implementation steps
5. Writes plan to `spec_content`
6. Tracks progress with todo list

## Output

**Plan stored in:** `agents/{adw_id}/adw_state.json` (spec_content field)

**Plan includes:**
- Ordered steps with file paths
- Implementation details
- Testing strategy
- Acceptance criteria

## Example Usage

```bash
# Create plan for bug fix
uv run adw plan 123

# What happens:
# 1. Workspace created at trees/{adw_id}/
# 2. Issue #123 read from GitHub
# 3. Codebase researched for context
# 4. Plan generated with steps to fix bug
# 5. Plan written to spec_content
```

## Best Practices

- **Bug fixes**: Focus on root cause and regression tests
- **Features**: Follow architecture patterns, plan module structure
- **Chores**: Minimize disruption, update docs

## Troubleshooting

**Workspace creation failed:**
- Check issue exists: `gh issue view <number>`
- Verify auth: `gh auth status`

**Missing context:**
- Add file paths to issue description
- Reference specific functions to modify

## References

- **ADW System**: `README.md`
- **Architecture**: `docs/Agent/architecture_reference.md`
- **Code Style**: `docs/Agent/code_style.md`
