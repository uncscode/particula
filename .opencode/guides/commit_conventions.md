# Commit Conventions

**Version:** 2.2.0
**Last Updated:** 2025-12-11

## Overview

This guide documents commit message format for the adw repository.

## Commit Message Format

adw follows the **Conventional Commits** format for clear, structured commit messages that are easy to parse and understand.

**Structure:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Example:**
```
feat(workflows): add patch workflow support

Implement patch workflow that skips test/review/document phases
for minor fixes. Includes state management and error handling.

Fixes #123
```

## git_commit Tool Parameters

When using the `git_commit` tool, the commit message is split into two parameters:

| Parameter | Maps To | Content |
|-----------|---------|---------|
| `summary` | Subject line | `<type>(<scope>): <description>` (max 72 chars) |
| `description` | Body + footer | Extended explanation and issue references |

Optional commit-only flag:

| Parameter | Maps To | Behavior |
|-----------|---------|----------|
| `no_verify` | `git commit --no-verify` | Commit-only flag. `true` appends `--no-verify`; `false` or omitted does not append it. |

**Example tool call:**
```python
git_commit({
  "summary": "feat(parser): add input validation for malformed data",
  "description": "Add validate_input() function to check data integrity before\nprocessing. This prevents cryptic errors when users provide\nincomplete or malformed input.\n\nCloses #123",
  "no_verify": false,
  "worktree_path": "/trees/abc12345",
  "adw_id": "abc12345"
})
```

Use `no_verify: true` only for explicit, exceptional cases (for example, when a
known-safe commit must bypass local hooks in automation).

## Commit Types

| Type | Use When |
|------|----------|
| `feat` | New user-facing feature or functionality |
| `fix` | Bug fix |
| `docs` | Documentation changes only |
| `style` | Code style changes (formatting, whitespace) |
| `refactor` | Code restructuring without behavior change |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `ci` | CI/CD configuration changes |
| `chore` | Maintenance tasks, dependency updates |

### Breaking Changes

For breaking changes, add `!` after the type:
```
feat!: remove deprecated API endpoints

BREAKING CHANGE: The /v1/legacy endpoints have been removed.
Migrate to /v2/ endpoints before upgrading.

Closes #456
```

## Scope Guidelines

Scope should identify the primary module affected:

| Directory | Scope |
|-----------|-------|
| `adw/core/` | `core` |
| `adw/utils/` | `utils` |
| `adw/workflows/` | `workflows` |
| `adw/github/` | `github` |
| `adw/platforms/` | `platforms` |
| `adw/state/` | `state` |
| `adw/git/` | `git` |
| `docs/` | `docs` |

**When to omit scope:**
- Changes span multiple modules equally
- Repository-wide changes (e.g., dependency updates)

## Message Tense

**Tense**: Imperative mood (present tense, command form)

| ✓ Correct | ✗ Wrong |
|-----------|---------|
| `add user authentication` | `added user authentication` (past) |
| `fix parser crash` | `fixes parser crash` (third person) |
| `update dependencies` | `updating dependencies` (continuous) |

## Length Limits

- **Subject line (summary)**: 72 characters maximum
- **Body lines**: 72-100 characters (wrap for readability)
- **Total body**: No strict limit, but keep concise

## Issue Linking

**Format**: Use `Fixes #<issue-number>` or `Closes #<issue-number>` in footer

**Placement**: Always in the footer (last section), separated by blank line

**Examples:**
```
Fixes #123
Closes #456
Refs #789
```

## Body Content Guidelines

The body should explain **why**, not just **what**:

**Good body:**
```
Add validate_input() function to check data integrity before
processing. This prevents cryptic errors when users provide
incomplete or malformed input.

- Validate JSON structure
- Check for required fields  
- Return helpful error messages
```

**Bad body:**
```
Made some changes to the parser to fix stuff.
```

## Integration with ADW

### adw-commit Subagent

The `adw-commit` subagent uses this guide to:
- Determine commit type from changes (feat/fix/refactor/etc.)
- Generate scope from most-changed module
- Format messages with proper structure
- Include issue references in footer

### Shipper Agent

The shipper agent delegates commits to `adw-commit` and expects:
- `ADW_COMMIT_SUCCESS` - Commit created successfully
- `ADW_COMMIT_SKIPPED` - No changes to commit
- `ADW_COMMIT_FAILED` - Commit failed after retries

## See Also

- **pr_conventions.md**: Pull request format
- **review_guide.md**: Code review process
- **.opencode/agent/adw-commit.md**: Commit subagent reference
