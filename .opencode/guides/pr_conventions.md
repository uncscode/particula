# Pull Request Conventions

**Version:** 2.1.0
**Last Updated:** 2025-11-14

## Overview

This guide documents pull request format and conventions for the adw repository.

### PR Title Format

**Format:**
```
<type>: <description>
```

**Example:**
```
feat: add patch workflow support
```

### PR Body Structure

All PRs must include:

```markdown
## Summary
Brief description of what this PR does and why.

## Changes
- Bullet list of specific changes made
- Keep it concise and focused

## Testing
How was this tested? What test cases were added or updated?
```

### Branch Naming

**Format**: `<type>/<short-description>` or `issue-<number>-<description>`

**Examples:**
- `feat/patch-workflow`
- `fix/error-handling`
- `issue-123-add-authentication`

### Platform Commands

**GitHub** (using `gh` CLI):
```bash
gh pr create --title "feat: add patch workflow support" --body "..."
```

**ADW** (platform router):
```bash
adw platform create-pr --title "feat: add patch workflow support" --body "..." \
  --head feature/patch-workflow --base main
```

`adw platform create-pr` applies the default PR labels (`agent`, `blocked`,
`request:fix`) unless `--no-default-labels` is provided. Remove `request:fix` or
add `blocked` when you want to pause the automated fix cycle.

### Issue Linking

**Format**: Reference issue number in PR body using `Fixes #<issue-number>` or `Closes #<issue-number>`

**Example**: `Fixes #123`

## Integration with ADW

ADW PR commands use this guide to:
- Format PR titles and bodies
- Create PRs using correct platform commands
- Link PRs to issues appropriately

## See Also

- **docs/ai_docs/commit_conventions.md**: Commit message format
- **docs/ai_docs/review_guide.md**: Review process and criteria
