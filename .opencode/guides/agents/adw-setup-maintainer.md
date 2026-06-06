# ADW Setup Maintainer Agent - Usage Guide

> **Status:** Deprecated. The `templates/Agent/` mirror and `adw setup template extract` commands have been removed. `adw-docs/` is now the sole source of truth for documentation. This guide remains as a migration aid for maintainers.

## Overview

Maintainers now edit `adw-docs/` directly and regenerate language-specific stubs when needed. There is no longer any bidirectional extraction from live docs back into templates. Use the docs scaffold/apply workflow to refresh stubs, and manage the ~15 keyword manifest when token updates are required.

## When to Use

| Scenario | Action |
|----------|--------|
| Update or add docs in `adw-docs/` | Edit files directly and commit |
| Regenerate stubs after edits | `adw setup docs scaffold --language <lang> --force` (optional) then `adw setup docs apply --language <lang>` |
| Manage keyword tokens | `adw setup template token list|add|remove` |
| Validate placeholder coverage | `adw setup template validate` |
| Label sync or other CLI work | Use the new `adw setup labels` command (alias `adw sync labels` is deprecated) |

## Key Commands (current workflow)

### Refresh docs stubs

```bash
# Regenerate stubs (overwrites existing stubs)
adw setup docs scaffold --language python --force

# Apply manifest placeholders to regenerated stubs
adw setup docs apply --language python
```

### Validate/Manage tokens

```bash
adw setup template validate
adw setup template token list
adw setup template token add <TOKEN> [--default ...] [--description ...]
adw setup template token remove <TOKEN> --yes
```

> `adw setup template extract` and `adw setup template extract --diff|--dry-run` no longer exist. Do not attempt to run them.

## Migration guidance

- Edit `adw-docs/` directly; do not maintain a `templates/Agent/` mirror.
- Use `adw setup docs scaffold/apply` to keep language stubs aligned after significant edits.
- Keep the keyword manifest (`adw/templates/keyword_manifest.yaml`) in sync with any new placeholders introduced in docs.
- For label sync, update scripts to use `adw setup labels`; the `adw sync labels` alias will be removed in v0.3.0.

## Example: regenerate stubs after a docs update

```bash
# After editing adw-docs/testing_guide.md
adw setup docs scaffold --language python --force
adw setup docs apply --language python
adw setup template validate
```

## Permissions (unchanged)

| Tool | Enabled | Rationale |
|------|---------|-----------|
| `read` | Yes | Read docs and manifests |
| `edit/write` | Yes | Update docs and manifests |
| `list/glob/grep` | Yes | File discovery |
| `git_diff` | Yes | Check status and inspect setup-doc changes |
| `git_stage` | Yes | Stage documentation/template updates |
| `git_commit` | Yes | Create commits for maintainer changes |
| `task` | Yes | Invoke adw-commit subagent |
| `platform_operations` | No | No GitHub/GitLab API needed |
| `run_pytest` | No | Not testing code |
| `run_linters` | No | Not linting code |
| `bash` | No | Always disabled |
