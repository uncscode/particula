# adw_setup Tool

Bounded wrapper for `adw setup` subcommands.

## Supported Commands

- `env`
- `validate`
- `check`
- `labels`
- `docs`
- `pull-opencode`
- `pull-plans`

`template` subcommands are intentionally excluded from this wrapper.

## Behavior

- `wizard: true` runs bare `adw setup` (no subcommand).
- `help: true` runs:
  - `adw setup --help` when `command` is omitted
  - `adw setup <command> --help` when `command` is provided
- `wizard` and `command` are mutually exclusive.
- Without `help`/`wizard`, `command` is required.

## Command-Scoped Options

- `with_templates`, `skip_templates`: allowed only for `env`.
- `format`: allowed only for `validate` (`panel|table|json`).
- `dry_run`: allowed only for `labels`.
- `args`: allowed only for `docs`, `pull-opencode`, `pull-plans`.

`args` entries must be non-empty strings and may not include protected flags:
`--help`, `--with-templates`, `--skip-templates`, `--format`, `--dry-run`.

## Examples

```jsonc
{ "wizard": true }
{ "help": true }
{ "command": "env", "help": true }
{ "command": "env", "with_templates": true }
{ "command": "validate", "format": "json" }
{ "command": "labels", "dry_run": true }
{ "command": "docs", "args": ["--strict"] }
```
