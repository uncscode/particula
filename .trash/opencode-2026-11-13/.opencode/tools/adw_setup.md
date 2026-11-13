# adw_setup Tool

Bounded wrapper for `adw setup` subcommands.

## Supported Commands

- `env`
- `validate`
- `check`
- `labels`
- `docs`
- `pull-opencode`
- `template`

`pull-plans` is no longer supported in this wrapper; use `pull-opencode`.

## Behavior

- `wizard: true` runs bare `adw setup` (no subcommand).
- `help: true` runs:
  - `adw setup --help` when `command` is omitted
  - `adw setup <command> --help` when `command` is provided
- `wizard` and `command` are mutually exclusive.
- `wizard`/`command` mutual exclusion is enforced even when `help: true`.
- Without `help`/`wizard`, `command` is required.

## Command-Scoped Options

- `options: "with-templates"` or `options: "skip-templates"`: allowed only for `env`.
- `options: "format=<panel|table|json>"`: allowed only for `validate`.
- `options: "dry-run"`: allowed only for `labels`.
- `args`: allowed only for `docs`, `pull-opencode`, `template`.

`args` entries must be non-empty strings and may not include protected flags:
`--help`, `--with-templates`, `--skip-templates`, `--format`.

Additional passthrough guardrails:

- `docs` requires a supported subcommand and bounded args only:
  - `scaffold --language <python|cpp|typescript|minimal> [--force] [--no-detect]`
  - `apply [--dry-run] [--check]`
  - `token list|set|remove`
- `pull-opencode` accepts only documented pull flags such as
  `--source-repo`, `--source-path`, `--dest`, `--ref`, `--yes`,
  `--preserve-manifest`, `--preserve`, and `--no-preserve`.
- `template` requires a supported subcommand and bounded args only:
  - `init [--yes|-y] [--gitignore-mode <active|commented>]`
  - `apply [--check] [--dry-run] [--yes|-y]`
  - `extract [--diff] [--dry-run] [--yes|-y]`
  - `validate [--format json]`
  - `token list|add|remove`

## Examples

```jsonc
{ "wizard": true }
{ "help": true }
{ "command": "env", "help": true }
{ "command": "env", "options": "with-templates" }
{ "command": "validate", "options": "format=json" }
{ "command": "labels", "options": "dry-run" }
{ "command": "docs", "args": ["apply", "--check"] }
{ "command": "pull-opencode", "args": ["--ref", "main", "--yes"] }
{ "command": "template", "args": ["validate", "--format", "json"] }
```
