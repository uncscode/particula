# git_diff tool

Read-only ADW git inspection wrapper for `status`, `diff`, `log`, and `show`.

## Supported commands

- `status`
  - Optional: `porcelain`, `worktree_path`
- `diff`
  - Optional: `stat`, `base`, `target`, `worktree_path`
- `log`
  - Optional: `ref`, `max_count`, `oneline`, `stat`, `worktree_path`
  - `max_count` defaults to `10`
  - Validation: must be an integer in `1..1000`
- `show`
  - Required: `ref` (except when `help: true`)
  - Optional: `path`, `stat`, `worktree_path`

## Help behavior

- `help: true` appends `--help` to the selected command.
- Help mode relaxes strict argument requirements for parity with existing wrappers.

## Success envelope

All non-help command successes return:

`Git Command: <command>\n\n<output>`

## Deterministic error behavior

- Missing `show` ref:
  - `ERROR: 'show' command requires 'ref'.`
- Invalid `log` max_count:
  - `ERROR: 'max_count' must be an integer between 1 and 1000.`
- Subprocess failure precedence:
  1. stderr snippet (`Git Command Failed: <command>\n<stderr>`)
  2. stdout snippet (`Git Command Failed:\n<stdout>`)
  3. fallback message / fixed unknown-error envelope

## Scope guardrail

This tool is intentionally read-only and excludes mutating git paths (commit/push/checkout/merge/etc.).
