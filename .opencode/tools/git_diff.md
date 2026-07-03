# git_diff tool

Read-only ADW git inspection wrapper for `status`, `diff`, `log`, and `show`.

## Supported commands

- `status`
  - Optional: `porcelain`, `worktree_path`
- `diff`
  - Optional: `stat`, `base`, `target`, `path`, `worktree_path`
  - Validation: accepts valid Git rev-spec syntax (for example `stash@{1}` and `HEAD^{tree}`), forwards only non-empty repo-relative scoped paths as `--path <value>`, and rejects malformed, absolute, traversal-like, and repo-root-equivalent path payloads before spawn
- `log`
  - Optional: `ref`, `max_count`, `oneline`, `stat`, `worktree_path`
  - `max_count` defaults to `10`
  - Validation: must be an integer in `1..1000`
- `show`
  - Required: `ref` (except when `help: true`)
  - Optional: `path`, `stat`, `worktree_path`
  - Validation: accepts valid Git rev-spec syntax (for example `HEAD@{1}`) while rejecting malformed `ref` values and option-like `path` values

## Help behavior

- `help: true` appends `--help` to the selected command.
- Help mode relaxes required-argument checks only; malformed refs and option-like paths still fail closed before spawn.

## Success envelope

All non-help command successes return:

`Git Command: <command>\n\n<output>`

## Deterministic error behavior

- Missing `show` ref:
  - `ERROR: 'show' command requires 'ref'.`
- Invalid `log` max_count:
  - `ERROR: 'max_count' must be an integer between 1 and 1000.`
- Invalid `diff` / `show` refs:
  - `ERROR: Invalid base: <value>.`
  - `ERROR: Invalid target: <value>.`
  - `ERROR: Invalid ref: <value>.`
- Invalid option-like paths:
  - `ERROR: Invalid path: <value>.`
  - `ERROR: Invalid worktree_path: <value>.`
- Invalid scoped diff paths do not fall back to repo-wide output; blank or omitted `path` is the only unscoped diff form, and `.` / repo-root-equivalent scoped values are rejected.
- Subprocess failure precedence:
  1. stderr snippet (`Git Command Failed: <command>\n<stderr>`)
  2. stdout snippet (`Git Command Failed:\n<stdout>`)
  3. fallback message / fixed unknown-error envelope
- When failure diagnostics are truncated, output may also include a repo-local
  `debug_log: adforge_local/opencode/tmp/git_diff-<command>-...` path.

## Scope guardrail

This tool is intentionally read-only and excludes mutating git paths (commit/push/checkout/merge/etc.).
