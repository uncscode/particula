# adw_status_health tool

Read-only ADW wrapper for:

- `status`
- `health`

## Required arguments

- No `issue_number` required.
- `help: true` appends `--help`.

## Optional arguments

- No additional wrapper arguments beyond `help`.

## Timeout and errors

- Default timeout: `120000ms`
- Timeout envelope: `ERROR: Failed to execute 'adw <command>' (timeout after 120000ms).`
- Non-zero exit envelope uses deterministic precedence: `stderr` → `stdout` → fallback.
