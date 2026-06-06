# adw_status_health tool

Read-only ADW wrapper for:

- `status`
- `health`

## Required arguments

- No `issue_number` required.
- `help: true` appends `--help`.

## Optional arguments

- `adw_id` (accepted for schema compatibility; not forwarded to CLI)
- `args` (validated additional CLI args; protected flags rejected)

## Timeout and errors

- Default timeout: `120000ms`
- Timeout envelope: `ERROR: Failed to execute 'adw <command>' (timeout after 120000ms).`
- Non-zero exit envelope uses deterministic precedence: `stderr` → `stdout` → fallback.
