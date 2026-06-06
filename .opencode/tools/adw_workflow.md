# adw_workflow tool

Workflow-only ADW wrapper for:

- `complete`
- `patch`
- `plan`
- `build`
- `test`
- `review`
- `document`
- `ship`

## Required arguments

- `issue_number` is required for normal execution.
- `help: true` bypasses `issue_number` requirement and appends `--help`.

## Optional arguments

- `adw_id` (8-char hex; normalized to lowercase)
- `model` (`light`, `base`, or `heavy`)
- `args` (validated additional CLI args; protected flags rejected)

## Timeout and errors

- Workflow timeout: `600000ms`
- Timeout envelope: `ERROR: Failed to execute 'adw <command>' (timeout after 600000ms).`
- Non-zero exit envelope uses deterministic precedence: `stderr` → `stdout` → fallback.
