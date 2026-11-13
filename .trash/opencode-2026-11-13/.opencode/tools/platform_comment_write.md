# platform_comment_write

Executes `adw platform comment` with strict validation and deterministic errors.

## Command

- `comment`

## Required args

- `issue_number` (strict positive integer token; digits only; leading zeros allowed but all-zero rejected)
- `body` (non-empty after trim)

## Optional args

- `prefer_scope`: `fork|upstream`
- `help`: when `true`, runs `--help` and bypasses required-arg validation

## Error envelope

Subprocess failures return:

- `ERROR: Failed to execute 'adw platform comment'`
- diagnostics in deterministic order: `STDERR` then `STDOUT`

## Validation and Recovery Examples

Pre-spawn validation example (required body):

```text
ERROR: comment command requires non-empty body
```

Delegated failure example:

```text
ERROR: Failed to execute 'adw platform comment'
```

Routing hint:

- Prefer `platform_comment_write` for comment writes; treat
  `platform_operations` as a compatibility-only delegation path for legacy
  flows.

## Example

```json
{"command":"comment","issue_number":"123","body":"LGTM","prefer_scope":"upstream"}
```
