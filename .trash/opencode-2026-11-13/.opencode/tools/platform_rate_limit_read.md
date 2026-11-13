# platform_rate_limit_read

Read-only wrapper for `adw platform rate-limit`.

## Supported commands

- `rate-limit`

## Arguments

- `command`: required command name
- `output_format`: optional `text` or `json`
- `prefer_scope`: optional `fork` or `upstream`
- `help`: show delegated CLI help

## Notes

- Optional args are validated before subprocess execution.
- JSON-mode failure output is returned only when it remains structured after sanitization.
- Error envelopes prefer `STDERR` then `STDOUT`.

## Example

```json
{ "command": "rate-limit", "output_format": "json", "prefer_scope": "upstream" }
```
