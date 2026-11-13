# platform_label_write

Wrapper for `adw platform add-labels` and `remove-labels`.

## Supported commands

- `add-labels`
- `remove-labels`

## Required arguments

- `command`: required command name
- `issue_number`: required positive integer token
- `labels`: required comma-separated list with at least one non-empty label

## Optional arguments

- `output_format`: `text|json`
- `prefer_scope`: `fork|upstream`
- `help`: bypasses required-argument validation and delegates `--help`

## Notes

- Label lists are trimmed and normalized before command assembly.
- JSON-mode failure payloads are only returned when they remain structured after sanitization.
- Failure envelopes use deterministic `STDERR` then `STDOUT` ordering.

## Example

```json
{ "command": "add-labels", "issue_number": "123", "labels": "bug,docs", "output_format": "json" }
```
