# adw_notes_read

Focused ADW notes wrapper for read-only note access.

## Commands
- `show`

## Behavior
- requires non-empty `ref`
- returns parsed pretty-printed JSON on success
- preserves deterministic parse-failure output for malformed `show` JSON
- preserves `VIRTUAL_ENV` so `uv run --active` can use the active environment
- sanitizes diagnostics and malformed-output snippets by redacting secrets and absolute paths

## Contract Note
- Success returns parsed JSON text.
- Failures are deterministic:
  - delegated non-zero exits report an `ERROR:` envelope for `notes show`
  - execution errors report an `ERROR:` envelope for `notes show`
