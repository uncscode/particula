# adw_notes_write

Focused ADW notes wrapper for mutating note operations.

## Commands
- `write`
- `write-from-state`

## Behavior
- requires non-empty `ref`
- `write-from-state` requires valid `adw_id`
- `fields` accepts ordered `[key, value]` entries, `{ key, value }` objects, plain objects, or JSON strings of those forms
- omitted `fields`, `fields: null`, and blank-string `fields` normalize to an empty field list
- blank note-field keys are rejected with a deterministic validation failure instead of being normalized or ignored
- malformed tuple/object/plain-object entries fail closed with the exact bad array index or object key plus null/missing/wrong-type classification
- preserves `VIRTUAL_ENV` so `uv run --active` can use the active environment
- sanitizes diagnostics by redacting secrets and absolute paths

## Contract Note
- Success is envelope-based:
  `ADW Notes Command: <command>\n\n<stdout>`
- Failures are deterministic:
  - delegated non-zero exits report an `ERROR:` envelope for `notes <command>`
  - execution errors report an `ERROR:` envelope for `notes <command>`
