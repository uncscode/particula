# adw_notes_write

Focused ADW notes wrapper for mutating note operations.

## Commands
- `write`
- `write-from-state`

## Behavior
- requires non-empty `ref`
- `write-from-state` requires valid `adw_id`; an omitted field retains its
  persisted value
- `fields` accepts ordered `[key, string|null]` entries, `{ key, value }`
  objects, plain objects, or JSON strings of those forms
- permitted keys are `plan_summary`, `architecture_notes`,
  `discovered_context`, and `review_findings`; only `architecture_notes` and
  `review_findings` may be `null`
- ordered entries preserve duplicate keys; the final supplied value is used by
  the CLI
- each entry is serialized as a separate
  `--field-json JSON.stringify([key, value])` argument, which preserves the
  distinction between JSON `null`, `""`, and the string `"null"`
- omitted `fields`, `fields: null`, and blank-string `fields` normalize to an
  empty field list
- blank note-field keys are rejected with a deterministic validation failure
  instead of being normalized or ignored
- malformed tuple/object/plain-object entries and unallowlisted or non-nullable
  keys fail closed before spawning, with the exact bad array index or object key
  plus classification
- preserves `VIRTUAL_ENV` so `uv run --active` can use the active environment
- sanitizes diagnostics by redacting secrets and absolute paths

## Field Updates

`write` includes only supplied agent fields. In `write-from-state`, a supplied
field replaces a valid persisted field, while omission retains its persisted
value. Only `architecture_notes` and `review_findings` accept JSON `null`;
`plan_summary` and `discovered_context` require strings. An empty string is an
intentional value for every field.

Ordered duplicates resolve to their final supplied value. Omission in
`write-from-state` retains a valid persisted field; only `architecture_notes`
and `review_findings` accept JSON `null`. The literal string `"null"` remains
text, not a clear request.

Markdown is field data. The wrapper forwards it unchanged; it does not render,
trim, or normalize headings, links, fragments, emphasis, or prose. Existing
note size limits and secret redaction still apply.

### CLI transport

The underlying CLI retains string-only legacy pairs for compatibility, but
`--field` and `--field-json` cannot be combined. A legacy field is the argument
pair `--field plan_summary "Completed #3567"`. The focused wrapper serializes
each supplied entry as repeatable `--field-json` arguments. Use an exact JSON
`[key, string|null]` entry whenever a null value must be represented.

Examples:

```typescript
adw_notes_write({
  command: "write-from-state",
  ref: "HEAD",
  adw_id: "abc12345",
  fields: [
    ["plan_summary", "[#3567](https://example.test/issues/3567#discussion)"],
    ["architecture_notes", null],
    ["discovered_context", ""],
    ["review_findings", "null"],
  ],
})
```

The example clears `architecture_notes`, intentionally sets
`discovered_context` to empty text, and stores `"null"` as text in
`review_findings`.

## Contract Note
- Success is envelope-based:
  `ADW Notes Command: <command>\n\n<stdout>`
- Failures are deterministic:
  - delegated non-zero exits report an `ERROR:` envelope for `notes <command>`
  - execution errors report an `ERROR:` envelope for `notes <command>`
