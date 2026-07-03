# get_datetime

Read-only wrapper for current date/time lookup in UTC or Denver local time.

## Supported inputs

- `format` is optional and defaults to `date`.
- Allowed `format` values:
  - `date`
  - `datetime`
  - `human`
- `localtime` is optional.
  - `false` or omitted: UTC
  - `true`: `America/Denver`
- Non-boolean `localtime` values fail closed with:
  `ERROR: 'localtime' must be a boolean when provided.`

Invalid `format` values fail closed with:
`ERROR: Invalid format. Expected one of: date, datetime, human.`

## Output guarantees

- `format: "date"` returns `YYYY-MM-DD`.
- `format: "datetime"` in UTC returns ISO 8601 without fractional seconds.
- `format: "datetime", localtime: true` returns a Denver-local timestamp with
  an explicit numeric offset such as `-06:00`.
- `format: "human"` returns a human-readable string in the selected timezone.

## Examples

```json
{}
```

```json
{ "format": "datetime" }
```

```json
{ "format": "datetime", "localtime": true }
```
