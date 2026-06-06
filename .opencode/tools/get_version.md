# get_version

Read-only wrapper for version discovery from repository metadata files.

## Supported inputs

- `file` omitted: auto-detect from standard project files
- `file: "pyproject.toml"`
- `file: "package.json"`

## Example

```json
{ "file": "pyproject.toml" }
```

## Notes

- This wrapper is read-only and does not modify repository files.
- Failure envelopes remain deterministic (`ERROR:`) with runtime/script hints when applicable.
