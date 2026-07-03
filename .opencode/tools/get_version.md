# get_version

Read-only wrapper for version discovery from repository metadata files.

## Supported input

- `file` is optional.
- Omit `file` to use the backend default lookup order.
- Blank or whitespace-only `file` values are treated as omitted.
- Non-string `file` values fail closed with:
  `ERROR: 'file' must be a string when provided.`

## Default lookup behavior

When `file` is omitted, the backend checks:

1. `pyproject.toml`
2. `package.json`

## Examples

```json
{}
```

```json
{ "file": "pyproject.toml" }
```

```json
{ "file": "package.json" }
```

## Failure behavior

- The wrapper prefers `stdout` when subprocess execution fails.
- If `stdout` is empty, it returns `ERROR: get_version failed` plus `stderr`.
- If both `stdout` and `stderr` are empty, it falls back to the subprocess message.
- Missing-runtime/script cases include targeted hints for:
  - unavailable `python3`
  - missing `.opencode/tools/get_version.py`
  - neutral ENOENT guidance when the backend cannot safely distinguish which path is missing

## Notes

- This wrapper is read-only and does not modify repository files.
- The wrapper invokes `.opencode/tools/get_version.py`.
- Backend reads are confined to the current repository/worktree root.
