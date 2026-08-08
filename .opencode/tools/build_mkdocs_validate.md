# build_mkdocs_validate

Validation-safe MkDocs wrapper.

## Ownership boundary

- Restricted to the `docs-validator` subagent only.
- Other agents must not call MkDocs wrappers directly; delegate documentation validation to `docs-validator`.

## Preferred wrapper

- Use `build_mkdocs_validate` for validation-only docs checks.
- This is the primary validation-safe MkDocs path for docs-validator flows.

## Compatibility status

- Preferred split wrapper for validator-style flows.
- Always runs strict, validate-only mode in a cleaned-up temporary site directory.

## Direct fields

- `timeout`
- `cwd`
- `configFile`

The default and maximum wrapper timeout is `300` seconds. Non-finite,
non-positive, and over-limit values are rejected before the helper launches.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `clean=<true|false>`

## Examples

```json
{ }
{ "configFile": "mkdocs.yml" }
{ "cwd": "/path/to/worktree", "options": "output=summary" }
```

## Notes

- `cwd` must resolve to a directory inside the repository root.
- `configFile` must resolve inside the repository root.
- `clean=false` emits `--no-clean` even in validate-only mode.
- `strict` is not an option: it is always enabled.
- Every outcome exposes one bounded, sanitized diagnostic model with outcome,
  stage, progress, combined output, truncation metadata, warnings, and
  conservative warning attribution. JSON never exposes raw stdout or stderr.
- A timeout is a failure, never a successful validation result.
