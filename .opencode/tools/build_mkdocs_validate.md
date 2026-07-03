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
- Always runs validate-only mode and does not persist build artifacts.

## Direct fields

- `timeout`
- `cwd`
- `configFile`

Default wrapper timeout is `120` seconds. The backing Python helper accepts a
longer runtime budget, but this wrapper intentionally keeps the shorter default
so validation-only calls fail fast unless the caller opts into a larger timeout.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `strict`
- `clean=<true|false>`

## Examples

```json
{ }
{ "options": "strict" }
{ "configFile": "docs/mkdocs.yml" }
{ "cwd": "/path/to/worktree", "options": "output=summary" }
```

## Notes

- `cwd` must resolve to a directory inside the repository root.
- `configFile` must resolve inside the repository root.
- `clean=false` emits `--no-clean` even in validate-only mode.
- Timeout failures are reported explicitly as validation timeouts, not generic
  stderr failures.
- `strict` maps to MkDocs strict mode and escalates warnings into failure.
