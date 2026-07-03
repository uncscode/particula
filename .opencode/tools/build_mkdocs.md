# build_mkdocs (compatibility)

Compatibility wrapper for MkDocs build and validate-only flows.

## Ownership boundary

- Restricted to the `docs-validator` subagent only.
- Other agents should not call MkDocs wrappers directly; delegate documentation validation to `docs-validator`.

## Preferred wrapper

- Use `build_mkdocs_validate` for validation-only checks.
- Use `build_mkdocs_build` when you intentionally want build artifacts.
- In docs-validator flows, use `build_mkdocs_validate` first unless an explicit workflow contract authorizes a build-wrapper path.

## Compatibility status

- `build_mkdocs` remains available as a compatibility delegator.
- It routes based on the direct `validateOnly` field.

## Direct fields

- `timeout`
- `cwd`
- `configFile`
- `validateOnly`

Keep `validateOnly` explicit; it is not an `options` token.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `strict`
- `clean=<true|false>`

## Examples

```json
{ "validateOnly": true }
{ "options": "output=json" }
{ "options": "strict clean=false" }
{ "configFile": "docs/mkdocs.yml", "validateOnly": true }
```

## Notes

- `cwd` and `configFile` must resolve within the current repository root.
- `build_mkdocs_validate` is the primary validation-safe path for docs-validator flows.
