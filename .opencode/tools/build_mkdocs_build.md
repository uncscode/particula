# build_mkdocs_build

Artifact-producing MkDocs wrapper.

## Ownership boundary

- Restricted to the `docs-validator` subagent only.
- Other agents should not call MkDocs wrappers directly; delegate documentation validation to `docs-validator`.

## Preferred wrapper

- Use `build_mkdocs_build` when you intentionally want MkDocs build artifacts.
- In docs-validator flows, prefer `build_mkdocs_validate` unless an explicit workflow contract authorizes artifact-producing build execution.

## Compatibility status

- Preferred split wrapper for normal build output.
- Produces build artifacts instead of forcing validate-only behavior.

## Direct fields

- `timeout`
- `cwd`
- `configFile`

## Bounded `options` tokens

- `output=<summary|full|json>`
- `strict`
- `clean=<true|false>`

## Examples

```json
{ }
{ "options": "output=json" }
{ "options": "strict clean=false" }
{ "configFile": "docs/mkdocs.yml" }
```

## Notes

- `cwd` and `configFile` are repository-root confined.
- `build_mkdocs_validate` remains the primary validation-safe path.
