# run_linters

Python linter wrapper for Ruff and mypy.

## Preferred wrapper

- Use `run_linters` for repository lint and type-check validation.

## Compatibility status

- Retained unsplit wrapper surface.
- This wrapper intentionally exposes validation-only and mutating modes through `autoFix`.

## Direct fields

- `autoFix`
- `targetDir`
- `ruffTimeout`
- `mypyTimeout`

## Bounded `options` tokens

- `output=<summary|full|json>`
- `linters=<ruff|mypy comma-list>`

## Examples

```json
{ "autoFix": false }
{ "autoFix": true, "targetDir": "." }
{ "options": "linters=ruff", "autoFix": false }
{ "options": "output=json linters=ruff,mypy" }
```

## Notes

- `autoFix: false` is validation-only and must not modify files.
- `autoFix: true` runs the Ruff fix/format/final-check flow before mypy.
- `targetDir` is optional; omitted values defer to project configuration.
