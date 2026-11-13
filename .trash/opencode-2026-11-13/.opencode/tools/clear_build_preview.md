# clear_build_preview

Read-only build cleanup preview wrapper.

## Preferred wrapper

- Use `clear_build_preview` before any destructive build cleanup.

## Compatibility status

- Preferred split wrapper for preview-only cleanup.
- Always runs with `--dry-run`.

## Direct fields

- `buildDir`

## Bounded `options` tokens

- None.

## Examples

```json
{ }
{ "buildDir": "build/debug" }
```

## Notes

- Blank `buildDir` values normalize to `build`.
- This wrapper never accepts destructive authorization flags.
