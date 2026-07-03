# clear_build_delete

Destructive build cleanup wrapper.

## Preferred wrapper

- Use `clear_build_delete` only after reviewing a preview.

## Compatibility status

- Preferred split wrapper for destructive cleanup.
- Fails closed unless `force: true` is provided.

## Direct fields

- `buildDir`
- `force`

## Bounded `options` tokens

- None.

## Examples

```json
{ "force": true }
{ "buildDir": "build/debug", "force": true }
```

## Notes

- `force` is required and must be `true`.
- Blank `buildDir` values normalize to `build`.
