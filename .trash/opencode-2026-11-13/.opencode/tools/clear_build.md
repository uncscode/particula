# clear_build (compatibility)

Compatibility wrapper for build-directory preview and deletion.

## Preferred wrapper

- Use `clear_build_preview` to inspect what would be deleted.
- Use `clear_build_delete` for explicit destructive cleanup.

## Compatibility status

- `clear_build` remains available as a compatibility delegator.
- `dryRun: true` routes to preview semantics.
- `dryRun: false` requires `force: true` for destructive execution.

## Direct fields

- `buildDir`
- `dryRun`
- `force`

Keep destructive authorization fields explicit.

## Bounded `options` tokens

- None.

## Examples

```json
{ "buildDir": "build", "dryRun": true }
{ "buildDir": "build", "force": true }
{ "buildDir": "build/debug", "force": true }
```

## Notes

- Do not pass both `dryRun` and `force`.
- If `dryRun` is false, `force: true` is required.
