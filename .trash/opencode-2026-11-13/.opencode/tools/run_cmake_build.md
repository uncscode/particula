# run_cmake_build

Build-only CMake wrapper.

## Preferred wrapper

- Use `run_cmake_build` for build-only operations.

## Compatibility status

- Preferred split wrapper for build execution.
- Requires an explicit build context.

## Direct fields

- `preset`
- `buildDir`
- `buildTimeout`
- `timeout`

## Bounded `options` tokens

- `output=<summary|full|json>`
- `jobs=<non-negative-int>`

## Examples

```json
{ "preset": "debug" }
{ "buildDir": "build/debug" }
{ "preset": "release", "options": "jobs=8", "buildTimeout": 3600 }
```

## Notes

- Provide either `preset` or `buildDir`.
- Manual `buildDir` inputs must remain within the repository root. When `preset` is used, `buildDir` is ignored and preset-derived build directories are still fail-closed if they escape the repository root.
- `jobs` may be `0` in the compatibility wrapper, but this split wrapper only emits `--jobs` when the parsed value is greater than zero.
