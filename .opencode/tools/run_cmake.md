# run_cmake (compatibility)

Compatibility wrapper for combined configure/build CMake flows.

## Preferred wrapper

- Use `run_cmake_configure` for configure-only operations.
- Use `run_cmake_build` for build-only operations.

## Compatibility status

- `run_cmake` remains available for combined configure/build flows.
- New integrations should prefer the split wrappers.

## Direct fields

- `preset`
- `sourceDir`
- `buildDir`
- `build`
- `buildTimeout`
- `timeout`
- `cmakeArgs`

Keep build context and timeout fields explicit.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `ninja`
- `jobs=<non-negative-int>`

## Examples

```json
{ "preset": "debug" }
{ "sourceDir": ".", "buildDir": "build/debug", "options": "ninja" }
{ "preset": "debug", "build": true, "options": "jobs=8" }
{ "preset": "release", "build": true, "buildTimeout": 3600 }
```

## Notes

- `preset` mode requires `CMakePresets.json` in the source tree.
- `ninja` is ignored when `preset` is provided.
- Repository-root confinement applies to manual `sourceDir` / `buildDir` inputs, and preset-derived build directories are also fail-closed when they resolve outside the repository root.
