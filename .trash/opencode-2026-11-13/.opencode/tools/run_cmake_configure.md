# run_cmake_configure

Configure-only CMake wrapper.

## Preferred wrapper

- Use `run_cmake_configure` for configure-only operations.

## Compatibility status

- Preferred split wrapper for configuration.
- Never emits build flags.

## Direct fields

- `preset`
- `sourceDir`
- `buildDir`
- `timeout`
- `cmakeArgs`

## Bounded `options` tokens

- `output=<summary|full|json>`
- `ninja`

## Examples

```json
{ "preset": "ninja-release" }
{ "sourceDir": ".", "buildDir": "build/debug", "options": "ninja" }
{ "sourceDir": "example_cpp_dev", "cmakeArgs": ["-DENABLE_TESTS=ON"] }
```

## Notes

- Blank `preset`, `sourceDir`, `buildDir`, and `cmakeArgs` entries fail closed.
- Manual `sourceDir` and `buildDir` inputs must remain within the repository root. In preset mode those manual fields are ignored, while preset-derived build directories are still fail-closed if they escape the repository root.
