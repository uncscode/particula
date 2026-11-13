# run_ctest

CTest wrapper for C++ test execution.

## Preferred wrapper

- Use `run_ctest` for CTest-based test execution.

## Compatibility status

- Retained unsplit wrapper surface.
- There is no split replacement for this family in the current migration window.

## Direct fields

- `buildDir`
- `timeout`
- `minTests`

## Bounded `options` tokens

- `output=<summary|full|json>`
- `test-filter=<regex>`
- `exclude-filter=<regex>`
- `parallel=<positive-int>`

## Examples

```json
{ "buildDir": "build" }
{ "buildDir": "build", "options": "test-filter=test_add" }
{ "buildDir": "build", "options": "exclude-filter=slow parallel=4" }
{ "buildDir": "build", "minTests": 5 }
```

## Notes

- `buildDir` is required.
- `buildDir` should point to a CMake build directory containing `CTestTestfile.cmake`.
- `timeout`, `minTests`, and `parallel` are validated as positive numbers.
