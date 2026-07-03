# run_bun_test

TypeScript/Bun test wrapper.

## Preferred wrapper

- Use `run_bun_test` for Bun-based wrapper-contract and TypeScript test execution.

## Compatibility status

- Retained unsplit wrapper surface.
- There is no split replacement for this family in the current migration window.

## Direct fields

- `testPath`
- `timeout`
- `minTests`
- `cwd`

Keep `testPath`, `timeout`, `minTests`, and `cwd` explicit.

## Bounded `options` tokens

- `output=<summary|full|json>`
- `test-filter=<value>`
- `fail-fast`

## Examples

```json
{ "testPath": "__tests__/run_pytest_basic.test.ts", "minTests": 1 }
{ "options": "test-filter=pytest", "minTests": 1 }
{ "options": "output=json fail-fast", "testPath": "__tests__/", "minTests": 1 }
{ "cwd": "/path/to/worktree/.opencode/tools", "testPath": "__tests__/run_bun_test.test.ts", "minTests": 1 }
```

## Notes

- `testPath` must not be blank or start with `-`.
- `cwd` and `testPath` are validated against the current repository root.
- Default `cwd` is `.opencode/tools/`.
