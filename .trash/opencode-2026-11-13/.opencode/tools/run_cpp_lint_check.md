# run_cpp_lint_check

Non-mutating C++ lint wrapper.

## Purpose

Use `run_cpp_lint_check` for routine **read-only** C++ lint validation.
Prefer this split wrapper for new validation-only flows; keep legacy
`run_cpp_linters` for compatibility paths only.

## Supported fields

- Direct fields: `sourceDir`, optional `buildDir`, `timeout`
- Bounded `options` tokens:
  - `output=<summary|full|json>`
  - `linters=<clang-format,clang-tidy,cppcheck[, ...]>`

## Behavior

- Delegates to `.opencode/tools/run_cpp_linters.py`
- Does **not** accept direct `outputMode` or `linters` fields; route both through `options`
- Does **not** expose `autoFix`
- Never appends `--auto-fix`
- Validates `sourceDir`/`buildDir` as non-empty directory paths within repo root
- Returns deterministic failure diagnostics (`stdout -> stderr -> fallback`)

Example:

```json
{ "sourceDir": "src", "buildDir": "build", "options": "linters=clang-tidy output=summary" }
```
