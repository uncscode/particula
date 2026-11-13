# run_cpp_lint_fix

Mutating C++ lint wrapper.

## Purpose

Use `run_cpp_lint_fix` when you want the split **mutating** C++ lint path.
Prefer it over legacy `run_cpp_linters` for new fix-capable integrations.

## Supported fields

- Direct fields: `sourceDir`, optional `buildDir`, `timeout`
- Bounded `options` tokens:
  - `output=<summary|full|json>`
  - `linters=<clang-format,clang-tidy,cppcheck[, ...]>`

## Behavior

- Delegates to `.opencode/tools/run_cpp_linters.py`
- Does **not** accept direct `outputMode` or `linters` fields; route both through `options`
- Does **not** expose `autoFix`
- Always appends `--auto-fix`
- Validates `sourceDir`/`buildDir` as non-empty directory paths within repo root
- Returns deterministic failure diagnostics (`stdout -> stderr -> fallback`)

Example:

```json
{ "sourceDir": "src", "options": "linters=clang-format output=summary" }
```
