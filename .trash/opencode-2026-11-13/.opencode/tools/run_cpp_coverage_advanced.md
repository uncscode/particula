# run_cpp_coverage_advanced

Advanced C++ coverage wrapper.

## Purpose

Use `run_cpp_coverage_advanced` when the routine summary wrapper is too narrow
and you need advanced coverage-tool selection or report-path controls.

## Supported fields

- Direct fields: `buildDir`, optional `threshold`, `timeout`, `filter`, `html`
- Bounded `options` tokens:
  - `output=<summary|full|json>`
  - `tool=<gcov|llvm-cov>`

## Behavior

- Delegates to `.opencode/tools/run_cpp_coverage.py`
- Does **not** accept direct `outputMode`, `tool`, or `extraArgs` fields; route tool selection through `options`
- Trims optional strings; blank `filter`/`html` values are omitted
- Validates `buildDir` directory confinement and `html` path confinement to repo root
- Canonicalizes validated `buildDir`/`html` paths before subprocess execution
- Allows safe nested new `html` output directories inside the repository when their nearest existing ancestor is repo-confined
- Returns deterministic failure diagnostics (`stdout -> stderr -> fallback`)

Example:

```json
{
  "buildDir": "build",
  "filter": "src/",
  "html": "coverage_html",
  "options": "output=json tool=llvm-cov"
}
```
