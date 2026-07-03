# run_cpp_coverage_summary

Routine C++ coverage wrapper.

## Purpose

Use `run_cpp_coverage_summary` for routine threshold-focused coverage runs.
Prefer this split wrapper for standard summaries; use
`run_cpp_coverage_advanced` only when you need advanced tool or report-path
controls.

## Supported fields

- Direct fields: `buildDir`, optional `threshold`, `timeout`
- Bounded `options` token: `output=<summary|full|json>`

## Behavior

- Delegates to `.opencode/tools/run_cpp_coverage.py`
- Does **not** accept direct `outputMode` or `tool` fields; keep output on `options`
- Rejects advanced controls: `tool`, `filter`, `html`
- `extraArgs` is not part of the shipped C++ coverage wrapper contract
- Use `run_cpp_coverage_advanced` for extended controls
- Validates `buildDir` as a non-empty directory path within repo root and forwards the canonical validated path to the backend
- Returns deterministic failure diagnostics (`stdout -> stderr -> fallback`)

Example:

```json
{ "buildDir": "build", "threshold": 80, "options": "output=summary" }
```
