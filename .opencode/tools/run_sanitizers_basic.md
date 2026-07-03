# run_sanitizers_basic

Routine sanitizer wrapper.

## Purpose

Use `run_sanitizers_basic` for standard ASAN, TSAN, or UBSAN execution.
Prefer this split wrapper for new routine sanitizer runs; keep legacy
`run_sanitizers` for compatibility routing only.

## Supported fields

- Direct fields: `buildDir`, `executable`, `sanitizer`
- Optional direct fields: `timeout`, `outputMode`

## Behavior

- Delegates to `.opencode/tools/run_sanitizers.py` through the shared sanitizer wrapper path
- Rejects advanced-only keys by presence: `suppressions`, `options`, `normalDuration`, `extraArgs`
- Use `run_sanitizers_advanced` when you need any advanced-only control
- Validates repository confinement, build-directory shape, executable input, and sanitizer allowlist before subprocess execution
- Returns deterministic failure diagnostics (`stdout -> stderr -> fallback`)

## Example

```json
{
  "buildDir": "build",
  "executable": "bin/my_tests",
  "sanitizer": "asan",
  "outputMode": "summary"
}
```
