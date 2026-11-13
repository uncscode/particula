# run_sanitizers_advanced

Advanced sanitizer wrapper.

## Purpose

Use `run_sanitizers_advanced` when routine sanitizer execution needs extra
runtime controls.

## Supported fields

- Direct base fields: `buildDir`, `executable`, `sanitizer`
- Optional direct fields: `timeout`, `outputMode`
- Advanced-only direct fields: `suppressions`, `options`, `normalDuration`, `extraArgs`

## Behavior

- Delegates to `.opencode/tools/run_sanitizers.py` through the shared sanitizer wrapper path
- Keeps the sanitizer family on direct fields; this wrapper does **not** use a bounded `options` token carrier
- Validates repository confinement, build-directory shape, executable input, sanitizer allowlist, timeout guards, and advanced field types before subprocess execution
- Returns deterministic failure diagnostics (`stdout -> stderr -> fallback`)

## Example

```json
{
  "buildDir": "build",
  "executable": "bin/my_tests",
  "sanitizer": "tsan",
  "outputMode": "json",
  "options": "halt_on_error=1",
  "normalDuration": 120,
  "extraArgs": ["--gtest_filter=Threading.*"]
}
```
