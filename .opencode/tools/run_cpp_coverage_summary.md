# run_cpp_coverage_summary

Routine C++ coverage wrapper.

- Delegates to `.opencode/tools/run_cpp_coverage.py`
- Accepts: `buildDir`, optional `outputMode`, `threshold`, `timeout`
- Rejects advanced controls: `tool`, `filter`, `html`, `extraArgs`
- Use `run_cpp_coverage_advanced` for extended controls
- Validates `buildDir` as a non-empty directory path within repo root
- Returns deterministic `ERROR:` envelopes (`stderr -> stdout -> fallback`)

Example:

```json
{ "buildDir": "build", "threshold": 80, "outputMode": "summary" }
```
