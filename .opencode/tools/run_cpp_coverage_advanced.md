# run_cpp_coverage_advanced

Advanced C++ coverage wrapper.

- Delegates to `.opencode/tools/run_cpp_coverage.py`
- Accepts routine inputs: `buildDir`, optional `outputMode`, `threshold`, `timeout`
- Also accepts advanced inputs: `tool` (`gcov|llvm-cov`), `filter`, `html`
- Trims optional strings; blank `filter`/`html` values are omitted
- Validates `buildDir` directory confinement and `html` path confinement to repo root
- Returns deterministic `ERROR:` envelopes (`stderr -> stdout -> fallback`)

Example:

```json
{
  "buildDir": "build",
  "outputMode": "json",
  "tool": "llvm-cov",
  "filter": "src/",
  "html": "coverage_html"
}
```
