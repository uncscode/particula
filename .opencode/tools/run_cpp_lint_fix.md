# run_cpp_lint_fix

Mutating C++ lint wrapper.

- Delegates to `.opencode/tools/run_cpp_linters.py`
- Accepts: `sourceDir`, optional `buildDir`, `linters`, `outputMode`, `timeout`
- Does **not** expose `autoFix`
- Always appends `--auto-fix`
- Validates `sourceDir`/`buildDir` are non-empty directory paths within repo root
- Returns deterministic `ERROR:` envelopes (`stderr -> stdout -> fallback`)

Example:

```json
{ "sourceDir": "src", "linters": ["clang-format"], "outputMode": "summary" }
```
