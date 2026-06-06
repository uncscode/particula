# Notebook Validation Guide

**Version:** 1.1.0
**Last Updated:** 2026-02-22

## Overview

Use the notebook tooling to validate `.ipynb` files before execution, keep `.ipynb`
and paired `.py` scripts in sync, and run notebooks and scripts safely with automatic backups.
Two entrypoints:
- `validate_notebook`: Validate/sync notebooks without executing code.
- `run_notebook`: Execute notebooks and `.py` scripts; validates first by default.

Skip rules and safety:
- Hidden paths and `.ipynb_checkpoints/` are skipped automatically.
- Line magics (`%magic`) are stripped before syntax checks; cell magics (`%%magic`)
  are skipped so validation is not blocked.
- `run_notebook` overwrites the source by default after execution and creates a
  `.ipynb.bak` backup unless disabled.

## Quick Start

### Validate a single notebook
```bash
validate_notebook docs/Examples/setup/template-init-tutorial.ipynb
```

### Validate recursively in a directory
```bash
validate_notebook docs/Examples/ --recursive
```

### JSON output for CI or tooling
```bash
validate_notebook docs/Examples/ --recursive --output-mode json
```

### Full verification (validation + docs build)
```bash
validate_notebook docs/Examples/ --recursive --full --output-mode json
python3 -m mkdocs build --strict
```

## Run with Validation

`run_notebook` validates before execution by default. Use `--skip-validation`
only when debugging validation itself.

Treat notebooks and scripts as untrusted code.

```bash
# Execute with validation (default), overwrite source, create backup
run_notebook docs/Examples/setup/template-init-tutorial.ipynb

# Execute without validation (not recommended except for debugging)
run_notebook docs/Examples/setup/template-init-tutorial.ipynb --skip-validation

# Preserve the original file and/or skip backup
run_notebook docs/Examples/setup/template-init-tutorial.ipynb --no-overwrite --no-backup
```

Defaults during execution:
- Overwrites the input notebook with executed output
- Writes a `.ipynb.bak` backup alongside the source
- You can also pass `--write-executed <path>` to save outputs elsewhere

## Jupytext Workflow

Use Jupytext to enable type checking and linting on notebook code.

- Convert notebook → script for mypy/ruff:
  ```bash
  validate_notebook docs/Examples/setup/template-init-tutorial.ipynb --convert-to-py
  mypy docs/Examples/setup/template-init-tutorial.py  # or ruff, pytest on the script
  ```

- Sync notebook and script (newer wins by mtime):
  ```bash
  validate_notebook docs/Examples/ --sync
  ```

- Check sync in CI (fails if out of sync, no writes):
  ```bash
  validate_notebook docs/Examples/ --check-sync --recursive
  ```

## Validation Errors

| Error type | Description |
|------------|-------------|
| Schema validation failed | Notebook JSON does not satisfy nbformat schema. |
| Missing `cell_type` | A cell is missing the required `cell_type` field. |
| Invalid `source` format | `source` is not a string or list of strings. |
| Syntax error in code cell | Python syntax error detected in a code cell. |

## Reading Validation Output (JSON)

Example (`--output-mode json`):
```json
{
  "path": "docs/Examples/setup/template-init-tutorial.ipynb",
  "valid": false,
  "errors": [
    {
      "type": "syntax_error",
      "cell_index": 2,
      "line": 4,
      "message": "invalid syntax"
    }
  ]
}
```

Exit codes:
- `0`: Valid / in-sync
- `1`: Validation or sync failure
- `2`: Tool/runtime error (e.g., dependency missing)

## CI Integration

Minimal CI job to validate and enforce sync:

```yaml
- name: Validate notebooks
  run: validate_notebook docs/Examples/ --recursive --output-mode json

- name: Enforce Jupytext sync
  run: validate_notebook docs/Examples/ --check-sync --recursive
```

## Troubleshooting

| Issue | Likely cause | Resolution |
|-------|--------------|------------|
| Path not found | Wrong path or glob | Re-run with a correct path; directories require `--recursive`. |
| Validation module not available | Tool not installed in environment | Install dev deps (`uv pip install -e ".[dev]"`) so `validate_notebook` is on PATH. |
| Jupytext missing | Dev dependency not installed | Install with `uv pip install -e ".[dev]"` or add `jupytext>=1.16`. |
| Unexpected skip | File is hidden or under `.ipynb_checkpoints/` | Move the notebook or disable skip logic in code if truly needed. |
| Magics reported unexpectedly | Non-Python magics present | `%` magics are stripped for syntax checks; `%%` cells are skipped. Convert to Python equivalents for validation. |

## Script Execution Mode

`run_notebook` can also execute `.py` scripts directly via `sys.executable` (the same
Python interpreter running the tool). This closes the gap between notebook-first and
script-first workflows.

### Auto-Detection

Pass a `.py` file path and `run_notebook` auto-detects script mode:

```bash
# Single script (auto-detected by .py extension)
run_notebook docs/Examples/panel-methods/regime-selection.py

# With output validation
run_notebook docs/Examples/panel-methods/regime-selection.py --expect-output "DataFrame" "plot"

# JSON output
run_notebook docs/Examples/panel-methods/regime-selection.py --output json
```

### Directory Collection with `--script`

When `notebookPath` is a directory, use the `--script` flag to collect `.py` files
instead of `.ipynb` notebooks:

```bash
# All scripts in a directory
run_notebook docs/Examples/panel-methods/ --script

# Recursive collection
run_notebook docs/Examples/panel-methods/ --script --recursive

# With timeout per script
run_notebook docs/Examples/panel-methods/ --script --recursive --timeout 300
```

### Script-Mode Behavior

- **Execution:** Scripts run via `subprocess.run([sys.executable, script_path])`,
  ensuring the same Python/venv as the tool itself.
- **Syntax validation:** By default, scripts are validated via `ast.parse` before
  execution. Use `--skip-validation` to bypass this check.
- **stdout/stderr capture:** Script output is captured and reported through the same
  summary/full/json output modes as notebooks.
- **`--expect-output`:** Validates expected substrings against captured **stdout only**
  (stderr is not included in validation).
- **Timeouts:** Per-script timeout via `subprocess.run(..., timeout=timeout)`.
- **Exit codes:** Non-zero exit codes produce structured failure results.

### Ignored Flags in Script Mode

The following notebook-specific flags are no-ops in script mode (a warning is logged
if passed):

- `--write-executed` (scripts don't produce executed copies)
- `--no-overwrite` (scripts are never modified)
- `--no-backup` (no backup needed for scripts)

### Script Output Formats

All three output modes work identically for scripts:

```bash
# Summary (default) - pass/fail counts and timing
run_notebook script.py --output summary

# Full - per-script stdout/stderr and details
run_notebook script.py --output full

# JSON - structured payload with stdout/stderr fields
run_notebook script.py --output json
```

Example JSON output:
```json
{
  "scripts_executed": 1,
  "scripts_passed": 1,
  "scripts_failed": 0,
  "total_execution_time": 0.42,
  "results": [
    {
      "script": "docs/Examples/panel-methods/regime-selection.py",
      "success": true,
      "execution_time": 0.42,
      "exit_code": 0,
      "error_message": null,
      "stdout": "Hello, world!\n",
      "stderr": "",
      "output_truncated": false
    }
  ],
  "validation_errors": {},
  "success": true,
  "truncated": false
}
```

## Separation of Tools

- `validate_notebook`: Validation and Jupytext convert/sync/check-sync only. Does **not** execute code.
- `run_notebook`: Executes notebooks and scripts after validation (unless `--skip-validation`). For notebooks: overwrites source and writes `.ipynb.bak` by default; supports `--no-overwrite`, `--no-backup`, and `--write-executed`. For scripts: runs via `sys.executable` subprocess with stdout/stderr capture; use `--script` flag for directory collection.
