# Notebook Validation Guide

**Version:** 1.0.0
**Last Updated:** 2026-01-23

## Overview

Use the notebook tooling to validate `.ipynb` files before execution, keep `.ipynb`
and paired `.py` scripts in sync, and run notebooks safely with automatic backups.
Two entrypoints:
- `validate_notebook`: Validate/sync notebooks without executing code.
- `run_notebook`: Execute notebooks; validates first by default.

Skip rules and safety:
- Hidden paths and `.ipynb_checkpoints/` are skipped automatically.
- Line magics (`%magic`) are stripped before syntax checks; cell magics (`%%magic`)
  are skipped so validation is not blocked.
- `run_notebook` overwrites the source by default after execution and creates a
  `.ipynb.bak` backup unless disabled.

## Quick Start

### Validate a single notebook
```bash
validate_notebook path/to/notebook.ipynb
```

### Validate recursively in a directory
```bash
validate_notebook path/to/notebooks/ --recursive
```

### JSON output for CI or tooling
```bash
validate_notebook path/to/notebooks --recursive --output-mode json
```

## Run with Validation

`run_notebook` validates before execution by default. Use `--skip-validation`
only when debugging validation itself.

```bash
# Execute with validation (default), overwrite source, create backup
run_notebook examples/demo.ipynb

# Execute without validation (not recommended except for debugging)
run_notebook examples/demo.ipynb --skip-validation

# Preserve the original file and/or skip backup
run_notebook examples/demo.ipynb --no-overwrite --no-backup
```

Defaults during execution:
- Overwrites the input notebook with executed output
- Writes a `.ipynb.bak` backup alongside the source
- You can also pass `--write-executed <path>` to save outputs elsewhere

## Jupytext Workflow

Use Jupytext to enable type checking and linting on notebook code.

- Convert notebook → script for mypy/ruff:
  ```bash
  validate_notebook notebooks/model.ipynb --convert-to-py
  mypy notebooks/model.py  # or ruff, pytest on the script
  ```

- Sync notebook and script (newer wins by mtime):
  ```bash
  validate_notebook notebooks/ --sync
  ```

- Check sync in CI (fails if out of sync, no writes):
  ```bash
  validate_notebook notebooks/ --check-sync --recursive
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
  "path": "notebooks/broken.ipynb",
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
  run: validate_notebook notebooks/ --recursive --output json

- name: Enforce Jupytext sync
  run: validate_notebook notebooks/ --check-sync --recursive
```

## Troubleshooting

| Issue | Likely cause | Resolution |
|-------|--------------|------------|
| Path not found | Wrong path or glob | Re-run with a correct path; directories require `--recursive`. |
| Validation module not available | Tool not installed in environment | Install dev deps (`uv pip install -e ".[dev]"`) so `validate_notebook` is on PATH. |
| Jupytext missing | Dev dependency not installed | Install with `uv pip install -e ".[dev]"` or add `jupytext>=1.16`. |
| Unexpected skip | File is hidden or under `.ipynb_checkpoints/` | Move the notebook or disable skip logic in code if truly needed. |
| Magics reported unexpectedly | Non-Python magics present | `%` magics are stripped for syntax checks; `%%` cells are skipped. Convert to Python equivalents for validation. |

## File Management Best Practices

### Keep Executed Notebooks in `docs/`

Executed notebooks with outputs should remain in `docs/` folders for the MkDocs
documentation site. The outputs provide learning examples and visual guides for
users browsing the web documentation.

```bash
# Execute and overwrite in place (recommended for docs/)
run_notebook docs/Examples/tutorial.ipynb

# The executed notebook with outputs stays in docs/ for the website
```

### Move Temporary Files to `.trash/`

After validation or sync operations, move temporary `.py` sync files and `.bak`
backup files to `.trash/` to preserve git history while keeping the working
directory clean.

```bash
# After Jupytext sync, move the .py file to .trash/
move docs/Examples/tutorial.py .trash/docs/Examples/tutorial.py

# After execution, move backup to .trash/ if no longer needed
move docs/Examples/tutorial.ipynb.bak .trash/docs/Examples/tutorial.ipynb.bak
```

**Why `.trash/` instead of deletion?**
- Git tracks moves, preserving file history for audit
- Files can be restored if needed
- Maintainers can review `.trash/` before permanent removal
- Cleaner than renaming to `.TO_BE_DELETED` patterns

**Workflow summary:**
1. Run `validate_notebook --convert-to-py` or `--sync` as needed
2. Run linters/type checkers on the `.py` file
3. Move `.py` file to `.trash/` when done
4. Execute notebook with `run_notebook` (outputs saved in place)
5. Move `.bak` file to `.trash/` if backup is no longer needed

## Separation of Tools

- `validate_notebook`: Validation and Jupytext convert/sync/check-sync only. Does **not** execute code.
- `run_notebook`: Executes after validation (unless `--skip-validation`); overwrites source and writes `.ipynb.bak` by default; supports `--no-overwrite`, `--no-backup`, and `--write-executed` to control outputs.
