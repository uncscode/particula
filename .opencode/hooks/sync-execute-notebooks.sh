#!/usr/bin/env bash
set -euo pipefail

# Sync and execute paired notebooks for changed Jupytext Python files.
# Skips slow Simulation notebooks.

if [[ "$#" -eq 0 ]]; then
  echo "No files provided to hook; nothing to do." >&2
  exit 0
fi

declare -A seen

for py_file in "$@"; do
  # normalize to forward slashes in case git passes Windows-style paths
  py_file="${py_file//\\/\/}"

  # de-dup processing
  if [[ -n "${seen["$py_file"]+set}" ]]; then
    echo "Skipping duplicate file: $py_file" >&2
    continue
  fi
  seen["$py_file"]=1

  # guard: only process .py files under docs/Examples
  case "$py_file" in
    docs/Examples/Simulations/*)
      echo "Skipping Simulations notebook (excluded from pre-commit): $py_file" >&2
      continue
      ;;
    docs/Examples/*.py|docs/Examples/*/*.py|docs/Examples/*/*/*.py|docs/Examples/*/*/*/*.py)
      ;; # allowed
    *)
      echo "Skipping non-example file: $py_file" >&2
      continue
      ;;
  esac

  ipynb_file="${py_file%.py}.ipynb"

  if [[ ! -f "$ipynb_file" ]]; then
    echo "Missing paired notebook for $py_file (expected $ipynb_file)" >&2
    exit 1
  fi

  echo "[hook] Syncing notebook: $ipynb_file"
  if ! python3 .opencode/tool/validate_notebook.py "$ipynb_file" --sync; then
    echo "Sync failed for $ipynb_file" >&2
    exit 1
  fi

  echo "[hook] Executing notebook: $ipynb_file"
  if ! python3 .opencode/tool/run_notebook.py "$ipynb_file"; then
    echo "Execution failed for $ipynb_file" >&2
    exit 1
  fi

  echo "[hook] Staging pair: $py_file and $ipynb_file"
  git add "$py_file" "$ipynb_file"
done

echo "Hook completed successfully."
