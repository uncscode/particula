# validate_notebook_readonly

Read-only wrapper for notebook validation and sync-state checks.

## Purpose

Use `validate_notebook_readonly` when you need **non-mutating** notebook checks only:

- structural/syntax validation
- check-sync status (`--check-sync`) for CI guardrails

## Supported options

- `notebookPath` (required)
- `recursive`
- `outputMode` (`summary|full|json`)
- `skipSyntax`
- `validationMode` (`fast|full`)
- `fast`
- `full`
- `checkSync`

## Not supported (fail-closed)

These mutating options are rejected before subprocess execution:

- `convertToPy`
- `convertToIpynb`
- `sync`
- `outputDir`

Error contract:

`ERROR: validate_notebook_readonly does not support mutating options (convertToPy, convertToIpynb, sync, outputDir). Use convert_notebook_to_py, convert_py_to_notebook, or sync_notebook_pair for mutating operations.`

## Examples

- Validation:
  - `validate_notebook_readonly({ notebookPath: 'docs/Examples/setup-template-init-tutorial.ipynb' })`
- Recursive validation:
  - `validate_notebook_readonly({ notebookPath: 'docs/Examples/', recursive: true, outputMode: 'json' })`
- Check-sync (read-only):
  - `validate_notebook_readonly({ notebookPath: 'docs/Examples/', recursive: true, checkSync: true })`

## Mutating workflows

For conversion/sync operations, prefer explicit mutating tools:

- `convert_notebook_to_py`
- `convert_py_to_notebook`
- `sync_notebook_pair`

`validate_notebook` remains compatibility-only for legacy mutating flows.
