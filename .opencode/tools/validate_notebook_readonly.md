# validate_notebook_readonly

Read-only wrapper for notebook validation and sync-state checks.

## Purpose

Use `validate_notebook_readonly` when you need **non-mutating** notebook checks only:

- structural or syntax validation
- check-sync status (`--check-sync`) for CI guardrails

Prefer this split wrapper for new read-only validation paths. `validate_notebook`
remains compatibility-only for legacy mutating conversion/sync flows.

## Supported fields

- `notebookPath` (required)
- `recursive`
- `checkSync`
- `options` with bounded tokens:
  - `output=<summary|full|json>`
  - `skip-syntax`
  - `validation-mode=<fast|full>`
  - `fast`
  - `full`

`checkSync` stays direct and is intentionally narrower than validation mode.
When `checkSync: true` is present, any explicit `options` value (including
`output=summary`) is rejected before subprocess execution.

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
  - `validate_notebook_readonly({ notebookPath: 'docs/Examples/', recursive: true, options: 'output=json' })`
- Fast validation:
  - `validate_notebook_readonly({ notebookPath: 'docs/Examples/', options: 'fast' })`
- Check-sync (read-only):
  - `validate_notebook_readonly({ notebookPath: 'docs/Examples/', recursive: true, checkSync: true })`

## Mutating workflows

For conversion or sync operations, use the explicit mutating split wrappers:

- `convert_notebook_to_py`
- `convert_py_to_notebook`
- `sync_notebook_pair`
