# refactor_astgrep (compatibility)

Compatibility wrapper for split ast-grep tools.

## Routing

- `dryRun` omitted or `true` → delegates to `refactor_astgrep_preview` (non-mutating)
- `dryRun: false` → delegates to `refactor_astgrep_apply` (mutating)

## Migration note

Prefer direct usage of split wrappers:

- `refactor_astgrep_preview` for read-only preview
- `refactor_astgrep_apply` for file mutation

Legacy no-match behavior is preserved:

- Preview: `No matches found for pattern: <pattern>`
- Apply: `No files modified (no matches).`

`refactor_astgrep` remains an active compatibility delegator during the compatibility window.
Prefer split wrappers for new/updated integrations.

Downstream allowlist/compat cleanup remains in **E20-F11**; this doc does not alter runtime behavior.
