# adw_plans_read

Read-only wrapper for `adw plans`. Prefer this split wrapper for new or updated
read-only integrations; keep `adw_plans` for compatibility or unified flows.

## Supported commands

- `list`
- `show`
- `validate`
- `schema`
- `list-sections`

Mutating commands are intentionally rejected. Use `adw_plans_mutate` for writes.

## Examples

```json
{ "command": "list", "json": true }
```

```json
{ "command": "show", "plan_id": "E17-F1" }
```

```json
{ "command": "list-sections", "plan_id": "E17-F1", "populate": true }
```

## Notes

- Optional `cwd` is accepted for read commands and validated when provided.
- Success/failure envelopes are preserved via delegation to `adw_plans`.
- Compatibility and split wrappers share the same spawned-command failure handling:
  - `stderr` -> `stdout` -> message/fallback precedence
  - bounded truncation for long diagnostics
  - absolute-path redaction to `<path>`
  - targeted runtime/tooling and cwd/worktree hints when recognized
- `plan_type` is passed through as a string so runtime registry-driven plan types (for example `research`) are not wrapper-rejected.
- Deterministic invalid-cwd errors (when provided):
  - `ERROR: cwd path does not exist: <path>`
  - `ERROR: cwd path is not a directory: <path>`
  - `ERROR: cwd path resolves outside repository root: <path> (canonical: <path>)`

Delegated failure envelope example:

```text
ERROR: adw plans show failed (exit N).
```

Routing hint:

- For mutating commands (`create`, `update`, `add-phase`, `update-phase`, `scaffold-sections`), switch to `adw_plans_mutate` and provide `cwd`.
