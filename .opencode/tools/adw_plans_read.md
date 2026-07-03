# adw_plans_read

Read-only wrapper for `adw plans`. Prefer this split wrapper for all active
read-only integrations.

## Supported commands

- `list`
- `show`
- `validate`
- `schema`
- `list-sections`

Mutating commands are intentionally rejected. Use `adw_plans_mutate` for writes.

## Examples

```json
{ "command": "list", "options": "json" }
```

```json
{ "command": "list", "options": "status=In Progress json" }
```

```json
{ "command": "show", "plan_id": "E17-F1", "options": "json" }
```

```json
{ "command": "list-sections", "plan_id": "E17-F1", "options": "json populate" }
```

```json
{ "command": "schema", "options": "check" }
```

```json
{ "command": "list", "plan_type": "research", "cwd": "/path/to/trees/abc12345", "options": "json" }
```

## Notes

- Optional `cwd` is accepted for read commands and validated when provided.
- Use bounded `options` tokens for optional wrapper aliases (`json`, `check`, `populate`,
  `status=<value>`). Direct `status` is not part of the split-wrapper contract.
- Keep direct fields for required identifiers such as `plan_id` and optional
  direct filters such as `plan_type`, `parent`, and `lifecycle`.
- Example of stale shape to avoid on split wrappers: `{ "command": "list",
  "status": "In Progress" }`.
- Success/failure envelopes are preserved by the active split wrapper implementation.
- Compatibility and split wrappers share the same spawned-command failure handling:
  - `stderr` -> `stdout` -> message/fallback precedence
  - bounded truncation for long diagnostics
  - absolute-path redaction to `<path>`
  - targeted runtime/tooling and cwd/worktree hints when recognized
- `plan_type` is passed through as a string so runtime registry-driven plan types (for example `research`) are not wrapper-rejected.
- Deterministic invalid-cwd errors (when provided):
  - `ERROR: cwd path does not exist: <path>`
  - `ERROR: cwd path is not a directory: <path>`
  - `ERROR: cwd path is not a repository/worktree root: <path> (missing .git metadata at <path>)`
  - `ERROR: cwd path resolves outside repository root: <path> (canonical: <path>)`

Delegated failure envelope example:

```text
ERROR: adw plans show failed (exit N).
```

Routing hint:

- For mutating commands (`create`, `update`, `add-phase`, `update-phase`, `scaffold-sections`), switch to `adw_plans_mutate` and provide `cwd`.
