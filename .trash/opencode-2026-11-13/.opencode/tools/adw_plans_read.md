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
{ "command": "list", "json": true, "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "list", "options": "status=In Progress", "json": true, "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "show", "plan_id": "E17-F1", "json": true, "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "list-sections", "plan_id": "E17-F1", "json": true, "populate": true, "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "schema", "check": true, "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "list", "plan_type": "research", "cwd": "/path/to/trees/abc12345", "json": true }
```

## Notes

- `cwd` is required for every command. It may be the active repository root or
  a canonical linked worktree in the same Git repository. For workflow reads,
  obtain `worktree_path` with `adw_spec_read` and pass it unchanged.
- `json`, `check`, and `populate` are direct booleans. Only `status=<value>` remains an options token.
- Keep direct fields for required identifiers such as `plan_id` and optional
  direct filters such as `plan_type`, `parent`, and `lifecycle`.
- Example of stale shape to avoid on split wrappers: `{ "command": "list",
  "status": "In Progress" }`.
- Success/failure envelopes are preserved by the active split wrapper implementation.
- Active split plan wrappers use the following spawned-command failure handling:
  - `stderr` -> `stdout` -> message/fallback precedence
  - bounded truncation for long diagnostics
  - absolute-path redaction to `<path>`
  - targeted runtime/tooling and cwd/worktree hints when recognized
- `plan_type` is passed through as a string so runtime registry-driven plan types (for example `research`) are not wrapper-rejected.
- Deterministic invalid-cwd errors (when provided):
  - `ERROR: cwd path does not exist: <path>`
  - `ERROR: cwd path is not a directory: <path>`
  - `ERROR: cwd path is not an admitted root or linked worktree for this repository: <path>; use the repository root or workflow worktree_path.`

Delegated failure envelope example:

```text
ERROR: adw plans show failed (exit N).
```

Routing hint:

- For mutating commands (`create`, `update`, `add-phase`, `update-phase`, `scaffold-sections`), switch to `adw_plans_mutate` and provide `cwd`.

## Native runtime boundary

This wrapper's availability does not grant native `ToolBridge` activation or
execution. E39-F5's executable native subset is only `adw_spec_read`,
`adw_plans_show`, and `adw_plans_list_sections`; wrapper `list`, `validate`, and
`schema` commands, and this wrapper's required `cwd`, are not native root
selection or executable behavior. Native calls require a caller-selected
canonical `ProjectContext.root`, final-narrowed exact activation, and an
effective grant, and return bounded/redacted results. Correct a denied explicit
request, context, or grant through its owning surface, then make a fresh
request; do not infer generic reads, writes, network access, fallback, or retry.
