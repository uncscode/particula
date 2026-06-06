# adw_plans_mutate

Mutation wrapper for `adw plans`. Prefer this split wrapper for new or updated
mutating integrations; keep `adw_plans` for compatibility or unified flows.

## Supported commands

- `create`
- `update`
- `add-phase`
- `update-phase`
- `scaffold-sections`

Read-only commands are intentionally rejected. Use `adw_plans_read` for inspection.

## Examples

```json
{ "command": "create", "plan_type": "feature", "title": "Add auth", "cwd": "./trees/abc" }
```

```json
{ "command": "update", "plan_id": "E17-F1", "status": "Ready", "cwd": "./trees/abc" }
```

```json
{ "command": "update-phase", "plan_id": "E17-F1", "phase_id": "E17-F1-P1", "patch": "{\"status\":\"Shipped\"}", "cwd": "./trees/abc" }
```

## Notes

- `cwd` is required for all mutating commands.
- `plan_type` is passed through as a string so runtime registry-driven plan types (for example `research`) are not wrapper-rejected.
- `patch` forwarding/validation semantics are unchanged from `adw_plans`.
- Compatibility and split wrappers share the same spawned-command failure handling:
  - `stderr` -> `stdout` -> message/fallback precedence
  - bounded truncation for long diagnostics
  - absolute-path redaction to `<path>`
  - targeted runtime/tooling and cwd/worktree hints when recognized
- Deterministic required-cwd error example:
  - `ERROR: update command requires 'cwd'`
- Additional deterministic pre-spawn path-validation examples:
  - `ERROR: cwd path does not exist: <path>`
  - `ERROR: cwd path is not a directory: <path>`
  - `ERROR: cwd path resolves outside repository root: <path> (canonical: <path>)`

Delegated failure envelope example:

```text
ERROR: adw plans update failed (exit N).
```

Routing hint:

- Keep this wrapper for write paths only; route read/list/validate/schema to `adw_plans_read`.
