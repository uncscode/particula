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

Resolve the active worktree first, then pass that exact value as `cwd` to every
mutating command:

```jsonc
// 1) adw_spec read: get the isolated worktree path from ADW state
adw_spec({ "command": "read", "adw_id": "abc12345", "field": "worktree_path" })

// 2) Reuse that returned value as cwd in mutating adw_plans calls
{ "command": "create", "plan_type": "feature", "title": "Add auth", "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "update", "plan_id": "E17-F1", "status": "Ready", "options": "priority=P1 size=M", "cwd": "/path/to/trees/abc12345" }
```

```json
{ "command": "update-phase", "plan_id": "E17-F1", "phase_id": "E17-F1-P1", "options": "phase-status=Blocked size=M issue=42", "patch": "{\"actuals\":\"done\"}", "cwd": "/path/to/trees/abc12345" }
```

## Notes

- `cwd` is required for all mutating commands.
- Resolve `cwd` from `adw_spec read -> worktree_path` so plan writes stay anchored to the
  active worktree.
- `plan_type` is passed through as a string so runtime registry-driven plan types (for example `research`) are not wrapper-rejected.
- Prefer bounded `options` tokens for optional wrapper aliases in new docs (`status=<value>`,
  `phase-status=<value>`, `priority=<value>`, `size=<value>`, `after=<phase_id>`, `issue=<n>`,
  `clear-issue-number`), while direct `status` / `phase_status` values remain valid and raw JSON
  `patch` stays a direct-field exception.
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
