# workflow_builder (compatibility)

Compatibility wrapper for workflow-builder operations.

Use split wrappers for new and updated workflows:

- `workflow_builder_read` for read-only commands: `list`, `get`, `validate`
- `workflow_builder_mutate` for mutating commands: `create`, `add_step`, `remove_step`, `update`

`workflow_builder` remains active during the compatibility window and supports the full command set.

Unsupported or blank commands fail closed before delegation with:

`ERROR: workflow_builder does not support command '<cmd>'. Use: create, add_step, remove_step, get, list, update, validate.`

Command-specific required fields are validated before subprocess execution, and blank optional string values are omitted instead of being forwarded as empty CLI flags.

For `remove_step`, provide at least one selector: `step_index` and/or `step_name`.
If both are provided, current backend precedence is preserved.

## Preferred Routing

| Intent | Preferred wrapper |
|---|---|
| Inspect/list workflows | `workflow_builder_read` |
| Validate workflow JSON | `workflow_builder_read` |
| Create/update workflows | `workflow_builder_mutate` |
| Add/remove steps | `workflow_builder_mutate` |

## Supported Commands (compatibility surface)

- `list`
- `get`
- `validate`
- `create`
- `add_step`
- `remove_step`
- `update`

## Examples

Read-only:

```json
{ "command": "list" }
```

Mutating:

```json
{
  "command": "create",
  "workflow_name": "quick-deploy",
  "description": "Quick deployment workflow",
  "workflow_type": "custom"
}
```

## Migration Guidance

- Prefer split wrappers in docs, agents, and new tool integrations.
- Keep `workflow_builder` references only for compatibility and legacy call paths.
- Downstream allowlist/compat cleanup remains in **E20-F11**; this document does not change runtime behavior.
