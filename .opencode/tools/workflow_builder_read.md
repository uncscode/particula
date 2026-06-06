# workflow_builder_read

Read-only wrapper for workflow-builder operations.

## Supported Commands

- `list`
- `get`
- `validate`

## Rejected Commands

- `create`
- `add_step`
- `remove_step`
- `update`

Rejected, omitted, or blank commands fail closed with deterministic errors:

`ERROR: workflow_builder_read does not support command '<cmd>'. Use: list, get, validate.`

## Example

```json
{ "command": "get", "workflow_name": "patch" }
```

Use `workflow_builder_mutate` for writes.
