# workflow_builder_mutate

Mutation wrapper for workflow-builder operations.

## Supported Commands

- `create`
- `add_step`
- `remove_step`
- `update`

## Rejected Commands

- `list`
- `get`
- `validate`

Rejected, omitted, or blank commands fail closed with deterministic errors:

`ERROR: workflow_builder_mutate does not support command '<cmd>'. Use: create, add_step, remove_step, update.`

## Example

```json
{
  "command": "create",
  "workflow_name": "quick-deploy",
  "description": "Quick deployment workflow"
}
```

Use `workflow_builder_read` for inspection and validation flows.
