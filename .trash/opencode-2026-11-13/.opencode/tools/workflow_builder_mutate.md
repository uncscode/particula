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

Required mutate inputs also fail closed when missing or whitespace-only, including the `remove_step` selector requirement that at least one of `step_index` or `step_name` be provided. Blank optional string inputs are omitted from delegated CLI argv.

## Example

```json
{
  "command": "create",
  "workflow_name": "quick-deploy",
  "description": "Quick deployment workflow"
}
```

Use `workflow_builder_read` for inspection and validation flows.
