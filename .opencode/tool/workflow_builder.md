# Workflow Builder Tool

OpenCode tool for creating and validating ADW workflow JSON files using the `WorkflowBuilderTool` class.

## Overview

The workflow_builder tool provides programmatic access to ADW's workflow creation and validation system. It enables:
- Creating new workflow files with proper structure
- Adding/removing steps with validation
- Validating workflow JSON before saving
- Listing and inspecting existing workflows

This tool is used by the workflow-builder agent (`.opencode/agent/workflow-builder.md`) to provide interactive workflow creation.

## Usage in OpenCode

### List Available Workflows

```typescript
// Via tool call
workflow_builder({ command: "list" })
```

**Output:**
```
Available Workflows (8):

  • build (complete) - 4 steps - Build workflow - Implement code based on plan
  • complete (complete) - 6 steps - Complete workflow - Full validation cycle
  • patch (patch) - 4 steps - Patch workflow - Quick fixes with minimal overhead
  ...
```

### Get Workflow Details

```typescript
// Via tool call
workflow_builder({
  command: "get",
  workflow_name: "patch"
})
```

**Output:**
```
✅ Workflow: patch
Description: Patch workflow - Quick fixes with minimal overhead
Type: patch
Steps: 4
  1. Planning
  2. Building
  3. Testing
  4. Shipping
```

### Create New Workflow

```typescript
// Via tool call
workflow_builder({
  command: "create",
  workflow_name: "quick-deploy",
  description: "Quick deployment workflow",
  workflow_type: "custom"
})
```

**Output:**
```
✅ Created workflow 'quick-deploy' at .opencode/workflow/quick-deploy.json
```

### Add Step to Workflow

**Using slash command:**

```typescript
// Via tool call
workflow_builder({
  command: "add_step",
  workflow_name: "quick-deploy",
  step_json: JSON.stringify({
    type: "agent",
    name: "Build",
    command: "/implement",
    prompt: "Implement from spec_content",
    model: "base"
  })
})
```

**Using agent directly:**

```typescript
// Via tool call
workflow_builder({
  command: "add_step",
  workflow_name: "quick-deploy",
  step_json: JSON.stringify({
    type: "agent",
    name: "Run Tests",
    agent: "tester",
    prompt: "run",
    model: "base"
  })
})
```

**Note:** Provide either `command` (slash command) OR `agent` (agent name), not both.

**Model tiers:**
- `light` - Lightweight tasks (linting, simple tests)
- `base` - Standard tasks (implementation, review) [DEFAULT]
- `heavy` - Complex tasks (architecture, debugging)

**Output:**
```
✅ Added step 'Build' to workflow 'quick-deploy'
```

### Validate Workflow JSON

```typescript
// Via tool call
workflow_builder({
  command: "validate",
  workflow_json: JSON.stringify({
    name: "test-workflow",
    version: "1.0.0",
    description: "Test workflow",
    workflow_type: "custom",
    steps: [
      {
        type: "agent",
        name: "Build",
        command: "/implement",
        prompt: "Build it"
      }
    ]
  })
})
```

**Output:**
```
✅ Workflow JSON is valid

Parsed workflow:
{
  "name": "test-workflow",
  ...
}
```

### Remove Step from Workflow

```typescript
// By index
workflow_builder({
  command: "remove_step",
  workflow_name: "quick-deploy",
  step_index: 0
})

// Or by name
workflow_builder({
  command: "remove_step",
  workflow_name: "quick-deploy",
  step_name: "Build"
})
```

### Update Entire Workflow

```typescript
// Via tool call
workflow_builder({
  command: "update",
  workflow_name: "quick-deploy",
  workflow_json: JSON.stringify({
    name: "quick-deploy",
    version: "1.1.0",
    description: "Updated workflow",
    workflow_type: "custom",
    steps: [...]
  })
})
```

## Commands Reference

### create
Create new workflow file with initial structure.

**Required Arguments:**
- `workflow_name`: Name for the workflow (used as filename)
- `description`: Short description of what the workflow does

**Optional Arguments:**
- `version`: Semantic version (default: "1.0.0")
- `workflow_type`: complete, patch, or custom (default: "custom")

**Example:**
```typescript
workflow_builder({
  command: "create",
  workflow_name: "my-workflow",
  description: "My custom workflow",
  workflow_type: "custom"
})
```

### add_step
Add validated step to existing workflow.

**Required Arguments:**
- `workflow_name`: Name of workflow to modify
- `step_json`: JSON string of step to add

**Optional Arguments:**
- `position`: Index to insert at (default: append to end)

**Example:**
```typescript
workflow_builder({
  command: "add_step",
  workflow_name: "my-workflow",
  step_json: JSON.stringify({
    type: "agent",
    name: "Test",
    command: "/test",
    prompt: "Run tests",
    model: "light"
  }),
  position: 1  // Insert at index 1
})
```

### remove_step
Remove step from workflow.

**Required Arguments:**
- `workflow_name`: Name of workflow to modify
- Either `step_index` (zero-based) OR `step_name`

**Example:**
```typescript
// Remove by index
workflow_builder({
  command: "remove_step",
  workflow_name: "my-workflow",
  step_index: 2
})

// Remove by name
workflow_builder({
  command: "remove_step",
  workflow_name: "my-workflow",
  step_name: "placeholder"
})
```

### get
Retrieve workflow details.

**Required Arguments:**
- `workflow_name`: Name of workflow to retrieve

**Example:**
```typescript
workflow_builder({
  command: "get",
  workflow_name: "my-workflow"
})
```

### list
List all available workflows.

**No Arguments Required**

**Example:**
```typescript
workflow_builder({ command: "list" })
```

### update
Update entire workflow with validated JSON.

**Required Arguments:**
- `workflow_name`: Name of workflow to update
- `workflow_json`: Complete workflow JSON string

**Example:**
```typescript
workflow_builder({
  command: "update",
  workflow_name: "my-workflow",
  workflow_json: JSON.stringify({
    name: "my-workflow",
    version: "1.1.0",
    description: "Updated",
    workflow_type: "custom",
    steps: [...]
  })
})
```

### validate
Validate workflow JSON without saving.

**Required Arguments:**
- `workflow_json`: Workflow JSON string to validate

**Example:**
```typescript
workflow_builder({
  command: "validate",
  workflow_json: JSON.stringify({
    name: "test",
    version: "1.0.0",
    description: "Test",
    workflow_type: "custom",
    steps: [...]
  })
})
```

## Output Modes

All commands support three output modes via the `output` parameter:

### summary (default)
Human-readable summary with key information.
```typescript
workflow_builder({ command: "list", output: "summary" })
```

### full
Complete details including full JSON.
```typescript
workflow_builder({ command: "get", workflow_name: "patch", output: "full" })
```

### json
Structured JSON for programmatic use.
```typescript
workflow_builder({ command: "list", output: "json" })
```

**Example JSON output:**
```json
{
  "workflows": ["build", "complete", "patch", ...],
  "count": 8
}
```

## Error Handling

Errors are returned as part of the tool output, making them visible to the LLM:

### Validation Error
```
❌ Validation failed:
Schema validation failed: {'type': 'invalid'} is not valid under any of the given schemas
```

### Missing Workflow
```
❌ Workflow 'nonexistent' not found at .opencode/workflow/nonexistent.json
```

### Missing Required Argument
```
ERROR: 'add_step' requires workflow_name and step_json
```

### Invalid Step JSON
```
❌ Step validation failed:
  - Agent step must provide either 'agent' or 'command' field
```

## CLI Usage

The tool can also be used directly from the command line:

```bash
# List workflows
python3 .opencode/tool/workflow_builder.py list

# Get workflow details
python3 .opencode/tool/workflow_builder.py get --workflow-name patch

# Create workflow
python3 .opencode/tool/workflow_builder.py create \
  --workflow-name my-workflow \
  --description "My custom workflow" \
  --workflow-type custom

# Validate JSON
python3 .opencode/tool/workflow_builder.py validate \
  --workflow-json '{"name":"test",...}'

# Get help
python3 .opencode/tool/workflow_builder.py --help
```

## Integration with Workflow-Builder Agent

The workflow-builder agent (`.opencode/agent/workflow-builder.md`) uses this tool to provide interactive workflow creation:

1. User invokes `/create-workflow`
2. Agent guides user through workflow creation
3. Agent calls this tool to create, validate, and modify workflows
4. All validation happens incrementally via tool calls
5. Final workflow is saved to `.opencode/workflow/`

## Step Schema

Steps must conform to the workflow step schema. Valid step types:

### Agent Step
```json
{
  "type": "agent",
  "name": "Step Name",
  "command": "/implement",
  "prompt": "Instructions for agent",
  "model": "base",
  "if": "state.needs_implementation == true"  // optional
}
```

### Workflow Step
```json
{
  "type": "workflow",
  "name": "Run Sub-Workflow",
  "workflow": "another-workflow",
  "skip_if": "state.is_patch == true"  // optional
}
```

## Workflow Types

Three standard workflow types:

### complete
Full validation workflow with all phases:
- Planning
- Building  
- Testing
- Review
- Documentation
- Shipping

### patch
Quick fix workflow:
- Planning
- Building
- Shipping (skips test/review/docs)

### custom
User-defined steps and execution flow.

## Files Created

Workflows are saved to:
```
.opencode/workflow/<workflow-name>.json
```

## See Also

- `.opencode/agent/workflow-builder.md` - Interactive workflow builder agent
- `.opencode/command/create-workflow.md` - `/create-workflow` slash command
- `adw/workflows/engine/builder.py` - WorkflowBuilderTool implementation
- `adw/workflows/engine/validator.py` - Step validation logic
- `adw/workflows/engine/models.py` - Workflow schema validation
- `docs/Examples/workflow-builder-interactive.md` - Interactive builder guide
