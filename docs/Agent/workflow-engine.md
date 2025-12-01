# Dynamic Workflow Engine

**Version:** 2.2.0  
**Last Updated:** 2025-11-29

## Overview

The ADW dynamic workflow engine enables declarative, JSON-based workflow definitions that can be composed, extended, and customized without modifying Python code. This provides a flexible, maintainable approach to defining and executing development workflows.

### Key Benefits

- **Declarative** - Define workflows in JSON instead of Python code
- **Composable** - Build complex workflows from reusable components
- **Extensible** - Add custom workflows without code changes
- **Validated** - Schema validation catches errors before execution
- **Conditional** - Execute steps based on workflow state
- **Robust** - Built-in retry logic with exponential backoff

## Getting Started

### Creating Your First Workflow

The easiest way to create a workflow is by manually creating a JSON file in `.opencode/workflow/`:

**Example:** `.opencode/workflow/my-workflow.json`
```json
{
  "name": "my-workflow",
  "version": "1.0.0",
  "description": "My custom workflow",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "First Step",
      "agent": "plan",
      "prompt": "Create implementation plan",
      "model": "base"
    }
  ]
}
```

### Running Workflows

```bash
# List available workflows
adw workflow list

# Get workflow details
adw workflow help my-workflow

# Execute workflow
adw workflow my-workflow <issue-number>

# Resume from specific step
adw workflow my-workflow <issue-number> --adw-id <id> --resume
adw workflow my-workflow <issue-number> --adw-id <id> --resume "Step Name"
```

## Core Concepts

### Workflow Definition Structure

A workflow definition consists of:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique workflow identifier (alphanumeric, hyphens, underscores) |
| `version` | string | No | Semantic version (e.g., "1.0.0") |
| `description` | string | No | Short summary of workflow purpose |
| `description_long` | string | No | Detailed description for status updates |
| `workflow_type` | string | No | Workflow type: `complete`, `patch`, `document`, `generate`, `custom` |
| `steps` | array | Yes | Ordered list of steps to execute |

**Example:**
```json
{
  "name": "simple-workflow",
  "version": "1.0.0",
  "description": "A simple example workflow",
  "workflow_type": "custom",
  "steps": [...]
}
```

### Step Types

#### Agent Steps

Execute an OpenCode agent or invoke an agent directly:

```json
{
  "type": "agent",
  "name": "Implement Feature",
  "agent": "implementor",
  "prompt": "Implement from specification",
  "model": "base",
  "timeout": 600,
  "retry": {
    "max_retries": 3,
    "initial_delay": 5.0,
    "backoff": 2.0
  }
}
```

**Agent Step Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"agent"` |
| `name` | string | Yes | Step name for display |
| `agent` | string | No* | Agent name (e.g., `"tester"`, `"plan"`) |
| `command` | string | No* | Slash command (e.g., `"/implement"`) |
| `prompt` | string | Yes | Instructions for agent execution |
| `model` | string | No | Model tier: `"light"`, `"base"`, `"heavy"` (default: `"base"`) |
| `timeout` | integer | No | Timeout in seconds (default: 600) |
| `retry` | object | No | Retry configuration with backoff |
| `continue_on_failure` | boolean | No | Continue workflow if step fails (default: false) |
| `condition` | object | No | Conditional execution rules |
| `description` | string | No | Detailed step description |

*Must provide either `agent` or `command`, but not both.

#### Workflow Steps

Compose workflows by referencing other workflow definitions:

```json
{
  "type": "workflow",
  "name": "Run Tests",
  "workflow_name": "test"
}
```

**Workflow Step Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"workflow"` |
| `name` | string | Yes | Step name for display |
| `workflow_name` | string | Yes | Name of workflow to inline |

### Conditional Execution

Control step execution based on workflow state using conditional expressions.

**Skip steps conditionally:**
```json
{
  "type": "agent",
  "name": "Run Tests",
  "agent": "tester",
  "prompt": "Run test suite",
  "condition": {
    "skip_if": "state.workflow_type == 'patch'"
  }
}
```

**Execute steps conditionally:**
```json
{
  "type": "agent",
  "name": "Generate Docs",
  "agent": "documenter",
  "prompt": "Generate documentation",
  "condition": {
    "if_condition": "state.workflow_type == 'complete'"
  }
}
```

See [Workflow Conditionals](workflow-conditionals.md) for complete syntax reference.

### Retry and Timeout Configuration

Configure retry behavior with exponential backoff:

```json
{
  "timeout": 900,
  "retry": {
    "max_retries": 3,
    "initial_delay": 5.0,
    "backoff": 2.0
  }
}
```

**Retry Configuration Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | integer | 3 | Maximum retry attempts |
| `initial_delay` | float | 5.0 | Initial delay in seconds |
| `backoff` | float | 2.0 | Exponential backoff multiplier |

**Retry behavior:**
- Attempt 1 fails → wait 5 seconds → retry
- Attempt 2 fails → wait 10 seconds (5 × 2) → retry
- Attempt 3 fails → wait 20 seconds (10 × 2) → retry
- Max retries reached → workflow fails (unless `continue_on_failure: true`)

### Model Tiers

Choose the appropriate model tier based on task complexity:

| Tier | Use Case | Cost | Example Models |
|------|----------|------|----------------|
| `light` | Simple tasks (linting, testing, commits) | Low | Haiku, GPT-3.5 |
| `base` | Standard implementation work | Medium | Sonnet, GPT-4 |
| `heavy` | Complex tasks (architecture, debugging) | High | Opus, GPT-4 Turbo |

**Example usage:**
```json
{
  "type": "agent",
  "name": "Quick Lint",
  "agent": "plan",
  "prompt": "Run linters",
  "model": "light"
}
```

## CLI Reference

### List Workflows

Display all available workflows:

```bash
adw workflow list
```

**Output:**
```
Available workflows:
- complete: Full validation workflow
- patch: Quick patch workflow
- test: Test workflow
- my-workflow: My custom workflow
```

### Get Workflow Help

Display detailed information about a specific workflow:

```bash
adw workflow help <name>
```

**Example:**
```bash
adw workflow help test

Workflow: test (v2.0.0)
Description: Test workflow - Run comprehensive tests
Type: complete

Steps:
  1. Run Tests (agent: tester, model: base)
```

### Execute Workflow

Run a workflow for a given issue:

```bash
adw workflow <name> <issue-number>
```

**Examples:**
```bash
# Run complete workflow
adw workflow complete 123

# Run custom workflow
adw workflow my-workflow 456
```

### Resume Workflow

Resume a workflow from the last failed step or a specific step:

```bash
# Resume from last checkpoint
adw workflow <name> <issue-number> --adw-id <id> --resume

# Resume from specific step
adw workflow <name> <issue-number> --adw-id <id> --resume "Step Name"
```

**Example:**
```bash
# Resume test workflow from last checkpoint
adw workflow test 123 --adw-id abc12345 --resume

# Resume from "Run Tests" step
adw workflow test 123 --adw-id abc12345 --resume "Run Tests"
```

## Advanced Topics

### Composition and Nesting

Workflows can reference other workflows to enable composition and reuse:

```json
{
  "name": "full-pipeline",
  "description": "Complete CI/CD pipeline",
  "steps": [
    {
      "type": "workflow",
      "name": "Lint",
      "workflow_name": "lint"
    },
    {
      "type": "workflow",
      "name": "Test",
      "workflow_name": "test"
    },
    {
      "type": "workflow",
      "name": "Deploy",
      "workflow_name": "deploy"
    }
  ]
}
```

**Benefits:**
- Reuse existing workflows
- Build complex pipelines from simple components
- Maintain separation of concerns
- Easy to test individual components

### Circular Dependency Detection

The workflow engine automatically detects and prevents circular references:

```json
// workflow-a.json
{
  "name": "workflow-a",
  "steps": [
    {"type": "workflow", "name": "B", "workflow_name": "workflow-b"}
  ]
}

// workflow-b.json
{
  "name": "workflow-b",
  "steps": [
    {"type": "workflow", "name": "A", "workflow_name": "workflow-a"}
  ]
}
```

**Error:**
```
Circular dependency detected: workflow-a -> workflow-b -> workflow-a
```

The engine uses depth-first search (DFS) validation to detect cycles before execution begins.

### Max Depth Limits

Workflow nesting is limited to 10 levels to prevent infinite recursion:

```
workflow-1
  → workflow-2
    → workflow-3
      → ... (up to 10 levels)
```

**Error when exceeded:**
```
Maximum workflow nesting depth (10) exceeded
```

### Continue on Failure

Allow non-critical steps to fail without stopping the workflow:

```json
{
  "type": "agent",
  "name": "Optional Security Scan",
  "agent": "plan",
  "prompt": "Run security checks",
  "continue_on_failure": true
}
```

**Behavior:**
- Step fails → logs warning → continues to next step
- Step succeeds → continues normally
- Useful for optional validation steps

## Troubleshooting

### Common Errors

**"Workflow not found"**
- **Cause:** Workflow file doesn't exist or name mismatch
- **Solution:** Verify file exists at `.opencode/workflow/<name>.json` and `name` field matches filename

**"Invalid step type"**
- **Cause:** Step type is not "agent" or "workflow"
- **Solution:** Use `"type": "agent"` or `"type": "workflow"`

**"Must provide either 'command' or 'agent' field"**
- **Cause:** Agent step missing both `command` and `agent` fields
- **Solution:** Add either `"agent": "tester"` or `"command": "/implement"`

**"Cannot provide both 'command' and 'agent' fields"**
- **Cause:** Agent step has both `command` and `agent` specified
- **Solution:** Remove one field (prefer `agent` for direct invocation, `command` for slash commands)

**"Circular dependency detected"**
- **Cause:** Workflow references create a cycle
- **Solution:** Restructure workflows to break the cycle

**"Maximum workflow nesting depth exceeded"**
- **Cause:** Workflows nested deeper than 10 levels
- **Solution:** Flatten workflow structure or split into separate workflows

**"Condition expression cannot be empty"**
- **Cause:** Condition object has empty `if_condition` or `skip_if`
- **Solution:** Provide valid condition expression or remove condition object

**"Invalid condition syntax"**
- **Cause:** Condition uses unsupported operator or syntax
- **Solution:** Use supported operators (`==`, `!=`, `in`, `not in`) with `state.*` fields

### Validation Errors

Run workflow validation manually:

```bash
python -c "from adw.workflows.engine.parser import load_workflow; load_workflow('.opencode/workflow/my-workflow.json')"
```

**Common validation issues:**
- Missing required fields (`name`, `steps`)
- Invalid field types (e.g., `timeout` must be integer)
- Empty arrays (steps must have at least one item)
- Pattern violations (name must match `^[a-zA-Z0-9_-]+$`)

### Debugging Workflow Execution

Enable detailed logging:

```bash
export ADW_DEBUG=true
adw workflow my-workflow 123
```

**Check workflow state:**
```bash
cat agents/<adw-id>/adw_state.json
```

**Review agent output:**
```bash
cat agents/<adw-id>/<agent-name>/raw_output.jsonl
```

## Best Practices

### Workflow Design

1. **Single Responsibility** - Each workflow should have one clear purpose
2. **Composition Over Duplication** - Reference existing workflows instead of duplicating steps
3. **Descriptive Names** - Use clear, descriptive step names for better status updates
4. **Appropriate Model Tiers** - Use `light` for simple tasks, `heavy` for complex reasoning
5. **Conservative Timeouts** - Set realistic timeouts based on expected execution time

### Conditional Logic

1. **Positive Conditions First** - Prefer `if_condition` over `skip_if` for clarity
2. **Simple Conditions** - Keep conditions simple (single comparison per step)
3. **Document Rationale** - Add comments explaining why conditions are needed

### Error Handling

1. **Fail Fast** - Let critical steps fail the workflow (default behavior)
2. **Graceful Degradation** - Use `continue_on_failure` for optional steps only
3. **Retry Configuration** - Set retries based on failure likelihood (network calls need more retries)

### Version Management

1. **Semantic Versioning** - Use semver for workflow versions (MAJOR.MINOR.PATCH)
2. **Breaking Changes** - Increment major version when changing workflow behavior significantly
3. **Backward Compatibility** - Maintain old versions when possible to avoid breaking existing users

## See Also

- **[Workflow JSON Schema](workflow-json-schema.md)** - Complete schema reference
- **[Workflow Conditionals](workflow-conditionals.md)** - Conditional syntax guide
- **[Workflow Migration Guide](workflow-migration-guide.md)** - Migrate Python workflows to JSON
- **[Workflow Examples](workflow-examples.md)** - Usage patterns and examples
- **[Architecture Reference](architecture_reference.md)** - Workflow engine architecture
