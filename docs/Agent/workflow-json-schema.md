# Workflow JSON Schema Reference

**Version:** 2.2.0  
**Last Updated:** 2025-11-29

## Overview

This document provides a complete reference for the ADW workflow JSON schema. All workflow definitions must conform to this schema for validation and execution.

**Source of Truth:** `adw/workflows/engine/models.py`

## Workflow Definition

The top-level workflow definition object.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | - | Unique workflow identifier. Must match pattern: `^[a-zA-Z0-9_-]+$` |
| `version` | string | No | - | Semantic version (e.g., "1.0.0", "2.1.0") |
| `description` | string | No | `""` | Short summary of workflow purpose |
| `description_long` | string | No | `null` | Detailed description for status updates |
| `workflow_type` | string | No | `null` | Workflow type: `complete`, `patch`, `document`, `generate`, `custom` |
| `steps` | array | Yes | - | Ordered list of steps to execute. Must contain at least one step. |

### Validation Rules

- **name:** Required, non-empty, alphanumeric with hyphens/underscores only
- **version:** Optional, recommended to use semantic versioning
- **steps:** Required, must be non-empty array

### Example

```json
{
  "name": "my-workflow",
  "version": "1.0.0",
  "description": "Custom workflow for feature implementation",
  "description_long": "This workflow implements features with full validation including tests and documentation",
  "workflow_type": "custom",
  "steps": [...]
}
```

## Step Types

Workflows contain a list of steps. Each step must be one of two types:
- **Agent Step:** Executes an OpenCode agent
- **Workflow Step:** References another workflow

### Step Type Field

All steps must include a `type` field:

```json
{
  "type": "agent"  // or "workflow"
}
```

## Agent Step

Executes an OpenCode agent with a specific command or direct invocation.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | Yes | - | Must be `"agent"` |
| `name` | string | Yes | - | Human-readable step name. Min length: 1 |
| `agent` | string | No* | `null` | Agent name (e.g., `"tester"`, `"plan"`). Pattern: `^[a-zA-Z_][a-zA-Z0-9_-]*$` |
| `command` | string | No* | `null` | Slash command (e.g., `"/implement"`). Pattern: `^/[a-zA-Z_][a-zA-Z0-9_]*$` |
| `prompt` | string | Yes | - | Prompt text for agent execution. Min length: 1 |
| `model` | string | No | `"base"` | Model tier: `"light"`, `"base"`, `"heavy"` |
| `timeout` | integer | No | `600` | Timeout in seconds. Minimum: 1 |
| `retry` | object | No | See RetryConfig | Retry configuration with exponential backoff |
| `continue_on_failure` | boolean | No | `false` | Continue workflow if this step fails |
| `condition` | object | No | `null` | Conditional execution rules |
| `description` | string | No | `null` | Detailed step description for status updates |

*Must provide either `agent` or `command`, but not both.

### Validation Rules

- **name:** Required, non-empty string
- **agent OR command:** Exactly one must be provided
- **prompt:** Required, non-empty string
- **model:** Must be one of: `"light"`, `"base"`, `"heavy"`
- **timeout:** Must be positive integer (≥ 1)

### Example

```json
{
  "type": "agent",
  "name": "Run Tests",
  "agent": "tester",
  "prompt": "Execute comprehensive test suite",
  "model": "base",
  "timeout": 900,
  "retry": {
    "max_retries": 3,
    "initial_delay": 5.0,
    "backoff": 2.0
  },
  "continue_on_failure": false,
  "condition": {
    "skip_if": "state.workflow_type == 'patch'"
  },
  "description": "Runs pytest with coverage reporting"
}
```

## Workflow Step

References another workflow definition for composition and reuse.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `type` | string | Yes | - | Must be `"workflow"` |
| `name` | string | Yes | - | Human-readable step name. Min length: 1 |
| `workflow_name` | string | Yes | - | Name of workflow to inline. Pattern: `^[a-zA-Z0-9_-]+$`, min length: 1 |

### Validation Rules

- **workflow_name:** Must match an existing workflow definition
- **No circular references:** Engine validates against circular dependencies

### Example

```json
{
  "type": "workflow",
  "name": "Run Tests",
  "workflow_name": "test"
}
```

## Retry Configuration

Controls retry behavior with exponential backoff.

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `max_retries` | integer | No | `3` | Maximum number of retry attempts |
| `initial_delay` | float | No | `5.0` | Initial delay between retries in seconds |
| `backoff` | float | No | `2.0` | Exponential backoff multiplier |

### Retry Behavior

Retry delays increase exponentially:
1. First retry: wait `initial_delay` seconds
2. Second retry: wait `initial_delay * backoff` seconds
3. Third retry: wait `initial_delay * backoff^2` seconds
4. And so on...

### Example

```json
{
  "retry": {
    "max_retries": 3,
    "initial_delay": 5.0,
    "backoff": 2.0
  }
}
```

**Retry sequence:**
- Attempt 1 fails → wait 5s → retry
- Attempt 2 fails → wait 10s → retry
- Attempt 3 fails → wait 20s → retry
- Max retries reached → fail (or continue if `continue_on_failure: true`)

## Condition Object

Controls conditional execution of workflow steps.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `if_condition` | string | No* | Expression that must be True for step to execute. If False, step is skipped. |
| `skip_if` | string | No* | Expression that causes step to be skipped if True. If False, step executes. |

*Provide either `if_condition` or `skip_if`, but not both.

### Supported Operators

- **Equality:** `==`, `!=`
- **Containment:** `in`, `not in`

### Available State Fields

| Field | Type | Description |
|-------|------|-------------|
| `state.workflow_type` | string | Workflow type: `complete`, `patch`, `document`, `generate`, `custom` |
| `state.issue_class` | string | Issue class: `feature`, `bug`, `chore` |
| `state.completed_steps` | array | List of completed step names |
| `state.adw_id` | string | Current ADW workflow ID |

### Security

- **Safe evaluation:** No `exec()` or `eval()` used
- **Whitelist-only:** Only `state.*` fields accessible
- **No function calls:** Only comparison operators allowed

See [Workflow Conditionals](workflow-conditionals.md) for complete syntax guide.

### Examples

**Execute step only for complete workflows:**
```json
{
  "condition": {
    "if_condition": "state.workflow_type == 'complete'"
  }
}
```

**Skip step for patch workflows:**
```json
{
  "condition": {
    "skip_if": "state.workflow_type == 'patch'"
  }
}
```

**Execute only for feature issues:**
```json
{
  "condition": {
    "if_condition": "state.issue_class == 'feature'"
  }
}
```

**Execute if step was completed:**
```json
{
  "condition": {
    "if_condition": "'Plan' in state.completed_steps"
  }
}
```

## Complete Example

A comprehensive workflow demonstrating all features:

```json
{
  "name": "custom-pipeline",
  "version": "1.0.0",
  "description": "Custom CI/CD pipeline with conditional steps",
  "description_long": "Runs linting, testing, and optional documentation based on workflow type",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "Lint Code",
      "agent": "plan",
      "prompt": "Run all configured linters",
      "model": "light",
      "timeout": 300,
      "retry": {
        "max_retries": 2,
        "initial_delay": 3.0,
        "backoff": 1.5
      },
      "description": "Runs ruff and mypy"
    },
    {
      "type": "workflow",
      "name": "Run Tests",
      "workflow_name": "test"
    },
    {
      "type": "agent",
      "name": "Generate Documentation",
      "agent": "documenter",
      "prompt": "Generate documentation from code",
      "model": "base",
      "condition": {
        "if_condition": "state.workflow_type == 'complete'"
      },
      "continue_on_failure": true,
      "description": "Generates API docs and user guides"
    },
    {
      "type": "agent",
      "name": "Commit Changes",
      "agent": "git-commit",
      "prompt": "Create semantic commit",
      "model": "light",
      "timeout": 120
    }
  ]
}
```

## Schema Validation

### Validating Workflows

Validate workflow JSON against schema:

```python
from adw.workflows.engine.parser import load_workflow

# Load and validate workflow
workflow = load_workflow('.opencode/workflow/my-workflow.json')
```

### Common Validation Errors

**Missing required field:**
```
ValidationError: Field 'name' is required
```

**Invalid field type:**
```
ValidationError: Field 'timeout' must be integer, got string
```

**Pattern violation:**
```
ValidationError: Field 'name' must match pattern ^[a-zA-Z0-9_-]+$
```

**Empty array:**
```
ValidationError: Field 'steps' must contain at least one item
```

**Agent/Command conflict:**
```
ValidationError: Cannot provide both 'command' and 'agent' fields
ValidationError: Must provide either 'command' or 'agent' field
```

## JSON Schema File

The complete JSON schema is located at:
```
.opencode/workflow/schema.json
```

This schema is used for:
- Pre-execution validation
- IDE auto-completion
- Documentation generation
- Error detection

## See Also

- **[Workflow Engine](workflow-engine.md)** - Main workflow engine guide
- **[Workflow Conditionals](workflow-conditionals.md)** - Conditional syntax reference
- **[Workflow Examples](workflow-examples.md)** - Usage patterns and examples
- **[Workflow Migration Guide](workflow-migration-guide.md)** - Migrate Python workflows
