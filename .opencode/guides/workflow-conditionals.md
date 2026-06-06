# Workflow Conditional Syntax

**Version:** 2.2.0  
**Last Updated:** 2025-11-29

## Overview

Conditionals allow workflow steps to execute or skip based on workflow state. This enables flexible, adaptive workflows that respond to different contexts without code duplication.

**Key Features:**
- **Simple Syntax:** Basic comparison operators only
- **State-Based:** Conditions evaluate against workflow state
- **Secure:** No arbitrary code execution (no exec/eval)
- **Validated:** Clear error messages for invalid syntax

## Conditional Types

### Execute If True (`if_condition`)

Step executes **only** if condition evaluates to `True`. If `False`, step is skipped.

```json
{
  "condition": {
    "if_condition": "state.workflow_type == 'complete'"
  }
}
```

**Behavior:**
- Condition `True` → Step executes
- Condition `False` → Step skipped

### Skip If True (`skip_if`)

Step is **skipped** if condition evaluates to `True`. If `False`, step executes normally.

```json
{
  "condition": {
    "skip_if": "state.workflow_type == 'patch'"
  }
}
```

**Behavior:**
- Condition `True` → Step skipped
- Condition `False` → Step executes

### Fail If True (`fail_if`)

Step **fails with error** if condition evaluates to `True`. Used for precondition validation to prevent cascading silent failures.

```json
{
  "condition": {
    "fail_if": "state.spec_content == null",
    "fail_message": "Plan step did not generate spec_content"
  }
}
```

**Behavior:**
- Condition `True` → Step fails with error message
- Condition `False` → Step executes normally
- Error propagates to GitHub status comment with ❌ indicator
- Workflow aborts (doesn't continue to next steps)

**Use Cases:**
- Validate required state fields exist before dependent steps
- Prevent cascading silent failures in multi-step workflows
- Provide clear error messages when preconditions aren't met

### Choosing Between Condition Types

| Use Case | Recommended Type | Example |
|----------|------------------|---------|
| Step only runs in specific mode | `if_condition` | "Only run tests for complete workflow" |
| Step skips in specific mode | `skip_if` | "Skip documentation for patch workflow" |
| Multiple valid modes | `if_condition` | "Run if complete OR document workflow" |
| Single exclusion case | `skip_if` | "Skip if patch workflow" |
| Validate precondition | `fail_if` | "Fail if spec_content is null" |
| Prevent silent failures | `fail_if` | "Fail if previous step had error" |

**General Rule:** 
- Use `if_condition` when you want to be explicit about when a step runs
- Use `skip_if` when you want to exclude specific cases
- Use `fail_if` when a precondition must be met or the workflow should abort with a clear error

**Evaluation Order:** When multiple conditions are set on a step, they are evaluated in this order:
1. `fail_if` - If True, step fails with error (checked first)
2. `skip_if` - If True, step is skipped
3. `if_condition` - If False, step is skipped

## Supported Operators

The condition evaluator supports a whitelist of safe comparison operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equality | `state.workflow_type == 'complete'` |
| `!=` | Inequality | `state.workflow_type != 'patch'` |
| `in` | Membership (contains) | `'test' in state.completed_steps` |
| `not in` | Non-membership (does not contain) | `'Plan' not in state.completed_steps` |

### Operator Precedence

Operators are evaluated in the order they appear (left to right, no precedence). Only one operator per condition is allowed.

**Valid:**
```json
"if_condition": "state.workflow_type == 'complete'"
```

**Invalid (multiple operators):**
```json
"if_condition": "state.workflow_type == 'complete' and state.issue_class == 'feature'"
```

To combine conditions, use multiple conditional steps instead.

## Available State Fields

Conditions can access the following workflow state fields:

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `state.workflow_type` | string | Type of workflow being executed | `complete`, `patch`, `document`, `generate`, `custom` |
| `state.issue_class` | string | Classification of GitHub issue | `feature`, `bug`, `chore` |
| `state.completed_steps` | array | List of step names that have completed | `["Plan", "Implement", "Test"]` |
| `state.adw_id` | string | Unique workflow identifier | `"abc12345"` |
| `state.review_feedback` | string/null | Slice review findings persisted in state | `"needs fix in parser"`, `null` |
| `state.request_fix` | boolean | Whether the workflow should run a fix pass | `True`, `False` |
| `state.fix_completed` | boolean | Whether the requested fix pass already ran | `True`, `False` |

### State Field Details

**workflow_type:**
- Set by workflow definition or inferred from issue
- Commonly used to skip/include validation steps
- Example: Skip tests for patch workflows

**issue_class:**
- Determined by issue labels or content analysis
- Used to tailor workflow behavior to issue type
- Example: Generate docs only for feature issues

**completed_steps:**
- Dynamically updated as workflow progresses
- Step names match the `name` field in step definitions
- Used for dependent step execution
- Example: Only deploy if tests passed

**adw_id:**
- Unique 8-character workflow identifier
- Rarely used in conditions (mostly for debugging)
- Available for advanced use cases

**review_feedback / request_fix / fix_completed:**
- Slice-scoped review/fix coordination fields used by auto workflows
- `request_fix == True` can gate fix execution after review
- `fix_completed == True` prevents re-entering the same fix step
- `review_feedback` stores review output for later steps or operator inspection

## Supported Literal Types

### String Literals

Enclose strings in single or double quotes:

```json
"if_condition": "state.workflow_type == 'complete'"
"skip_if": "state.issue_class != \"feature\""
```

### Boolean Literals

Use `True` or `False` (capitalized):

```json
"if_condition": "state.enable_tests == True"
"skip_if": "state.skip_docs == False"
```

### Null Literal

Use `null` (lowercase) to check for null/None values:

```json
"fail_if": "state.spec_content == null"
"if_condition": "state.optional_field != null"
```

**Notes:**
- `null` is case-sensitive (only lowercase works, not `NULL` or `Null`)
- `null` is distinct from empty string (`""`)
- Useful for validating required state fields exist

### Numeric Literals

**Integers:**
```json
"if_condition": "state.retry_count == 3"
```

**Floats:**
```json
"if_condition": "state.coverage == 0.85"
```

## Common Patterns

### Skip Tests for Patch Workflows

Patch workflows are for quick fixes that don't need full validation:

```json
{
  "type": "agent",
  "name": "Run Tests",
  "agent": "tester",
  "prompt": "Run comprehensive test suite",
  "condition": {
    "skip_if": "state.workflow_type == 'patch'"
  }
}
```

### Execute Documentation Only for Features

Documentation is most valuable for new features:

```json
{
  "type": "agent",
  "name": "Generate Documentation",
  "agent": "documenter",
  "prompt": "Generate user-facing documentation",
  "condition": {
    "if_condition": "state.issue_class == 'feature'"
  }
}
```

### Conditional Linting Based on Workflow Type

Run full linting for complete workflows, quick linting for patches:

```json
{
  "type": "agent",
  "name": "Full Lint",
  "agent": "plan",
  "prompt": "Run all linters (ruff, mypy, pylint)",
  "condition": {
    "if_condition": "state.workflow_type == 'complete'"
  }
},
{
  "type": "agent",
  "name": "Quick Lint",
  "agent": "plan",
  "prompt": "Run fast linters only (ruff)",
  "condition": {
    "if_condition": "state.workflow_type == 'patch'"
  }
}
```

### Execute Step Only After Dependency

Ensure a step only runs if its dependency completed:

```json
{
  "type": "agent",
  "name": "Deploy",
  "agent": "deploy",
  "prompt": "Deploy to production",
  "condition": {
    "if_condition": "'Test' in state.completed_steps"
  }
}
```

### Skip Step If Already Completed

Useful for workflow resumption:

```json
{
  "type": "agent",
  "name": "Plan",
  "agent": "plan",
  "prompt": "Create implementation plan",
  "condition": {
    "skip_if": "'Plan' in state.completed_steps"
  }
}
```

### Validate Required State Before Execution

Ensure a step only runs if previous step produced required output:

```json
{
  "type": "agent",
  "name": "Execute",
  "agent": "implementor",
  "prompt": "Implement the plan",
  "condition": {
    "fail_if": "state.spec_content == null",
    "fail_message": "Plan step did not generate spec_content. The planning agent may have failed silently."
  }
}
```

### Prevent Execution After Previous Error

Fail early if a previous step encountered an error:

```json
{
  "type": "agent",
  "name": "Deploy",
  "agent": "deploy",
  "prompt": "Deploy to production",
  "condition": {
    "fail_if": "state.build_error != null",
    "fail_message": "Cannot deploy: build step had errors"
  }
}
```

## Security Design

The conditional evaluator is designed with security as a top priority:

### Safe Evaluation

- **NO exec() or eval():** Uses whitelist-based parsing
- **NO arbitrary code:** Only comparison operators allowed
- **NO function calls:** Cannot call functions or methods
- **NO imports:** Cannot import modules

### Restricted Access

- **State-only:** Can only access `state.*` fields
- **No globals:** Cannot access global variables
- **No builtins:** Cannot access built-in functions
- **Read-only:** Cannot modify state

### Validation

- **Pattern matching:** Uses regex to validate syntax
- **Whitelist operators:** Only allows `==`, `!=`, `in`, `not in`
- **Type checking:** Validates operand types before evaluation
- **Clear errors:** Provides descriptive error messages

**Source:** `adw/workflows/engine/conditions.py`

## Error Messages

### Common Errors

**Empty condition:**
```
ConditionEvaluationError: Condition expression cannot be empty
```

**Invalid operator:**
```
ConditionEvaluationError: Invalid condition syntax: 'state.x > 5'. 
Must use one of: ==, !=, in, not in
```

**State field not found:**
```
ConditionEvaluationError: State field 'invalid_field' not found
```

**Invalid operand:**
```
ConditionEvaluationError: Invalid operand: 'state.x + 1'. 
Must be a state field (state.field_name), string literal ('value'), 
boolean (True/False), float (0.5), or integer (123)
```

**Type mismatch:**
```
ConditionEvaluationError: 'in' operator requires right operand to be iterable, 
got int
```

## Examples

### Basic Equality

```json
{
  "condition": {
    "if_condition": "state.workflow_type == 'complete'"
  }
}
```

### Inequality

```json
{
  "condition": {
    "skip_if": "state.issue_class != 'feature'"
  }
}
```

### Membership (in)

```json
{
  "condition": {
    "if_condition": "'Test' in state.completed_steps"
  }
}
```

### Non-membership (not in)

```json
{
  "condition": {
    "skip_if": "'Plan' not in state.completed_steps"
  }
}
```

### Boolean Comparison

```json
{
  "condition": {
    "if_condition": "state.enable_tests == True"
  }
}
```

### Multiple Conditional Steps

Since only one operator per condition is allowed, use multiple steps for complex logic:

```json
{
  "steps": [
    {
      "type": "agent",
      "name": "Feature Docs",
      "agent": "documenter",
      "prompt": "Generate feature documentation",
      "condition": {
        "if_condition": "state.issue_class == 'feature'"
      }
    },
    {
      "type": "agent",
      "name": "Complete Workflow Docs",
      "agent": "documenter",
      "prompt": "Generate comprehensive docs",
      "condition": {
        "if_condition": "state.workflow_type == 'complete'"
      }
    }
  ]
}
```

## Testing Conditionals

### Manual Testing

Test conditional expressions in Python:

```python
from adw.workflows.engine.conditions import ConditionEvaluator

evaluator = ConditionEvaluator()
state = {
    "workflow_type": "patch",
    "issue_class": "bug",
    "completed_steps": ["Plan", "Implement"],
    "adw_id": "abc12345"
}

# Test equality
result = evaluator.evaluate("state.workflow_type == 'patch'", state)
assert result == True

# Test membership
result = evaluator.evaluate("'Plan' in state.completed_steps", state)
assert result == True

# Test inequality
result = evaluator.evaluate("state.issue_class != 'feature'", state)
assert result == True
```

### Debugging Conditions

Enable conditional logging:

```bash
export ADW_DEBUG=true
adw workflow my-workflow 123
```

Logs will show:
- Which conditions were evaluated
- Condition results (True/False)
- Steps that were skipped due to conditions

## Best Practices

### Keep Conditions Simple

**Good:**
```json
"if_condition": "state.workflow_type == 'complete'"
```

**Avoid:**
Use multiple steps instead of complex logic

### Use Descriptive State Fields

When extending state, use clear field names:
- ✅ `state.enable_security_checks`
- ❌ `state.sec`

### Document Condition Rationale

Add comments in workflow definitions:
```json
{
  "type": "agent",
  "name": "Security Scan",
  "condition": {
    "if_condition": "state.workflow_type == 'complete'"
  },
  "description": "Only run security scans for complete workflows to save time on patches"
}
```

### Test Edge Cases

Ensure conditions handle:
- Empty arrays (`state.completed_steps == []`)
- Null values (`state.optional_field == null`)
- Missing fields (will raise error, which is good)

### Prefer Positive Conditions

**Preferred:**
```json
"if_condition": "state.workflow_type == 'complete'"
```

**Less Clear:**
```json
"skip_if": "state.workflow_type != 'complete'"
```

Positive conditions are easier to understand and reason about.

## See Also

- **[Workflow Engine](workflow-engine.md)** - Main workflow engine guide
- **[Workflow JSON Schema](workflow-json-schema.md)** - Complete schema reference
- **[Workflow Examples](workflow-examples.md)** - Usage patterns and examples
- **`adw/workflows/engine/conditions.py`** - Conditional evaluator implementation
