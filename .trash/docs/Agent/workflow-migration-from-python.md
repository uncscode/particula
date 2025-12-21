# Migrating from Python Workflows to JSON Workflow Engine

**Version:** 2.2.0  
**Last Updated:** 2025-11-29

## Overview

Starting in ADW v2.2.0, workflows are defined in JSON files and executed through the Dynamic Workflow Engine. The old Python workflow functions are deprecated and will be removed in v3.0.0.

This guide explains why the change was made, how to migrate existing code, and how to work with the new JSON-based workflow system.

## Why the Change?

The migration from Python workflows to JSON workflows provides several key benefits:

### 1. **Declarative and Easier to Understand**
- JSON workflows are declarative configurations, not imperative code
- Workflow structure is immediately visible without reading implementation details
- Non-programmers can understand and modify workflows

### 2. **Single Engine for Orchestration**
- One unified execution engine handles all workflows consistently
- Consistent retry logic, error handling, and status updates across all workflows
- Easier to debug and maintain than 12+ separate Python implementations

### 3. **User-Customizable Without Python Knowledge**
- Users can create custom workflows by copying and modifying JSON files
- No need to understand Python, imports, or ADW internals
- Workflows are data, not code

### 4. **Dynamic CLI Registration**
- New workflows are automatically discovered and registered
- CLI commands are auto-generated from workflow definitions
- No manual CLI configuration needed

### 5. **Significant Code Reduction**
- Reduced from ~2500 LOC (Python) to ~500 LOC (JSON)
- 80% less code to maintain and test
- Engine handles complexity, workflows stay simple

## Migration Timeline

| Version | Status | Action Required |
|---------|--------|-----------------|
| **v2.2.0** (Current) | Deprecation warnings added | None - warnings appear on direct import |
| **v2.3.0** | Final reminder warnings | Review code for direct imports |
| **v3.0.0** | Old workflows removed | Must migrate to JSON workflows or engine API |

## For CLI Users

**Good news: No migration needed!**

If you're using ADW via the CLI, you're already using the new engine:

```bash
# These commands use the JSON workflow engine automatically
uv run adw complete 123
uv run adw patch 456
uv run adw plan 789
```

The CLI automatically loads JSON workflow definitions from `.opencode/workflow/` and executes them through the engine. You won't see any deprecation warnings.

## For Python API Users

If your code directly imports and calls workflow functions, you need to migrate.

### Before (Deprecated)

```python
from adw.workflows.complete import run_complete_workflow
from adw.core import WorkflowContext

ctx = WorkflowContext(
    adw_id="abc12345",
    issue_number=123,
    triggered_by="api",
)

result = run_complete_workflow(ctx)
```

**This will emit a deprecation warning:**
```
DeprecationWarning: run_complete_workflow() is deprecated and will be removed in v3.0.0. 
Workflows are now defined in JSON and executed via the workflow engine. 
See docs/Agent/workflow-migration-from-python.md for migration guide.
```

### After (Recommended)

```python
from adw.workflows.engine.executor import execute_workflow
from adw.workflows.engine.registry import WorkflowRegistry
from adw.core import WorkflowContext

ctx = WorkflowContext(
    adw_id="abc12345",
    issue_number=123,
    triggered_by="api",
)

# Load workflow registry (discovers all JSON workflows)
registry = WorkflowRegistry()

# Execute workflow by name
result = execute_workflow("complete", ctx, registry)
```

### What Changed

1. **Workflow Loading**: Workflows are now loaded from JSON files in `.opencode/workflow/`
2. **Execution**: The `execute_workflow()` function handles execution
3. **Registry**: `WorkflowRegistry` discovers and validates workflows
4. **Context**: `WorkflowContext` remains the same

## Available Workflows

All built-in workflows have been migrated to JSON format:

| Workflow Name | JSON File | Description |
|---------------|-----------|-------------|
| `complete` | `.opencode/workflow/complete.json` | Full validation workflow |
| `patch` | `.opencode/workflow/patch.json` | Quick patch workflow |
| `plan` | `.opencode/workflow/plan.json` | Planning workflow |
| `build` | `.opencode/workflow/build.json` | Build/implementation workflow |
| `test` | `.opencode/workflow/test.json` | Testing workflow |
| `review` | `.opencode/workflow/review.json` | Code review workflow |
| `document` | `.opencode/workflow/document.json` | Documentation workflow |
| `ship` | `.opencode/workflow/ship.json` | Shipping/PR workflow |
| `plan_document` | `.opencode/workflow/plan_document.json` | Plan + document workflow |
| `architecture` | `.opencode/workflow/architecture.json` | Architecture review workflow |
| `docstring` | `.opencode/workflow/docstring.json` | Docstring update workflow |
| `finalize_docs` | `.opencode/workflow/finalize_docs.json` | ADR documentation workflow |

## Creating Custom Workflows

Custom workflows are now easier to create using JSON definitions.

### Example: Custom Review Workflow

Create `.opencode/workflow/my_custom_review.json`:

```json
{
  "name": "my_custom_review",
  "description": "Custom review workflow with extra linting",
  "description_long": "Reviews code with additional linting steps before standard review",
  "phases": [
    {
      "name": "Extra Linting",
      "steps": [
        {
          "name": "Run custom linters",
          "type": "command",
          "command": "/lint_extended"
        }
      ]
    },
    {
      "name": "Standard Review",
      "steps": [
        {
          "name": "Review code",
          "type": "command",
          "command": "/review"
        }
      ]
    }
  ],
  "error_handling": {
    "on_step_failure": "continue",
    "max_retries": 2
  }
}
```

### Using Your Custom Workflow

```bash
# CLI automatically discovers new workflows
uv run adw my_custom_review 123
```

```python
# Or via Python API
from adw.workflows.engine.executor import execute_workflow
from adw.workflows.engine.registry import WorkflowRegistry

registry = WorkflowRegistry()
result = execute_workflow("my_custom_review", ctx, registry)
```

See **[JSON Workflow Guide](workflow-engine.md)** for complete workflow syntax and features.

## Understanding the Engine

The Dynamic Workflow Engine provides:

### 1. **Automatic Discovery**
- Scans `.opencode/workflow/` for JSON files
- Validates workflow definitions against schema
- Registers workflows automatically

### 2. **Execution Management**
- Executes workflow steps in order
- Handles step failures with retry logic
- Updates GitHub status comments
- Tracks workflow state

### 3. **Conditional Logic**
- Skip steps based on conditions
- Conditional phase execution
- Dynamic workflow composition

### 4. **Error Handling**
- Configurable retry behavior
- Graceful failure handling
- Detailed error reporting

### 5. **Status Integration**
- Real-time GitHub status updates
- Phase progress tracking
- Detailed execution logs

## Advanced Migration Scenarios

### Scenario 1: Custom Workflow Modifications

**Problem**: You've modified a built-in workflow's Python implementation.

**Solution**: Create a custom JSON workflow with your modifications:

1. Copy the corresponding JSON workflow from `.opencode/workflow/`
2. Rename it (e.g., `custom_complete.json`)
3. Modify steps, phases, or error handling as needed
4. Use via CLI or API with new name

### Scenario 2: Programmatic Workflow Execution

**Problem**: You have scripts that execute workflows programmatically.

**Solution**: Update scripts to use the workflow engine:

```python
# Old (deprecated)
from adw.workflows.complete import run_complete_workflow

# New (recommended)
from adw.workflows.engine.executor import execute_workflow
from adw.workflows.engine.registry import WorkflowRegistry

registry = WorkflowRegistry()
result = execute_workflow("complete", ctx, registry)
```

### Scenario 3: Testing Workflow Behavior

**Problem**: Tests import workflow functions directly.

**Solution**: Update tests to use the engine:

```python
# Old test
def test_complete_workflow():
    from adw.workflows.complete import run_complete_workflow
    result = run_complete_workflow(ctx)
    assert result.success

# New test
def test_complete_workflow():
    from adw.workflows.engine.executor import execute_workflow
    from adw.workflows.engine.registry import WorkflowRegistry
    
    registry = WorkflowRegistry()
    result = execute_workflow("complete", ctx, registry)
    assert result.success
```

## Troubleshooting

### Issue: Deprecation Warning Appears

**Symptom:**
```
DeprecationWarning: run_complete_workflow() is deprecated...
```

**Cause**: Code is directly importing workflow functions.

**Solution**: Update imports to use the workflow engine (see "For Python API Users" above).

### Issue: Deprecation Warnings and Orchestrator Workflows

**Symptom**: When calling an orchestrator workflow like `run_complete_workflow()`, 
`run_patch_workflow()`, or `run_plan_document_workflow()`, you only see one deprecation warning
(not multiple).

**Explanation**: Orchestrator workflows call other workflow functions internally (e.g., 
`complete` calls `plan`, `build`, `test`, etc.). To avoid confusing users with a cascade of 
deprecation warnings, the orchestrator workflows suppress warnings from their sub-workflows.
This means you'll see a single deprecation warning for the orchestrator workflow itself, 
directing you to migrate to the JSON workflow engine.

**Note**: If you call individual workflow functions directly (e.g., `run_plan_workflow()`), 
you will still see deprecation warnings for each function you call.

### Issue: Workflow Not Found

**Symptom:**
```
WorkflowNotFoundError: Workflow 'my_workflow' not found
```

**Cause**: JSON file doesn't exist or is invalid.

**Solution**:
1. Check `.opencode/workflow/my_workflow.json` exists
2. Validate JSON syntax
3. Run `WorkflowRegistry().validate_workflow("my_workflow")`

### Issue: Step Execution Fails

**Symptom**: Workflow fails at specific step.

**Solution**:
1. Check step configuration in JSON
2. Verify command exists and is executable
3. Review error logs in `agents/{adw_id}/logs/`
4. Check retry configuration if needed

## Getting Help

### Documentation Resources

- **[Workflow Engine Guide](workflow-engine.md)** - Complete engine documentation
- **[JSON Workflow Schema](.opencode/workflow/schema.json)** - Workflow definition schema
- **[Examples](../../Examples/)** - Example workflows and use cases

### Support Channels

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions or share workflows
- **Documentation**: Browse guides and API reference

## FAQ

### Q: Do I need to migrate if I only use the CLI?

**A:** No. CLI users automatically use the new engine. No changes needed.

### Q: When will the old Python workflows be removed?

**A:** In v3.0.0, planned for Q2 2026.

### Q: Can I still modify built-in workflows?

**A:** Yes! Create a custom JSON workflow based on the built-in one.

### Q: Are there performance differences?

**A:** No significant difference. The engine overhead is minimal (~10ms per workflow).

### Q: Can I mix old and new approaches?

**A:** Yes, during the deprecation period (v2.2.0 - v2.3.0), but you'll see warnings.

### Q: What if I find a bug in the engine?

**A:** Report it on GitHub Issues. Critical bugs will be fixed in patch releases.

## Summary

The migration from Python to JSON workflows:

✅ **Makes workflows easier to understand and modify**  
✅ **Reduces codebase complexity by 80%**  
✅ **Enables user customization without Python knowledge**  
✅ **Provides consistent execution and error handling**  
✅ **Automatically discovers and registers new workflows**

**Action Required:**
- **CLI users**: No action needed
- **Python API users**: Update imports to use `execute_workflow()`
- **Custom workflow authors**: Migrate to JSON format

**Timeline:**
- **v2.2.0** (Current): Deprecation warnings
- **v2.3.0**: Final reminders
- **v3.0.0**: Old workflows removed

For questions or assistance, consult the documentation or open a GitHub issue.
