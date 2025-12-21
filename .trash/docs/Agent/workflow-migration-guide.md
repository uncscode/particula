# Migrating Python Workflows to JSON

**Version:** 2.2.0  
**Last Updated:** 2025-11-29

## Overview

This guide walks through the process of converting Python-based workflows to JSON workflow definitions. It explains when migration is appropriate, provides a step-by-step process, and demonstrates common migration patterns.

**Related:** See [workflow-migration-from-python.md](workflow-migration-from-python.md) for deprecation timeline and API migration.

## Why Migrate?

### Benefits of JSON Workflows

- **No Code Changes Required** - Modify workflows without Python knowledge
- **Composability** - Build complex workflows from reusable components
- **Validation** - Schema catches errors before execution
- **Visibility** - Workflow structure is immediately apparent
- **User-Friendly** - Non-programmers can create and modify workflows
- **Auto-Registration** - New workflows automatically appear in CLI

### When to Keep Python

JSON workflows are suitable for most use cases, but Python may be preferable when:

- **Complex Conditional Logic** - Beyond simple if/skip_if expressions
- **Custom Error Handling** - Specialized error recovery logic
- **External Integrations** - Complex API interactions or data transformations
- **Dynamic Prompt Generation** - Prompts constructed from complex logic
- **State Manipulation** - Need to modify workflow state in custom ways

**Rule of Thumb:** If your workflow is primarily a sequence of agent invocations with simple conditions, migrate to JSON. If it has complex Python logic, consider keeping it as Python or extracting the complex logic into a helper module.

## Migration Process

### Step 1: Identify Workflow Steps

Analyze your Python workflow to identify discrete steps.

**Example Python Workflow:**
```python
# adw/workflows/complete.py
def run_complete_workflow(ctx: WorkflowContext) -> WorkflowResult:
    """Execute complete workflow with full validation."""
    
    # Step 1: Planning
    plan_result = run_plan(ctx)
    if not plan_result.success:
        return plan_result
    
    # Step 2: Implementation
    build_result = run_build(ctx)
    if not build_result.success:
        return build_result
    
    # Step 3: Testing
    test_result = run_test(ctx)
    if not test_result.success:
        return test_result
    
    # Step 4: Review
    review_result = run_review(ctx)
    if not review_result.success:
        return review_result
    
    # Step 5: Documentation
    doc_result = run_document(ctx)
    if not doc_result.success:
        return doc_result
    
    # Step 6: Shipping
    ship_result = run_ship(ctx)
    return ship_result
```

**Identified Steps:**
1. Planning (run_plan)
2. Implementation (run_build)
3. Testing (run_test)
4. Review (run_review)
5. Documentation (run_document)
6. Shipping (run_ship)

### Step 2: Map to JSON Steps

Convert each Python function call to a JSON step. Use `workflow` type to reference other workflows:

```json
{
  "steps": [
    {
      "type": "workflow",
      "name": "Planning",
      "workflow_name": "plan"
    },
    {
      "type": "workflow",
      "name": "Building",
      "workflow_name": "build"
    },
    {
      "type": "workflow",
      "name": "Testing",
      "workflow_name": "test"
    },
    {
      "type": "workflow",
      "name": "Reviewing",
      "workflow_name": "review"
    },
    {
      "type": "workflow",
      "name": "Documenting",
      "workflow_name": "document"
    },
    {
      "type": "workflow",
      "name": "Shipping",
      "workflow_name": "ship"
    }
  ]
}
```

### Step 3: Add Metadata

Add required workflow metadata fields:

```json
{
  "name": "complete",
  "version": "2.0.0",
  "description": "Complete workflow with full validation",
  "description_long": "Executes comprehensive workflow including planning, implementation, testing, review, documentation, and shipping phases with full validation at each step",
  "workflow_type": "complete",
  "steps": [...]
}
```

**Metadata Guidelines:**
- **name:** Must match filename (without .json extension)
- **version:** Use semantic versioning (MAJOR.MINOR.PATCH)
- **description:** Short summary (1 sentence)
- **description_long:** Detailed description for status comments
- **workflow_type:** complete, patch, document, generate, or custom

### Step 4: Test Equivalence

Verify the JSON workflow produces the same results as the Python workflow:

```bash
# Test Python workflow (if still available)
uv run adw complete 123

# Test JSON workflow
uv run adw workflow complete 123

# Compare results:
# - Same steps executed
# - Same order
# - Same success/failure behavior
# - Same output artifacts
```

**Validation Checklist:**
- ✅ All steps execute in correct order
- ✅ Conditional logic behaves identically
- ✅ Error handling is equivalent
- ✅ Timeout behavior matches
- ✅ Retry logic is consistent

## Common Migration Patterns

### Pattern 1: Sequential Steps

**Python:**
```python
def run_pipeline(ctx):
    lint(ctx)
    test(ctx)
    deploy(ctx)
```

**JSON:**
```json
{
  "steps": [
    {"type": "workflow", "name": "Lint", "workflow_name": "lint"},
    {"type": "workflow", "name": "Test", "workflow_name": "test"},
    {"type": "workflow", "name": "Deploy", "workflow_name": "deploy"}
  ]
}
```

### Pattern 2: Conditional Execution

**Python:**
```python
def run_build(ctx):
    implement(ctx)
    if ctx.workflow_type == "complete":
        update_docstrings(ctx)
        run_linters(ctx)
    commit(ctx)
```

**JSON:**
```json
{
  "steps": [
    {
      "type": "agent",
      "name": "Implement",
      "agent": "implementor",
      "prompt": "Implement from specification"
    },
    {
      "type": "agent",
      "name": "Update Docstrings",
      "agent": "docstring",
      "prompt": "Update docstrings",
      "condition": {
        "if_condition": "state.workflow_type == 'complete'"
      }
    },
    {
      "type": "agent",
      "name": "Lint",
      "agent": "plan",
      "prompt": "Run linters",
      "condition": {
        "if_condition": "state.workflow_type == 'complete'"
      }
    },
    {
      "type": "agent",
      "name": "Commit",
      "agent": "git-commit",
      "prompt": "Create commit"
    }
  ]
}
```

### Pattern 3: Error Handling

**Python with try/except:**
```python
def run_workflow(ctx):
    try:
        optional_step(ctx)
    except Exception as e:
        logger.warning(f"Optional step failed: {e}")
    
    critical_step(ctx)
```

**JSON with continue_on_failure:**
```json
{
  "steps": [
    {
      "type": "agent",
      "name": "Optional Step",
      "agent": "plan",
      "prompt": "Run optional validation",
      "continue_on_failure": true
    },
    {
      "type": "agent",
      "name": "Critical Step",
      "agent": "implementor",
      "prompt": "Execute critical implementation"
    }
  ]
}
```

### Pattern 4: Retry Logic

**Python with retry decorator:**
```python
@retry(max_attempts=3, backoff=2.0)
def flaky_step(ctx):
    # Implementation that may fail
    pass
```

**JSON with retry config:**
```json
{
  "type": "agent",
  "name": "Flaky Step",
  "agent": "plan",
  "prompt": "Execute potentially flaky operation",
  "retry": {
    "max_retries": 3,
    "initial_delay": 5.0,
    "backoff": 2.0
  }
}
```

## Complete Example: build.py → build.json

### Before (Python)

```python
# adw/workflows/build.py
from adw.core import WorkflowContext, WorkflowResult
from adw.core.agent import execute_agent

def run_build(ctx: WorkflowContext) -> WorkflowResult:
    """Execute build workflow with conditional steps."""
    
    # Always run implementation
    impl_result = execute_agent(
        agent="implementor",
        prompt="Implement from specification",
        ctx=ctx,
        model_tier="base"
    )
    
    if not impl_result.success:
        return WorkflowResult(success=False, message="Implementation failed")
    
    # Conditional: update docstrings only for complete workflows
    if ctx.workflow_type == "complete":
        doc_result = execute_agent(
            agent="docstring",
            prompt="Update docstrings",
            ctx=ctx,
            model_tier="base"
        )
        
        if not doc_result.success:
            return WorkflowResult(success=False, message="Docstring update failed")
        
        # Run linters
        lint_result = execute_agent(
            agent="plan",
            prompt="Run all configured linters",
            ctx=ctx,
            model_tier="light"
        )
        
        if not lint_result.success:
            return WorkflowResult(success=False, message="Linting failed")
    
    # Always commit
    commit_result = execute_agent(
        agent="git-commit",
        prompt="Create semantic commit",
        ctx=ctx,
        model_tier="light"
    )
    
    if not commit_result.success:
        return WorkflowResult(success=False, message="Commit failed")
    
    return WorkflowResult(success=True, message="Build completed successfully")
```

### After (JSON)

```json
{
  "name": "build",
  "version": "2.0.0",
  "description": "Build workflow - Implementation with conditional validation",
  "description_long": "Implements changes from specification with conditional docstring updates and linting for complete workflows, followed by commit creation",
  "workflow_type": "complete",
  "steps": [
    {
      "type": "agent",
      "name": "Implement",
      "agent": "implementor",
      "prompt": "Implement from specification",
      "model": "base",
      "description": "Core implementation step"
    },
    {
      "type": "agent",
      "name": "Update Docstrings",
      "agent": "docstring",
      "prompt": "Update docstrings",
      "model": "base",
      "condition": {
        "if_condition": "state.workflow_type == 'complete'"
      },
      "description": "Updates docstrings for complete workflows only"
    },
    {
      "type": "agent",
      "name": "Lint",
      "agent": "plan",
      "prompt": "Run all configured linters",
      "model": "light",
      "condition": {
        "if_condition": "state.workflow_type == 'complete'"
      },
      "description": "Lints code for complete workflows only"
    },
    {
      "type": "agent",
      "name": "Commit",
      "agent": "git-commit",
      "prompt": "Create semantic commit",
      "model": "light",
      "description": "Creates git commit with conventional commit message"
    }
  ]
}
```

**Key Differences:**
- Python: ~60 lines with imports, error handling, return statements
- JSON: ~30 lines with declarative structure
- Conditional logic: Python if/else → JSON condition objects
- Error handling: Python return statements → JSON engine handles automatically
- Model tiers: Python string parameters → JSON enum values

## Troubleshooting Migration

### State Access Differences

**Python:** Direct object attribute access
```python
if ctx.workflow_type == "complete":
```

**JSON:** State prefix required
```json
"if_condition": "state.workflow_type == 'complete'"
```

### Complex Logic Breakdown

If Python workflow has complex logic:

**Option 1: Extract to Helper**
```python
# adw/utils/workflow_helpers.py
def should_run_docs(ctx):
    return ctx.workflow_type == "complete" and ctx.issue_class == "feature"
```

Then use in JSON indirectly via separate steps with simple conditions.

**Option 2: Multiple Simple Steps**
Break complex condition into multiple steps:
```json
{
  "steps": [
    {
      "name": "Docs for Complete",
      "condition": {"if_condition": "state.workflow_type == 'complete'"}
    },
    {
      "name": "Docs for Feature",
      "condition": {"if_condition": "state.issue_class == 'feature'"}
    }
  ]
}
```

**Option 3: Keep as Python**
If logic is too complex, keep workflow in Python and call from JSON as an agent.

### Testing Strategy

**Parallel Testing Period:**
1. Deploy JSON workflow as `workflow-name-v2.json`
2. Run both Python and JSON workflows in parallel
3. Compare results for 5-10 issues
4. Once confident, deprecate Python workflow
5. Rename JSON workflow to final name

**Example:**
```bash
# Week 1-2: Parallel testing
uv run adw complete 123        # Python workflow
uv run adw workflow complete-v2 123  # JSON workflow

# Week 3: Switch
uv run adw workflow complete 123     # JSON workflow (now default)
```

## Migration Checklist

### Before Migration
- [ ] Identify all workflow steps in Python code
- [ ] Document conditional logic and dependencies
- [ ] Note error handling requirements
- [ ] List all agent invocations and their parameters

### During Migration
- [ ] Create JSON workflow file in `.opencode/workflow/`
- [ ] Map each Python function call to JSON step
- [ ] Convert Python conditionals to JSON condition objects
- [ ] Add retry configurations where appropriate
- [ ] Set model tiers based on task complexity
- [ ] Add descriptive names and descriptions

### After Migration
- [ ] Validate JSON against schema
- [ ] Test workflow with real issues
- [ ] Compare results with Python workflow
- [ ] Update documentation and references
- [ ] Remove or deprecate Python workflow

## See Also

- **[Workflow Engine](workflow-engine.md)** - Main workflow engine guide
- **[Workflow JSON Schema](workflow-json-schema.md)** - Complete schema reference
- **[Workflow Conditionals](workflow-conditionals.md)** - Conditional syntax guide
- **[Workflow Examples](workflow-examples.md)** - Usage patterns and examples
- **[workflow-migration-from-python.md](workflow-migration-from-python.md)** - Deprecation timeline and API migration
- **[ADR-007: Python to JSON Workflow Migration](architecture/decisions/ADR-007-python-to-json-workflow-migration.md)** - Migration decision rationale
