# Workflow Examples Guide

**Version:** 2.2.0  
**Last Updated:** 2025-11-29

## Overview

This guide demonstrates common workflow patterns and best practices through practical examples. Each example includes the complete JSON definition, explanation of key features, and usage instructions.

**Example Files:** `.opencode/workflow/examples/`

## Example Categories

- **Simple Sequential Workflow** - Basic agent steps in sequence
- **Conditional Workflow** - Steps that execute based on state
- **Composed Workflow** - Workflow referencing other workflows
- **Error-Tolerant Workflow** - Non-critical steps with continue_on_failure
- **Custom CI Workflow** - Validation pipeline for continuous integration
- **Quick Fix Workflow** - Minimal workflow for rapid patches

## Example 1: Simple Sequential Workflow

**Use Case:** Basic workflow executing steps in order without conditionals.

**File:** `.opencode/workflow/examples/sequential-example.json`

```json
{
  "name": "sequential-example",
  "version": "1.0.0",
  "description": "Simple sequential workflow demonstrating basic structure",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "Validate Input",
      "agent": "plan",
      "prompt": "Validate issue requirements",
      "model": "light"
    },
    {
      "type": "agent",
      "name": "Implement",
      "agent": "implementor",
      "prompt": "Implement feature from specification",
      "model": "base"
    },
    {
      "type": "agent",
      "name": "Commit",
      "agent": "git-commit",
      "prompt": "Create semantic commit message",
      "model": "light"
    }
  ]
}
```

**Key Features:**
- Three sequential steps
- No conditionals (all steps always execute)
- Different model tiers for different complexity levels
- Clear, descriptive step names

**Execution:**
```bash
adw workflow sequential-example <issue-number>
```

**When to Use:**
- Simple linear workflows
- Prototyping new workflows
- Learning the workflow system

## Example 2: Conditional Workflow

**Use Case:** Workflow with steps that execute based on workflow type or issue class.

**File:** `.opencode/workflow/examples/conditional-build.json`

```json
{
  "name": "conditional-build",
  "version": "1.0.0",
  "description": "Build workflow with conditional steps based on workflow type",
  "description_long": "Implements changes with optional docstring updates and linting for complete workflows",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "Implement",
      "agent": "implementor",
      "prompt": "Implement from specification",
      "model": "base",
      "description": "Core implementation - always runs"
    },
    {
      "type": "agent",
      "name": "Update Docstrings",
      "agent": "docstring",
      "prompt": "Update docstrings following repository standards",
      "model": "base",
      "condition": {
        "skip_if": "state.workflow_type == 'patch'"
      },
      "description": "Skip docstring updates for quick patches"
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
      "description": "Full linting only for complete workflows"
    },
    {
      "type": "agent",
      "name": "Commit",
      "agent": "git-commit",
      "prompt": "Create semantic commit",
      "model": "light",
      "description": "Final commit - always runs"
    }
  ]
}
```

**Key Features:**
- Conditional step execution based on `workflow_type`
- `skip_if` to exclude steps for patches
- `if_condition` to include steps only for complete workflows
- First and last steps always execute

**Execution:**
```bash
# Complete workflow - all steps run
adw workflow conditional-build <issue-number>

# For patch workflow - docstrings and linting skipped
# (Set workflow_type in issue or via label)
```

**When to Use:**
- Different validation levels for different workflow types
- Skip expensive operations for quick fixes
- Conditional documentation generation

## Example 3: Composed Workflow

**Use Case:** Building complex workflows from reusable components.

**File:** `.opencode/workflow/examples/composed-pipeline.json`

```json
{
  "name": "composed-pipeline",
  "version": "1.0.0",
  "description": "Composed workflow demonstrating workflow step references",
  "workflow_type": "complete",
  "steps": [
    {
      "type": "workflow",
      "name": "Planning Phase",
      "workflow_name": "plan"
    },
    {
      "type": "workflow",
      "name": "Build Phase",
      "workflow_name": "build"
    },
    {
      "type": "workflow",
      "name": "Test Phase",
      "workflow_name": "test"
    },
    {
      "type": "agent",
      "name": "Create Pull Request",
      "agent": "git-commit",
      "prompt": "Create pull request with summary",
      "model": "base"
    }
  ]
}
```

**Key Features:**
- References existing workflows via `workflow_name`
- Enables reuse of tested workflow components
- Combines multiple workflows into a pipeline
- Mixes workflow steps and agent steps

**Execution:**
```bash
adw workflow composed-pipeline <issue-number>
```

**When to Use:**
- Building complex multi-phase workflows
- Reusing existing workflow definitions
- Creating custom variations of standard workflows
- Maintaining separation of concerns

## Example 4: Error-Tolerant Workflow

**Use Case:** Workflow with optional steps that shouldn't fail the entire workflow.

**File:** `.opencode/workflow/examples/tolerant-validation.json`

```json
{
  "name": "tolerant-validation",
  "version": "1.0.0",
  "description": "Validation workflow with optional checks",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "Core Validation",
      "agent": "plan",
      "prompt": "Run critical validation checks",
      "model": "light",
      "description": "Must pass for workflow to continue"
    },
    {
      "type": "agent",
      "name": "Security Scan",
      "agent": "plan",
      "prompt": "Run security vulnerability scan",
      "model": "light",
      "continue_on_failure": true,
      "description": "Optional - warns if fails but doesn't block"
    },
    {
      "type": "agent",
      "name": "Performance Benchmark",
      "agent": "plan",
      "prompt": "Run performance benchmarks",
      "model": "light",
      "continue_on_failure": true,
      "description": "Optional - informational only"
    },
    {
      "type": "agent",
      "name": "Deploy",
      "agent": "deploy",
      "prompt": "Deploy to staging environment",
      "model": "base",
      "description": "Deploy if core validation passed"
    }
  ]
}
```

**Key Features:**
- `continue_on_failure: true` for optional steps
- Critical steps fail workflow if they fail
- Optional steps log warnings but don't block
- Useful for informational checks

**Execution:**
```bash
adw workflow tolerant-validation <issue-number>
```

**When to Use:**
- Optional validation steps (security scans, benchmarks)
- Informational checks that shouldn't block deployment
- Gradual rollout of new validation requirements
- External services that may be temporarily unavailable

## Example 5: Custom CI Workflow

**Use Case:** Continuous integration validation pipeline.

**File:** `.opencode/workflow/examples/custom-ci.json`

```json
{
  "name": "custom-ci",
  "version": "1.0.0",
  "description": "Custom CI workflow for validation",
  "description_long": "Runs linting, testing, and security checks. Suitable for pre-merge validation without requiring issue context",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "Lint Code",
      "agent": "plan",
      "prompt": "Run all configured linters (ruff, mypy)",
      "model": "light",
      "timeout": 300,
      "retry": {
        "max_retries": 2,
        "initial_delay": 3.0,
        "backoff": 1.5
      },
      "description": "Fast linting with minimal retries"
    },
    {
      "type": "agent",
      "name": "Run Tests",
      "agent": "tester",
      "prompt": "Execute test suite with coverage reporting",
      "model": "light",
      "timeout": 900,
      "retry": {
        "max_retries": 2,
        "initial_delay": 5.0,
        "backoff": 2.0
      },
      "description": "Full test suite with coverage"
    },
    {
      "type": "agent",
      "name": "Security Checks",
      "agent": "plan",
      "prompt": "Run security scans (bandit, safety)",
      "model": "light",
      "timeout": 300,
      "continue_on_failure": true,
      "description": "Optional security validation"
    }
  ]
}
```

**Key Features:**
- Light model tier for cost-effective CI
- Short timeouts for fast feedback
- Conservative retry configuration
- Security checks optional (won't block merge)

**Execution:**
```bash
# CI pipeline validation
adw workflow custom-ci <issue-number>
```

**When to Use:**
- Pre-commit validation
- Pull request checks
- Automated quality gates
- Cost-effective CI pipelines

## Example 6: Quick Fix Workflow

**Use Case:** Minimal workflow for rapid patches and hotfixes.

**File:** `.opencode/workflow/examples/quick-fix.json`

```json
{
  "name": "quick-fix",
  "version": "1.0.0",
  "description": "Minimal workflow for rapid fixes",
  "description_long": "Implements quick fixes with minimal overhead - suitable for urgent patches, typos, and simple bug fixes",
  "workflow_type": "patch",
  "steps": [
    {
      "type": "agent",
      "name": "Implement Fix",
      "agent": "implementor",
      "prompt": "Implement fix from issue description",
      "model": "base",
      "timeout": 300,
      "description": "Fast implementation for urgent fixes"
    },
    {
      "type": "agent",
      "name": "Commit",
      "agent": "git-commit",
      "prompt": "Create commit with conventional message",
      "model": "light",
      "timeout": 120,
      "description": "Quick commit creation"
    }
  ]
}
```

**Key Features:**
- Only 2 steps (minimal overhead)
- Short timeouts for rapid execution
- Skips testing, review, documentation
- Base model for implementation, light for commit

**Execution:**
```bash
adw workflow quick-fix <issue-number>
```

**When to Use:**
- Urgent hotfixes
- Typo corrections
- Simple bug fixes
- Documentation updates
- When speed is critical

## Comparison Matrix

| Feature | Simple | Conditional | Composed | Tolerant | CI | Quick Fix |
|---------|--------|-------------|----------|----------|----|-----------
| **Complexity** | Low | Medium | Medium | Medium | Medium | Low |
| **Steps** | 3 | 4 | 4 | 4 | 3 | 2 |
| **Conditionals** | No | Yes | No | No | No | No |
| **Composition** | No | No | Yes | No | No | No |
| **Error Tolerance** | No | No | No | Yes | Yes | No |
| **Use Case** | Learning | Production | Complex | CI/CD | CI/CD | Hotfixes |
| **Cost** | Low | Medium | High | Low | Low | Low |

## Best Practices from Examples

### Model Tier Selection

- **Light:** Linting, commits, security scans, simple validation
- **Base:** Implementation, planning, standard tasks
- **Heavy:** Complex reasoning, architecture decisions (not shown in examples)

### Timeout Guidelines

- **120s:** Simple tasks (commits, quick validation)
- **300s:** Linting, security scans
- **600s:** Standard implementation, planning
- **900s:** Testing, comprehensive operations

### Retry Configuration

- **Linting:** 2 retries, short backoff (fast feedback)
- **Tests:** 3 retries, exponential backoff (handle flakiness)
- **Implementation:** 3 retries, standard backoff (network resilience)

### Conditional Usage

- **Skip for patches:** Documentation, comprehensive linting, testing
- **Include for complete:** All validation steps, documentation generation
- **Feature-only:** User-facing documentation, API docs

## Creating Your Own Workflows

### Step 1: Choose a Template

Pick the example closest to your use case:
- Simple → Start from sequential-example
- Conditional logic needed → Start from conditional-build
- Multiple phases → Start from composed-pipeline
- Optional steps → Start from tolerant-validation
- CI validation → Start from custom-ci
- Quick operations → Start from quick-fix

### Step 2: Customize

1. Copy example to `.opencode/workflow/my-workflow.json`
2. Update `name` to match filename
3. Update `description` and `description_long`
4. Modify steps to match your requirements
5. Adjust model tiers and timeouts

### Step 3: Validate

```bash
# Validate JSON syntax
python -c "from adw.workflows.engine.parser import load_workflow; load_workflow('.opencode/workflow/my-workflow.json')"

# Test execution
adw workflow my-workflow <test-issue-number>
```

### Step 4: Iterate

1. Run workflow on test issues
2. Monitor execution times and costs
3. Adjust timeouts and model tiers
4. Add conditionals where needed
5. Document in `description_long`

## Accessing Example Files

All example workflows are available in:
```
.opencode/workflow/examples/
```

**List examples:**
```bash
ls .opencode/workflow/examples/
```

**View example:**
```bash
cat .opencode/workflow/examples/custom-ci.json
```

**Run example:**
```bash
adw workflow custom-ci <issue-number>
```

## See Also

- **[Workflow Engine](workflow-engine.md)** - Main workflow engine guide
- **[Workflow JSON Schema](workflow-json-schema.md)** - Complete schema reference
- **[Workflow Conditionals](workflow-conditionals.md)** - Conditional syntax guide
- **[Workflow Migration Guide](workflow-migration-guide.md)** - Migrate Python workflows
- **Example Files:** `.opencode/workflow/examples/` - Actual example workflow definitions
