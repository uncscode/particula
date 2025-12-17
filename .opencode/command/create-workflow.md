---
description: "Create custom ADW workflow with interactive builder"
---

# /create-workflow

Create a new ADW workflow JSON file with the interactive workflow builder.

## Usage

```bash
/create-workflow
```

## Description

This command launches the interactive workflow builder agent to help you create
custom workflow JSON files step-by-step with validation.

The builder will guide you through:
1. **Workflow basics** - Name, description, and type
2. **Adding steps** - One step at a time with validation
3. **Finalization** - Review and confirm the workflow

Each step is validated before being added, so you'll get immediate feedback
if something is incorrect.

## Benefits

- **No JSON expertise required** - The agent handles JSON formatting
- **Validation at each step** - Catch errors early
- **Best practice suggestions** - Agent recommends model tiers and patterns
- **Clear error messages** - Understand what went wrong and how to fix it

## Example

```
User: /create-workflow

Agent: Let's create a new workflow! 

What would you like to name your workflow?

User: deploy-staging

Agent: Great! Provide a short description:

User: Deploy code to staging environment

Agent: What type of workflow?
- complete (includes tests, review, docs)
- patch (quick: build and ship)
- custom (you define all steps)

User: custom

Agent: ✓ Created workflow 'deploy-staging'

Now let's add steps...
```

## Workflow Types

### Complete Workflow
Full validation workflow with all phases:
- Planning
- Building
- Testing
- Review
- Documentation
- Shipping

### Patch Workflow
Quick fix workflow with minimal validation:
- Planning
- Building
- Shipping

### Custom Workflow
Define your own steps and execution flow.

## Step Configuration

For each step, you'll specify:

### Step Type
- **agent**: Run a slash command or agent
- **workflow**: Execute another workflow

### Agent Step Fields
- **name**: Descriptive name for the step
- **command**: Slash command to execute (e.g., `/implement`)
- **prompt**: Instructions for the agent
- **model**: Model tier (light, base, heavy)

### Optional Conditions
- **if**: Run step only if condition is true
- **skip_if**: Skip step if condition is true

Example conditions:
- `"state.issue_class == '/feature'"` - Only for features
- `"state.needs_docs == true"` - Only if docs needed
- `"'critical' in state.labels"` - Only for critical issues

## Model Tier Selection

Choose appropriate model tier for each step:

| Tier | Use Case | Cost | Examples |
|------|----------|------|----------|
| **light** | Simple, fast tasks | Low | Commit messages, branch names |
| **base** | Standard tasks | Medium | Implementation, testing |
| **heavy** | Complex tasks | High | Architecture, debugging |

## Example Workflows

### Simple Patch Workflow
```json
{
  "name": "quick-patch",
  "version": "1.0.0",
  "description": "Quick patch for minor fixes",
  "workflow_type": "patch",
  "steps": [
    {
      "type": "agent",
      "name": "Implement",
      "command": "/implement",
      "prompt": "Implement from spec_content",
      "model": "base"
    },
    {
      "type": "agent",
      "name": "Commit",
      "command": "/commit",
      "prompt": "Create semantic commit",
      "model": "light"
    }
  ]
}
```

### Conditional Feature Workflow
```json
{
  "name": "feature-with-docs",
  "version": "1.0.0",
  "description": "Feature with conditional documentation",
  "workflow_type": "custom",
  "steps": [
    {
      "type": "agent",
      "name": "Implement",
      "command": "/implement",
      "prompt": "Implement feature from spec",
      "model": "base"
    },
    {
      "type": "agent",
      "name": "Document",
      "command": "/document",
      "prompt": "Generate documentation",
      "model": "base",
      "if": "state.needs_docs == true"
    },
    {
      "type": "agent",
      "name": "Commit",
      "command": "/commit",
      "prompt": "Create commit",
      "model": "light"
    }
  ]
}
```

## See Also

- `adw workflow list` - List available workflows
- `adw workflow help <name>` - Get workflow details
- `.opencode/workflow/` - Workflow JSON files directory
- `.opencode/workflow/schema.json` - Workflow schema reference

## Technical Notes

The workflow builder uses the `WorkflowBuilderTool` from Phase 13 to:
- Create workflows with proper structure
- Validate steps against the schema
- Provide clear error messages
- Ensure JSON correctness

Workflows are saved to `.opencode/workflow/<name>.json` and can be executed with:
```bash
adw workflow <name> <issue-number>
```
