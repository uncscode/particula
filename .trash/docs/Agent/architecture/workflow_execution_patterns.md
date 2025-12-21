# Workflow Execution Patterns

**Version:** 1.0.0
**Last Updated:** 2025-11-22

## Overview

This guide documents the correct patterns for executing workflows in the ADW system. It explains how backend abstraction works, when to use each execution layer, and how to create new workflows following established patterns.

## How Backend Abstraction Actually Works

### Architecture Overview

Backend abstraction in ADW is implemented in `adw/core/agent.py` through **dependency injection**, NOT through direct imports from `adw/backends/`.

The abstraction works through three key mechanisms:

1. **`execute_agent()` accepts optional parameters:**
   - `cli_path`: Path to CLI executable
   - `command_builder`: Function that builds CLI commands
   - `install_checker`: Function that checks if CLI is installed

2. **Different backends inject different implementations:**
   - `OpenCodeBackend` injects `build_opencode_command()`

3. **Workflows use `execute_template()` which delegates to `execute_agent()`**

### Three-Layer Architecture

```
Layer 1 (Workflows):
  - Import from adw.core.agent
  - Use execute_template() with slash commands
  - Backend-agnostic by design

Layer 2 (Execution Engine):
  - adw/core/agent.py provides execution interface
  - execute_agent() with pluggable builders
  - Model selection via get_model_for_slash_command()

Layer 3 (Backend Implementations):
  - adw/backends/ implements AgentBackend interface
  - Each backend wraps Layer 2 with its command builder
  - Workflows NEVER import from this layer
```

## Correct Pattern for Workflows

### Use execute_template() for All Workflows

**✅ CORRECT:**
```python
from adw.core.agent import execute_template
from adw.core.models import AgentTemplateRequest

request = AgentTemplateRequest(
    agent_name="my_workflow",
    slash_command="/my_command",
    args=["arg1", "arg2"],
    adw_id=adw_id,
    working_dir=str(working_dir) if working_dir else None,
)
response = execute_template(request)
```

**❌ WRONG:**
```python
from adw.backends import BackendFactory  # Wrong layer!
backend = BackendFactory.get_backend()   # Don't do this
```

### Why This Pattern?

1. **Automatic model selection** based on ADW state (base vs heavy)
2. **Consistent output file paths** managed by execute_template()
3. **Centralized slash command management** in .opencode/command/
4. **Backend-agnostic by design** - works with all backends
5. **Follows established patterns** used by 6+ workflows

## When to Use Each Layer

### Use `execute_template()`:
- ✅ All workflow operations
- ✅ Any time you need to run an agent
- ✅ When you want automatic model selection

### Use `execute_agent()` with builders:
- ✅ Implementing a new backend type
- ✅ Testing with custom command builders
- ❌ NOT for regular workflows

### Use `adw/backends/` directly:
- ✅ Creating a new backend implementation
- ✅ Testing with mock backends
- ❌ NEVER in workflow code

## Creating Slash Commands

### Step 1: Create Command File

Create a markdown file in `.opencode/command/`:

```bash
.opencode/command/my_command.md
```

**Example:**
```markdown
# My Command

**IMPORTANT**: This command expects 2 arguments:
1. Input data (JSON string or text)
2. Configuration options (JSON string)

Your task is to process the input data according to the configuration...

## Output Format

Return your response as JSON:
```json
{
  "result": "processed output",
  "status": "success"
}
```
```

### Step 2: Register in SlashCommand Type

Update `adw/core/models.py`:

```python
SlashCommand = Literal[
    # ... existing commands
    "/my_command",  # Add your command
    # ... more commands
]
```

Also update the `validate_slash_commands()` function's `defined_commands` set:

```python
defined_commands = {
    # ... existing commands
    "/my_command",  # Add your command
    # ... more commands
}
```

### Step 3: (Optional) Configure Model Selection

By default, all commands use:
- `base` model set → `sonnet`
- `heavy` model set → `opus`

If your command should use lightweight models even in heavy mode, add it to `LIGHTWEIGHT_COMMANDS` in `adw/core/agent.py`:

```python
LIGHTWEIGHT_COMMANDS = {
    "/recommend_workflow",
    "/generate_branch_name",
    "/my_lightweight_command",  # Add here
    # ...
}
```

### Step 4: Use in Workflow

```python
from adw.core.agent import execute_template
from adw.core.models import AgentTemplateRequest

def my_workflow_function(data: str, config: dict, adw_id: str) -> AgentPromptResponse:
    """Execute my workflow using /my_command."""

    # Prepare arguments
    args = [data, json.dumps(config)]

    # Create request
    request = AgentTemplateRequest(
        agent_name="my_workflow",
        slash_command="/my_command",
        args=args,
        adw_id=adw_id,
        working_dir=None,  # Optional
    )

    # Execute
    response = execute_template(request)

    return response
```

## How Backend Switching Works

Backend selection happens via environment variables, NOT code changes:

```bash
export AGENT_CLI_TOOL=opencode  # Use OpenCode
```

The process:
1. `BackendConfigManager` reads `AGENT_CLI_TOOL` environment variable
2. `BackendFactory` instantiates the correct backend
3. Backend wraps `execute_agent()` with its command builder
4. Workflows call `execute_template()` which uses the active backend
5. **NO CODE CHANGES NEEDED IN WORKFLOWS**

## Examples from Production Workflows

### Example 1: Implementation Workflow

From `adw/workflows/operations/implementation.py`:

```python
from adw.core.agent import execute_template
from adw.core.models import AgentTemplateRequest

def implement_plan(
    plan_file: str,
    adw_id: str,
    logger: LoggerType,
    agent_name: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> AgentPromptResponse:
    """Execute a plan file using the /implement slash command."""

    implement_template_request = AgentTemplateRequest(
        agent_name=agent_name or AGENT_IMPLEMENTOR,
        slash_command="/implement",
        args=[plan_file],
        adw_id=adw_id,
        working_dir=working_dir,
    )

    return execute_template(implement_template_request)
```

### Example 2: Planning Workflow

From `adw/workflows/operations/planning.py`:

```python
def build_plan(
    issue: GitHubIssue,
    command: Literal["/patch", "/feature"],
    adw_id: str,
    logger: LoggerType,
    working_dir: Optional[str] = None,
    parent_issue: Optional[GitHubIssue] = None,
) -> AgentPromptResponse:
    """Build implementation plan for a GitHub issue."""

    # Prepare data as JSON
    minimal_issue_json = issue.model_dump_json(
        by_alias=True,
        include={"number", "title", "body"}
    )

    # Build args list
    args = [str(issue.number), adw_id, minimal_issue_json]

    # Add optional parent context
    if parent_issue:
        parent_issue_json = parent_issue.model_dump_json(
            by_alias=True,
            include={"number", "title", "body"}
        )
        args.append(parent_issue_json)

    # Create and execute request
    request = AgentTemplateRequest(
        agent_name=AGENT_PLANNER,
        slash_command=command,
        args=args,
        adw_id=adw_id,
        working_dir=working_dir,
    )

    return execute_template(request)
```

### Example 3: Issue Interpretation Workflow

From `adw/workflows/issue_interpret.py`:

```python
def run_issue_interpret_workflow(ctx: WorkflowContext) -> WorkflowResult:
    """Execute issue interpretation workflow."""

    # Get source text and available labels
    source_text = ctx.metadata.get("source_text")
    available_labels = _get_available_labels()

    # Build args
    args = [source_text, available_labels]

    # Execute with automatic model selection
    request = AgentTemplateRequest(
        agent_name="issue_interpret",
        slash_command="/interpret_issue",
        args=args,
        adw_id=ctx.adw_id,
        working_dir=str(ctx.working_dir) if ctx.working_dir else None,
    )

    agent_result = execute_template(request)
    # ... process result
```

## Common Patterns

### Pattern: Passing Complex Data

When passing complex data structures, serialize to JSON:

```python
# Good: Serialize complex data
data = {"users": [...], "config": {...}}
args = [json.dumps(data)]

# Also good: Pass Pydantic models as JSON
user_json = user.model_dump_json(by_alias=True)
args = [user_json]
```

### Pattern: Optional Arguments

Slash commands can accept optional arguments:

```python
args = [required_arg1, required_arg2]

# Add optional argument conditionally
if optional_context:
    args.append(optional_context)

request = AgentTemplateRequest(
    slash_command="/my_command",
    args=args,
    # ...
)
```

Update your slash command to document optional arguments:

```markdown
# My Command

**IMPORTANT**: This command expects 2-3 arguments:
1. Required argument 1
2. Required argument 2
3. (Optional) Additional context
```

### Pattern: Model Selection Override

Normally, `execute_template()` selects models automatically based on ADW state. You can override if needed:

```python
# Automatic (recommended)
request = AgentTemplateRequest(
    slash_command="/my_command",
    args=args,
    adw_id=adw_id,
    # model is automatically selected
)

# Manual override (rarely needed)
request = AgentTemplateRequest(
    slash_command="/my_command",
    args=args,
    adw_id=adw_id,
    model="opus",  # Force specific model
)
```

## Migration Checklist

When migrating old workflows to the new pattern:

- [ ] Remove old import: `from adw.core.agent import execute_agent`
- [ ] Remove old import: `from adw.core.models import AgentPromptRequest`
- [ ] Add new import: `from adw.core.agent import execute_template`
- [ ] Add new import: `from adw.core.models import AgentTemplateRequest`
- [ ] Create slash command file in `.opencode/command/`
- [ ] Register slash command in `SlashCommand` type
- [ ] Update slash command in `validate_slash_commands()`
- [ ] Remove manual model selection logic (let `execute_template` handle it)
- [ ] Replace inline prompts with arguments to slash command
- [ ] Update tests to mock `execute_template` instead of `execute_agent`
- [ ] Test that automatic model selection works (check ADW state integration)

## Testing Workflow Execution

### Unit Testing Pattern

```python
from unittest.mock import patch
from adw.core.models import AgentPromptResponse

@patch("adw.workflows.my_module.execute_template")
def test_my_workflow(mock_execute):
    """Test workflow execution."""

    # Mock response
    mock_execute.return_value = AgentPromptResponse(
        success=True,
        output='{"result": "success"}',
    )

    # Run workflow
    result = my_workflow_function(data="test", adw_id="test123")

    # Verify
    assert result.success
    mock_execute.assert_called_once()

    # Verify correct slash command used
    call_args = mock_execute.call_args[0][0]  # Get AgentTemplateRequest
    assert call_args.slash_command == "/my_command"
    assert call_args.args == ["test"]
```

### Integration Testing

Test that slash command is properly configured:

```python
def test_my_command_slash_command_configured():
    """Verify /my_command is registered."""
    from adw.core.models import SlashCommand
    from adw.core.agent import SLASH_COMMAND_MODEL_MAP

    # Check type literal
    assert "/my_command" in SlashCommand.__args__

    # Check model map
    assert "/my_command" in SLASH_COMMAND_MODEL_MAP
    assert SLASH_COMMAND_MODEL_MAP["/my_command"]["base"] == "sonnet"
    assert SLASH_COMMAND_MODEL_MAP["/my_command"]["heavy"] == "opus"
```

## Troubleshooting

### Command Not Found

If you see "Command not found" errors:

1. Verify file exists: `.opencode/command/my_command.md`
2. Check `SlashCommand` type includes `/my_command`
3. Check `validate_slash_commands()` includes `/my_command`
4. Run validation: `python -c "from adw.core.models import validate_slash_commands; print(validate_slash_commands())"`

### Wrong Model Selected

If the wrong model is being used:

1. Check ADW state: `python -c "from adw.state.manager import ADWState; print(ADWState.load('your_adw_id'))"`
2. Verify model map: `python -c "from adw.core.agent import SLASH_COMMAND_MODEL_MAP; print(SLASH_COMMAND_MODEL_MAP['/my_command'])"`
3. Check if command is in `LIGHTWEIGHT_COMMANDS` (always uses sonnet)

### Arguments Not Passed Correctly

If arguments aren't reaching the agent:

1. Verify args are strings (serialize complex data to JSON)
2. Check slash command documents expected argument count
3. Ensure `execute_template` receives correct `AgentTemplateRequest`

## See Also

- **docs/Agent/architecture/architecture_guide.md**: Overall system architecture
- **docs/Agent/architecture/architecture_outline.md**: High-level component overview
- **docs/Agent/architecture/decisions/002-backend-abstraction.md**: ADR for backend abstraction design
- **docs/Agent/architecture/decisions/003-opencode-backend.md**: ADR for OpenCode backend implementation
- **docs/Agent/code_style.md**: Code style and conventions
- **docs/Agent/testing_guide.md**: Testing strategies and patterns

## Summary

### Key Takeaways

1. **Use `execute_template()` for all workflows** - it's the high-level, backend-agnostic interface
2. **Never import from `adw/backends/` in workflows** - that's internal infrastructure
3. **Backend abstraction works via pluggable builders** in `adw/core/agent.py`
4. **Slash commands centralize agent instructions** in `.opencode/command/`
5. **Model selection is automatic** based on ADW state and command configuration
6. **Backend switching is via environment variables** - no code changes needed

### Anti-Patterns to Avoid

❌ Importing from `adw/backends/` in workflow code
❌ Using `BackendFactory` directly in workflows
❌ Inline prompts instead of slash commands
❌ Manual model selection (let `execute_template` handle it)
❌ Calling `execute_agent()` directly in new workflows

### Best Practices

✅ Use `execute_template()` with slash commands
✅ Create descriptive slash command files
✅ Document expected arguments in command files
✅ Let ADW state drive model selection
✅ Serialize complex data to JSON for args
✅ Test slash command registration
✅ Follow patterns from existing workflows
