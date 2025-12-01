# Agent Fallback System

## Overview

The ADW system includes an **agent fallback mechanism** that automatically handles missing agent configurations by falling back to the `adw_default` agent. This prevents workflow failures when agent files are not present and provides clear warnings about the fallback behavior.

## Problem Statement

### Agent vs. Command Confusion

There was confusion between:
- **OpenCode Agents** (`.opencode/agent/*.md`) - Define agent behavior and context
- **OpenCode Commands** (`.opencode/command/*.md`) - Define specific tasks to execute
- **Agent Output Directory Constants** (`AGENT_PLANNER`, `AGENT_IMPLEMENTOR`) - Organization for output files

The ADW codebase used constants like `AGENT_PLANNER = "complete_planner"` which suggested an OpenCode agent named `complete_planner` should exist, but no such agent file was present.

### What Actually Exists

Available OpenCode agents in `.opencode/agent/`:
- `agent_creator.md`
- `architecture-planner.md`
- `implementor.md`
- `issue-creator-executor.md`
- `issue-generator.md`
- `tester.md`

Missing agents referenced in code:
- `complete_planner` (referenced in `AGENT_PLANNER` constant)
- `issue_classifier` (referenced in `AGENT_CLASSIFIER` constant)
- `branch_generator` (referenced in `AGENT_BRANCH_GENERATOR` constant)
- `pr_creator` (referenced in `AGENT_PR_CREATOR` constant)

## Solution: Agent Fallback System

### Automatic Fallback

When an agent is not found, the system:

1. **Logs a clear warning** with all available agents listed
2. **Falls back to the `adw_default` agent** - A general-purpose agent that handles custom slash commands and workflow operations
3. **Continues workflow execution** without failing

This allows workflows to proceed even when specific agent configurations are missing.

### Implementation

#### Updated `validate_agent_exists()`

```python
def validate_agent_exists(
    agent_name: str, 
    project_root: Optional[str] = None, 
    fallback_to_default: bool = True
) -> str:
    """Validate that agent configuration exists, with fallback to adw_default agent.
    
    Args:
        agent_name: Name of the agent to validate
        project_root: Optional path to project root
        fallback_to_default: If True, falls back to "adw_default" agent when requested
            agent is not found. If False, raises ValueError on missing agent.
    
    Returns:
        Agent name to use (either the requested agent or "adw_default" if fallback enabled)
    
    Raises:
        ValueError: If agent not found and fallback_to_default is False
    """
    # ADW default fallback agent
    adw_fallback_agent = "adw_default"
    
    # Allow ADW default agent
    if agent_name == adw_fallback_agent:
        return agent_name
    
    # Get available agents
    available_agents = get_available_agents(project_root)
    
    # Check if agent exists
    if agent_name not in available_agents:
        available_list = ", ".join(available_agents)
        error_msg = f"Agent '{agent_name}' not found. Available agents: {available_list}"
        
        if fallback_to_default:
            logger.warning(
                f"{error_msg}. Falling back to ADW default agent '{adw_fallback_agent}'.",
                requested_agent=agent_name,
                fallback_agent=adw_fallback_agent,
                available_agents=available_list,
            )
            return adw_fallback_agent
        else:
            raise ValueError(error_msg)
    
    return agent_name
```

#### Updated `execute_opencode_agent()`

```python
def execute_opencode_agent(
    prompt: str,
    adw_id: str,
    agent_name: Optional[str] = None,
    ...
) -> AgentPromptResponse:
    """Execute an OpenCode agent with automatic fallback."""
    # Default agent_name to "adw_default" if None
    if agent_name is None:
        agent_name = "adw_default"
    
    # Get project root
    project_root = get_project_root()
    
    # Validate agent exists and get fallback if needed
    agent_name = validate_agent_exists(agent_name, project_root, fallback_to_default=True)
    
    # Continue with execution using fallback agent if necessary
    ...
```

## Warning Messages

When fallback occurs, the system logs a structured warning:

```json
{
  "timestamp": "2025-11-28T23:20:57.461848",
  "adw_id": "validation",
  "component": "agent_validation",
  "level": "WARNING",
  "message": "Agent 'complete_planner' not found. Available agents: adw_default, agent_creator, architecture-planner, implementor, issue-creator-executor, issue-generator, tester. Falling back to ADW default agent 'adw_default'.",
  "requested_agent": "complete_planner",
  "fallback_agent": "adw_default",
  "available_agents": "adw_default, agent_creator, architecture-planner, implementor, issue-creator-executor, issue-generator, tester"
}
```

## Agent Output Directory Constants

The constants in `adw/workflows/operations/constants.py` serve as **output directory names** for organizational purposes, not necessarily OpenCode agent names:

```python
# Agent name constants - used for agent output directories
# NOTE: These are OUTPUT DIRECTORY names, not OpenCode agent names.
# OpenCode agents (in .opencode/agent/) execute commands (in .opencode/command/).
# These constants define where output files are stored for organizational purposes.

AGENT_PLANNER = "complete_planner"
"""Identifier for the complete planner agent OUTPUT DIRECTORY.

Used for agent output directory naming only (e.g., agents/abc123/complete_planner/).
The actual OpenCode agent used may vary (build, architecture-planner, etc.) but
output is consistently stored in this directory for easier navigation.
"""
```

This allows:
- **Consistent output organization** regardless of which actual agent executes
- **Easy navigation** to find planning, implementation, testing outputs
- **Backwards compatibility** with existing output paths

## Testing

Comprehensive tests verify the fallback system:

### Validation Tests

```python
def test_validate_agent_exists_fallback_enabled():
    """Test validation falls back to 'adw_default' agent when agent not found."""
    mock_get_agents.return_value = ["adw_default", "implementor", "tester"]
    
    result = validate_agent_exists("complete_planner", "/mock/path", fallback_to_default=True)
    assert result == "adw_default"
    
    # Verify warning was logged
    mock_log.warning.assert_called_once()

def test_validate_agent_exists_fallback_disabled():
    """Test validation fails when agent not found and fallback disabled."""
    mock_get_agents.return_value = ["adw_default", "implementor", "tester"]
    
    with pytest.raises(ValueError) as exc_info:
        validate_agent_exists("nonexistent", "/mock/path", fallback_to_default=False)
    
    assert "Agent 'nonexistent' not found" in str(exc_info.value)
```

### Execution Tests

```python
def test_execute_opencode_agent_fallback_to_adw_default():
    """Test execution falls back to 'adw_default' agent when requested agent not found."""
    mock_validate.return_value = "adw_default"
    
    response = execute_opencode_agent("test prompt", "abc123", agent_name="complete_planner")
    
    # Verify validate_agent_exists was called with fallback enabled
    mock_validate.assert_called_once_with(
        "complete_planner", "/tmp/test_project", fallback_to_default=True
    )
    
    # Verify build_opencode_command was called with "adw_default" (the fallback)
    assert call_args["agent_name"] == "adw_default"
    
    # Verify execution succeeded despite agent not existing
    assert response.success is True
```

## Benefits

### 1. Workflow Resilience

Workflows continue executing even when specific agent configurations are missing:

```bash
$ uv run adw patch 405
Created ADW ID: 280809cb
INFO - Starting patch workflow for issue #405
...
WARNING - Agent 'complete_planner' not found. Falling back to 'adw_default' agent.
INFO - Starting OpenCode agent execution: opencode run --agent adw_default ...
# Workflow continues successfully
```

### 2. Clear Error Messages

When agents are missing, users see:
- **What agent was requested** (`complete_planner`)
- **All available agents** (adw_default, agent_creator, architecture-planner, ...)
- **What fallback is being used** (`adw_default`)
- **Why fallback is happening** ("Agent not found")

### 3. Gradual Migration Path

Teams can:
- Start with minimal agent configurations (just the `adw_default` agent)
- Add specialized agents incrementally (implementor, tester, architecture-planner, etc.)
- Not worry about breaking existing workflows during migration

### 4. Consistent Output Organization

Output files remain organized in consistent directories (`agents/abc123/complete_planner/`) regardless of which actual agent executes.

## When to Create Custom Agents

### Use Default `adw_default` Agent When:

- Task is straightforward implementation from spec
- Need general-purpose workflow orchestration
- Executing custom slash commands
- No specialized context or guardrails needed beyond repository documentation

### Create Custom Agent When:

- Task requires specialized context (architecture patterns, testing conventions)
- Need to enforce specific guardrails (code style, design patterns)
- Want to provide domain-specific instructions
- Need consistent behavior across multiple executions
- Specialized agent offers better performance than general-purpose fallback

### Example: Creating `complete_planner` Agent

If you want consistent planning behavior beyond the default agent:

1. **Create agent file**: `.opencode/agent/complete_planner.md`
2. **Define agent context**: Repository-specific planning guidelines
3. **Add guardrails**: Enforce planning format, token conservation, etc.
4. **Test thoroughly**: Ensure agent produces consistent, high-quality plans

Once created, the system will automatically use it instead of falling back to `adw_default`.

## Architecture Decision

### Why Fallback to `adw_default` Instead of Failing?

**Alternative 1: Fail workflow when agent missing**
- ❌ Breaks existing workflows immediately
- ❌ Requires all agents to exist before any workflow runs
- ❌ Harder to deploy and test incrementally

**Alternative 2: Fallback to `adw_default` with warning** (chosen)
- ✅ Workflows continue executing with general-purpose agent
- ✅ Clear warnings guide users to create missing agents
- ✅ Gradual migration path
- ✅ System remains functional during agent development
- ✅ `adw_default` understands ADW tools and repository context

### Why `adw_default` Instead of OpenCode's Built-in `build` Agent?

**`adw_default` Advantages:**
- ✅ Understands ADW-specific tools (`adw_spec`, `adw`, `run_pytest`)
- ✅ Knows how to read repository documentation (README.md, docs/Agent/)
- ✅ Can handle custom slash commands
- ✅ Aware of ADW workflow structure and state management
- ✅ Designed for ADW repository conventions

**OpenCode `build` Agent Limitations:**
- ❌ Generic agent with no ADW-specific knowledge
- ❌ Doesn't know about `adw_spec` tool or workflow state
- ❌ No understanding of repository documentation structure
- ❌ Cannot handle ADW-specific custom commands

### Why Return Agent Name Instead of Raising?

```python
# Before: validate_agent_exists() -> None (raises on error)
validate_agent_exists("complete_planner")  # Raises ValueError

# After: validate_agent_exists() -> str (returns fallback)
agent = validate_agent_exists("complete_planner")  # Returns "adw_default"
```

Benefits:
- **Single source of truth** for agent name resolution
- **Clearer call sites** (`agent = validate_agent_exists(...)`)
- **Easier testing** (check return value instead of exception handling)
- **Better integration** with fallback logic

## Future Enhancements

### 1. Configurable Fallback Behavior

Allow per-workflow fallback configuration:

```python
# In adw_state.json or config
"agent_fallback": {
    "enabled": true,
    "fallback_agent": "adw_default",
    "warn_on_fallback": true
}
```

### 2. Agent Recommendation System

When fallback occurs, suggest which custom agent to create:

```
WARNING - Agent 'complete_planner' not found.
Falling back to 'adw_default' agent.

SUGGESTION: Create .opencode/agent/complete_planner.md to define
specialized planning behavior. See .opencode/agent/implementor.md
or .opencode/agent/adw_default.md for examples.
```

### 3. Fallback Analytics

Track fallback frequency to identify which agents should be prioritized:

```python
# Collect metrics
fallback_count["complete_planner"] += 1

# Generate report
print(f"Most frequent fallbacks: {sorted_by_count(fallback_count)}")
# Output: complete_planner (15x), issue_classifier (8x), ...
```

## Related Documentation

- **Agent Architecture**: `docs/Agent/architecture_reference.md`
- **OpenCode Integration**: `adw/core/opencode.py`
- **Workflow Operations**: `adw/workflows/operations/`
- **Testing Guide**: `docs/Agent/testing_guide.md`

## See Also

- **Issue #405**: Original issue that revealed agent fallback need
- **Constants Documentation**: `adw/workflows/operations/constants.py`
- **Agent Discovery**: `adw/core/opencode.py` - `get_available_agents()`
