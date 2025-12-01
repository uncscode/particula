# OpenCode Agents Documentation

This directory contains documentation for custom OpenCode agents used in this repository.

## What Are OpenCode Agents?

OpenCode agents are specialized AI assistants designed for specific workflows or tasks. Each agent has:
- **Clear purpose and scope** - Focused responsibilities
- **Defined permissions** - Read-only, write-specific files, or full access
- **Repository integration** - References to docs/Agent/ conventions
- **Tool access** - Recommended tools for the agent to use

## Available Agents

### Agent Creator
**File**: `agent-creator.md`  
**Purpose**: Create new custom OpenCode agents with proper documentation and configuration  
**Mode**: Read-only  
**Use when**: You need to design a new specialized agent for your workflow

**See**: [agent-creator.md](agent-creator.md)

## Agent Locations

- **Active agents**: `.opencode/agent/` - Agents currently available in this repository
- **Templates**: `adw/templates/opencode_config/agent/` - Agent templates for new repositories
- **Documentation**: `docs/Agent/agents/` - This directory (usage guides and examples)

## Creating a New Agent

To create a new agent:

1. **Use the Agent Creator agent**:
   ```
   "I need an agent to [specific task] with [permissions and scope]"
   ```

2. **Or manually create**:
   - Create agent definition: `.opencode/agent/[name].md`
   - Create documentation: `docs/Agent/agents/[name].md`
   - Follow OpenCode agent format: https://opencode.ai/docs/agents/

3. **Document here**:
   - Add entry to this README under "Available Agents"
   - Include purpose, mode, and usage guidance

## Agent File Structure

OpenCode agents use markdown with YAML frontmatter:

```markdown
---
description: >-
  Multi-line description with:
  - When to use this agent
  - What it does
  - Example scenarios
mode: read | write | all
---

Agent instructions in markdown format.
```

## Permission Modes

- **`read`**: Agent can only read files (review, analysis, recommendations)
- **`write`**: Agent can read and write files (implementation, documentation, refactoring)
- **`all`**: Full access including tool usage (orchestration, complex workflows)

**Security principle**: Use minimum permissions required for the agent's purpose.

## Agent Design Best Practices

1. **Focused scope**: Create specialized agents rather than general-purpose ones
2. **Minimum permissions**: Default to read-only unless write access is essential
3. **File restrictions**: For write mode, explicitly restrict file types (e.g., only `.md` files)
4. **Convention integration**: Reference repository guides in `docs/Agent/`
5. **Clear documentation**: Provide usage examples and troubleshooting
6. **Tool recommendations**: Suggest relevant tools for the agent to use
7. **Composition**: Design agents that work together in workflows

## Common Agent Types

- **Implementation**: Execute plans and write code (`mode: write` or `all`)
- **Planning**: Design architecture and specs (`mode: read` or limited `write`)
- **Review**: Analyze code quality (`mode: read`)
- **Documentation**: Update README and guides (`mode: write`, `.md` files only)
- **Testing**: Generate and run tests (`mode: write`, `*_test.*` files only)
- **Refactoring**: Improve code structure (`mode: write`)
- **Maintenance**: Update dependencies, fix linting (`mode: write`)

## Documentation Standards

Each agent should have:
- **Agent definition**: `.opencode/agent/[name].md` with complete instructions
- **Usage guide**: `docs/Agent/agents/[name].md` with examples
- **Entry in this README**: Brief description and link

## Tools Available to Agents

Agents can use these tools:
- **`get_version`**: Get project version information
- **`get_date`**: Get current date/time for timestamps
- **`run_pytest`**: Execute tests with coverage (Python projects)
- **`adw`**: ADW workflow operations

## Repository Conventions

All agents should reference these guides:
- `docs/Agent/architecture_reference.md` - Design principles and patterns
- `docs/Agent/code_style.md` - Coding conventions
- `docs/Agent/testing_guide.md` - Test framework and patterns
- `docs/Agent/linting_guide.md` - Code quality standards
- `docs/Agent/docstring_guide.md` - Documentation format
- `docs/Agent/documentation_guide.md` - Doc file standards
- `docs/Agent/review_guide.md` - Review criteria

## See Also

- **OpenCode Agent Documentation**: https://opencode.ai/docs/agents/
- **Agent Templates**: `adw/templates/opencode_config/agent/`
- **Active Agents**: `.opencode/agent/` directory
- **Repository Guides**: `docs/Agent/` directory

## Contributing

To add a new agent:
1. Create agent definition in `.opencode/agent/[name].md`
2. Create documentation in `docs/Agent/agents/[name].md`
3. Update this README with new agent entry
4. Test agent with example scenarios
5. Share with team and gather feedback
