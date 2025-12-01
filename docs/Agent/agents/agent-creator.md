# Agent Creator - Usage Guide

## Overview

The **Agent Creator** is a specialized OpenCode agent designed to help you create custom agents for your repository. It guides you through the entire process of designing, documenting, and configuring new agents that follow OpenCode best practices and integrate seamlessly with your repository workflows.

## When to Use

Use the Agent Creator agent when you need to:

- **Create a new specialized agent** for a specific workflow or task (e.g., security reviewer, documentation updater, migration manager)
- **Design an agent with specific permissions** (read-only, write-specific file types, full access)
- **Document agent usage and best practices** with comprehensive guides and examples
- **Integrate an agent with repository conventions** by referencing docs/Agent/ guides
- **Recommend tools and context files** for the new agent to use
- **Follow security best practices** by applying the principle of least privilege
- **Ensure agent maintainability** with clear scope and documentation

## Permissions

- **Mode**: `read` (read-only agent)
- **Read Access**: Can read all repository files to understand context and conventions
- **Write Access**: **NONE** - This agent does NOT write files directly
  - Instead, it **generates agent definitions and documentation** that you review and save
  - You maintain control over what gets created and where
- **File Type Restrictions**: N/A (read-only)

## Design Philosophy

The Agent Creator follows these principles:

1. **Least Privilege**: Recommends minimum permissions needed for agent purpose
2. **Convention Integration**: Ensures agents reference repository guides (docs/Agent/)
3. **Clear Scope**: Creates focused agents with well-defined responsibilities
4. **Comprehensive Documentation**: Produces usage guides and examples
5. **Tool Recommendations**: Suggests relevant tools and context files

## Required Context Files

The Agent Creator will consult these files to design agents that align with your repository:

- `docs/Agent/architecture_reference.md` - For architecture/planning agents
- `docs/Agent/code_style.md` - For implementation agents
- `docs/Agent/testing_guide.md` - For testing-related agents
- `docs/Agent/linting_guide.md` - For code quality agents
- `docs/Agent/docstring_guide.md` - For documentation agents
- `docs/Agent/documentation_guide.md` - For doc format standards
- `docs/Agent/review_guide.md` - For review agents
- `docs/Agent/feature/` - For feature development agents
- `.opencode/agent/` - To understand existing agent patterns

## Agent Creation Process

The Agent Creator follows a structured 5-phase process:

### Phase 1: Requirements Gathering

The agent will ask clarifying questions:
- What is the agent's purpose? What specific task will it handle?
- What scope should the agent have? What should it NOT do?
- What permissions does it need? Read-only, write-specific files, or full access?
- Should it integrate with specific tools or workflows?
- What documentation or files should it reference for context?

### Phase 2: Design

The agent designs your custom agent with:
- **Clear purpose and scope** - Specific use cases and boundaries
- **Appropriate permissions** - Minimum required (read, write, or all)
- **File type restrictions** - For write-enabled agents (e.g., only .md files)
- **Required reading** - Context files the agent should consult
- **Tool recommendations** - Relevant tools for the agent to use
- **Repository integration** - References to docs/Agent/ conventions

### Phase 3: Documentation

The agent creates comprehensive documentation:
- **Agent definition file** - `.opencode/agent/[name].md` with YAML frontmatter
- **Usage guide** - `docs/Agent/agents/[name].md` with examples
- **Integration notes** - How the agent works with other agents/tools

### Phase 4: Validation

The agent validates the design against a checklist:
- Clear, focused purpose
- Appropriate permission mode
- Well-specified restrictions
- Proper tool recommendations
- Quality standards defined
- Examples included

### Phase 5: Implementation

The agent provides:
- Complete agent definition (for you to save to `.opencode/agent/`)
- Supporting documentation (for you to save to `docs/Agent/agents/`)
- Summary of design decisions and next steps

## Recommended Tools

The Agent Creator agent does NOT directly use tools (it's read-only), but it **recommends tools** for the agents it creates:

- **`get_version`** - For agents that need project version information
- **`get_date`** - For agents that need timestamps
- **`run_pytest`** - For testing agents (Python projects)
- **`adw`** - For workflow orchestration agents

## Usage Examples

### Example 1: Create a Security Review Agent

**Context**: You want an agent to review code for security vulnerabilities without making changes.

**Invocation**:
```
"I need an agent to review PRs for security issues like SQL injection, XSS, and hardcoded secrets"
```

**Agent Creator Process**:
1. **Clarifies requirements**:
   - Should it only identify issues or suggest fixes?
   - What security frameworks should it reference (OWASP, CWE)?
   - Should it have write access to add review comments?

2. **Design decisions**:
   - **Mode**: `read` (review only, no modifications)
   - **Scope**: Security-focused code review
   - **Context**: `docs/Agent/security_guide.md`, OWASP Top 10
   - **Tools**: None (analysis only)

3. **Output**:
   - Agent definition: `security_reviewer.md`
   - Documentation: `docs/Agent/agents/security-reviewer.md`
   - Security checklist and common vulnerability patterns

### Example 2: Create a Documentation Updater Agent

**Context**: You need an agent to update README files and documentation, but NOT touch code.

**Invocation**:
```
"Create an agent that updates README and documentation files based on code changes, but can't modify any code"
```

**Agent Creator Process**:
1. **Clarifies requirements**:
   - Which doc files should it update (README, CHANGELOG, guides)?
   - Should it generate docs from code or update existing docs?
   - What documentation format should it follow?

2. **Design decisions**:
   - **Mode**: `write` (needs to update files)
   - **Write restrictions**: Only `.md`, `.txt`, `.rst` files
   - **Context**: `docs/Agent/documentation_guide.md`, `docs/Agent/docstring_guide.md`
   - **Tools**: `get_version`, `get_date`

3. **Output**:
   - Agent definition: `doc_updater.md`
   - Documentation: `docs/Agent/agents/doc-updater.md`
   - File type restrictions and validation rules

### Example 3: Create a Feature Development Agent

**Context**: You want an agent to help implement new features following your repository's conventions.

**Invocation**:
```
"I need an agent to implement new features by reading the feature specs in docs/Agent/feature/ and following our architecture patterns"
```

**Agent Creator Process**:
1. **Clarifies requirements**:
   - Should it handle planning, implementation, testing, or all three?
   - What files should it be allowed to modify?
   - Should it invoke other agents (planner, implementor)?

2. **Design decisions**:
   - **Mode**: `all` (needs full workflow capabilities)
   - **Scope**: Feature implementation from spec to completion
   - **Context**: `docs/Agent/feature/`, `docs/Agent/architecture_reference.md`, `docs/Agent/testing_guide.md`
   - **Tools**: `run_pytest`, `adw`

3. **Output**:
   - Agent definition: `feature_developer.md`
   - Documentation: `docs/Agent/agents/feature-developer.md`
   - Workflow integration guide

### Example 4: Create a Test Generator Agent

**Context**: You want an agent to generate test files for existing code.

**Invocation**:
```
"Create an agent that generates pytest test files for Python modules, following our *_test.py naming convention"
```

**Agent Creator Process**:
1. **Clarifies requirements**:
   - Should it generate unit tests, integration tests, or both?
   - What test coverage percentage should it target?
   - Should it analyze existing code to generate tests?

2. **Design decisions**:
   - **Mode**: `write` (needs to create test files)
   - **Write restrictions**: Only `*_test.py` files in appropriate test directories
   - **Context**: `docs/Agent/testing_guide.md`, `docs/Agent/code_style.md`
   - **Tools**: `run_pytest` (to validate generated tests)

3. **Output**:
   - Agent definition: `test_generator.md`
   - Documentation: `docs/Agent/agents/test-generator.md`
   - Test template examples and coverage guidelines

## Agent Design Patterns

The Agent Creator recommends these patterns:

### Pattern 1: Focused Specialist
- **Use for**: Single-purpose agents with narrow scope
- **Example**: Documentation updater that only modifies README files
- **Characteristics**: Minimal permissions, fast execution, clear invocation

### Pattern 2: Workflow Orchestrator
- **Use for**: Agents that coordinate multiple steps
- **Example**: Feature development agent that plans, implements, tests
- **Characteristics**: Broader permissions, delegates tasks, manages workflow

### Pattern 3: Review and Analysis
- **Use for**: Agents that examine code without modifying
- **Example**: Security audit agent that scans for vulnerabilities
- **Characteristics**: Read-only, produces reports, safe to run anywhere

### Pattern 4: Generator and Builder
- **Use for**: Agents that create new files or structures
- **Example**: Test suite generator that creates test files
- **Characteristics**: Write permissions to specific directories, validates outputs

## Common Agent Types

The Agent Creator can help you create:

1. **Implementation Agents** - Execute plans, write code, follow standards (`mode: write` or `all`)
2. **Planning Agents** - Design architecture, create specs (`mode: read` or limited `write`)
3. **Review Agents** - Analyze quality, identify issues (`mode: read`)
4. **Documentation Agents** - Update README, guides (`mode: write`, `.md` files only)
5. **Testing Agents** - Write/execute tests (`mode: write`, `*_test.*` files only)
6. **Refactoring Agents** - Improve code structure (`mode: write`)
7. **Maintenance Agents** - Update dependencies, fix linting (`mode: write`)

## Best Practices

### When Creating Agents

1. **Start narrow**: Create focused agents rather than general-purpose ones
2. **Minimize permissions**: Use read-only unless write access is essential
3. **Specify file restrictions**: For write mode, clearly restrict file types
4. **Reference conventions**: Always link to repository guides in docs/Agent/
5. **Provide examples**: Include 3-5 concrete usage scenarios
6. **Document limitations**: Be clear about what the agent cannot do
7. **Enable composition**: Design agents that can work with other agents
8. **Test thoroughly**: Validate agent behavior with examples

### Security Considerations

- **Principle of least privilege**: Grant minimum permissions required
- **File type restrictions**: For write agents, explicitly whitelist file types
- **Read-only first**: Default to `mode: read` unless write is essential
- **Audit regularly**: Review agent permissions periodically
- **Document risks**: Note potential security implications in docs

## Limitations

The Agent Creator agent:

- **Does NOT write files directly** - It generates content for you to review and save
- **Cannot execute the agents it creates** - You must save them first
- **Does not test agents** - You should validate agent behavior after creation
- **Requires manual file creation** - You save generated agent definitions
- **Cannot modify existing agents** - It only creates new agents (you can modify them manually)

## Integration with Other Agents

- **architecture-planner**: Use Agent Creator to design specialized planning agents
- **implementor**: Create focused implementation agents for specific domains
- **Custom agents**: Agent Creator can design agents that work together in workflows

## Troubleshooting

### Issue: Agent permissions too broad
**Solution**: Request the Agent Creator to redesign with narrower scope and minimum permissions. Ask: "Can this agent work with read-only access?" or "Can we restrict write access to specific file types?"

### Issue: Agent purpose unclear
**Solution**: Provide more specific requirements. Instead of "create a helper agent", say "create an agent that validates JSON configuration files without modifying them".

### Issue: Agent doesn't reference repository conventions
**Solution**: The Agent Creator automatically includes references to docs/Agent/ guides. If missing, explicitly request: "Ensure this agent references docs/Agent/code_style.md and testing_guide.md".

### Issue: Generated agent definition has errors
**Solution**: Review the YAML frontmatter syntax and markdown format. The Agent Creator follows OpenCode documentation (https://opencode.ai/docs/agents/), so check against that reference.

### Issue: Need to modify existing agent
**Solution**: Agent Creator only creates new agents. To modify an existing agent, either:
1. Manually edit the agent file in `.opencode/agent/`
2. Ask Agent Creator to create a new version, then manually merge changes

## See Also

- **OpenCode Agent Documentation**: https://opencode.ai/docs/agents/
- **Existing Agents**: `.opencode/agent/` directory
- **Repository Conventions**: `docs/Agent/` directory
- **Agent Templates**: `adw/templates/opencode_config/agent/`
- **Feature Development Guide**: `docs/Agent/feature/`
- **Architecture Reference**: `docs/Agent/architecture_reference.md`

## Quick Reference

**Creating a new agent**:
```
"I need an agent to [specific task]. It should [permissions and scope]."
```

**Modifying permissions**:
```
"Make this a read-only agent" or "Allow write access only to .md files"
```

**Adding tool recommendations**:
```
"This agent should use the get_version tool to include version info in outputs"
```

**Referencing conventions**:
```
"Ensure the agent follows our testing conventions from docs/Agent/testing_guide.md"
```

## Next Steps After Creation

1. **Review generated content** - Carefully read the agent definition and documentation
2. **Save agent file** - Copy to `.opencode/agent/[name].md`
3. **Save documentation** - Copy to `docs/Agent/agents/[name].md`
4. **Test the agent** - Invoke it with example scenarios
5. **Iterate if needed** - Refine agent definition based on testing
6. **Document in README** - Add entry to `.opencode/agent/README.md` (if exists)
7. **Share with team** - Communicate new agent availability and usage
