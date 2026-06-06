# OpenCode Agents Documentation

This directory contains documentation for custom OpenCode agents used in this repository.

## What Are OpenCode Agents?

OpenCode agents are specialized AI assistants designed for specific workflows or tasks. Each agent has:
- **Clear purpose and scope** - Focused responsibilities
- **Defined permissions** - Read-only, write-specific files, or full access
- **Repository integration** - References to `.opencode/guides/` conventions
- **Tool access** - Recommended tools for the agent to use

## Available Agents

### Agent Systems

| System | Agents | Purpose |
|--------|--------|---------|
| [Review System](adw-review-system.md) | 11 agents | Multi-agent code review for PRs |
| [Build Family](adw-build-family.md) | 4 agents | Implementation and validation |
| [Planning System](planning-system.md) | 7 agents | Multi-reviewer implementation-plan validation |
| Plan Document Pipeline | 6 agents | Deterministic epic/feature/research/maintenance drafting + phase splitting |
| [Documentation System](documentation-system.md) | 7 agents | Documentation maintenance |

### ADW Review System (11 Agents)

Multi-agent code review with parallel specialized reviewers:

| Agent | Purpose |
|-------|---------|
| `adw-review-orchestrator` | Coordinate review, dispatch subagents |
| `adw-review-code-quality` | Style, idioms, linting |
| `adw-review-correctness` | Bugs, edge cases, numerical stability |
| `adw-review-cpp-performance` | C++ HPC optimization |
| `adw-review-python-performance` | Python scientific computing |
| `adw-review-security` | Memory, input validation, resources |
| `adw-review-test-coverage` | Test existence and quality |
| `adw-review-documentation` | Docstrings, type hints |
| `adw-review-architecture` | Module bounds, APIs (optional) |
| `adw-review-consolidation` | Merge, dedupe, rank findings |
| `adw-review-feedback-poster` | Post to GitHub/GitLab |

**See**: [adw-review-system.md](adw-review-system.md)

### Agent Creator
**File**: `agent-creator.md`  
**Purpose**: Create new custom OpenCode agents with proper documentation and configuration  
**Mode**: Read-only  
**Use when**: You need to design a new specialized agent for your workflow

**See**: [agent-creator.md](agent-creator.md)

### PR Comment Analyzer
**File**: `pr-comment-analyzer.md`  
**Purpose**: Analyze actionable PR review comments and generate implementation plans  
**Mode**: Primary (read-only with adw_spec write)  
**Use when**: PR-fix workflow needs to analyze review comments

**See**: [pr-comment-analyzer.md](pr-comment-analyzer.md)

### Plan Document Drafters

**Files**: `plan-orchestrator.md`, `plan-epic-drafter.md`, `plan-feature-drafter.md`,
`plan-research-drafter.md`, `plan-maintenance-drafter.md`, `plan-phase-splitter.md`  
**Purpose**: Classify plan scope, orchestrate deterministic drafting, and generate first-pass
epic/feature/research/maintenance plan documents from runtime templates, then enforce phase sizing
before review. The canonical default dispatch order is `epic -> feature -> research -> maintenance`,
with any research-first exception documented only when dependency analysis makes it explicit.  
**Mode**: One primary orchestrator + subagent drafters  
**Use when**: Running E15 plan-document drafting workflows

**See**:
- [plan-orchestrator.md](plan-orchestrator.md)
- [plan-epic-drafter.md](plan-epic-drafter.md)
- [plan-feature-drafter.md](plan-feature-drafter.md)
- [plan-research-drafter.md](plan-research-drafter.md)
- [plan-maintenance-drafter.md](plan-maintenance-drafter.md)
- [plan-phase-splitter.md](plan-phase-splitter.md)

Standalone maintenance flow (quick example):

```bash
WORKTREE_PATH="$(uv run adw spec read --adw-id <id> --field worktree_path --raw)"
uv run adw plans create --type maintenance --title "Stabilize pipeline" --id M{n} --cwd "$WORKTREE_PATH"
uv run adw plans add-phase --plan-id M{n} --title "Draft maintenance plan" --cwd "$WORKTREE_PATH"
uv run adw workflow plan <issue-number> --adw-id <adw-id>
```

For full maintenance drafter contract details (canonical keys, section path shape, completion
payload), see [plan-maintenance-drafter.md](plan-maintenance-drafter.md).

### Plan-Issue Generation (E15-F7 status)

`plan-issue-generator` was delivered in E15-F7-P1 (agent definition + contract
tests). E15-F7-P2 completed workflow wiring in `plan-fix.json` with a
precondition gate before Generate Issues.

- **Current runtime in `plan-fix`**: `plan-issue-generator`
- **Guardrail**: Generate Issues fails closed unless `Ship` is recorded in
  `state.workflow_steps_completed`

### Auto-Mode Bootstrap

**File**: `auto-mode-bootstrap.md`
**Purpose**: Bootstrap deterministic auto-mode after issue generation by
ensuring the source branch exists from `main` without switching to it,
pushing that named branch directly, building the manifest, and labeling
created issues with `auto:enabled`
**Mode**: Primary
**Use when**: Running the second step of `generate-auto`

**See**: [auto-mode-bootstrap.md](auto-mode-bootstrap.md)

## Agent Locations

- **Active agents**: `.opencode/agent/` - Agents currently available in this repository
- **Templates**: `adw/templates/opencode_config/agent/` - Agent templates for new repositories
- **Documentation**: `.opencode/guides/agents/` - This directory (usage guides and examples)

## Creating a New Agent

To create a new agent:

1. **Use the Agent Creator agent**:
   ```
   "I need an agent to [specific task] with [permissions and scope]"
   ```

2. **Or manually create**:
   - Create agent definition: `.opencode/agent/[name].md`
   - Create documentation: `.opencode/guides/agents/[name].md`
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
mode: primary | subagent
permission:
  "*": deny
  read: allow
  adw_spec_messages: allow
  feedback_log: allow
---

Agent instructions in markdown format.
```

## Runtime Mode Taxonomy

ADW runtime uses role taxonomy rather than `read`/`write`/`all` mode labels:

- **`primary`**: Top-level workflow agent that orchestrates phases and may dispatch subagents.
- **`subagent`**: Focused worker invoked by a primary agent for scoped tasks.

Capability boundaries are enforced by each agent's explicit `permission:` map in
frontmatter, using a deny-by-default baseline (`"*": deny`) with explicit
`allow`/`deny`/`ask` semantics and targeted allowlisting.

**Security principle**: Keep least privilege—enable only tools required for the agent's purpose.

## Agent Design Best Practices

1. **Focused scope**: Create specialized agents rather than general-purpose ones
2. **Minimum permissions**: Start with `"*": deny` and allow only required tools/capabilities
3. **Write restrictions**: For mutating agents, explicitly constrain file paths/types (for example, `.md`-only scopes)
4. **Convention integration**: Reference repository guides in `.opencode/guides/`
5. **Clear documentation**: Provide usage examples and troubleshooting
6. **Tool recommendations**: Suggest relevant tools for the agent to use
7. **Composition**: Design agents that work together in workflows

## Common Agent Types

- **Implementation**: Usually `mode: primary` with mutating permissions for scoped code changes
- **Planning**: Usually `mode: primary` with mostly read + targeted state writes
- **Review**: Usually `mode: primary` or `subagent` with read-focused permissions
- **Documentation**: Usually `mode: primary`/`subagent` with `.md`-scoped mutating permissions
- **Testing**: Usually `mode: subagent` with test execution + scoped test-file writes
- **Refactoring**: Usually `mode: primary` with tightly scoped mutating permissions
- **Maintenance**: Usually `mode: primary` with minimal required mutating permissions

## Documentation Standards

Each agent should have:
- **Agent definition**: `.opencode/agent/[name].md` with complete instructions
- **Usage guide**: `.opencode/guides/agents/[name].md` with examples
- **Entry in this README**: Brief description and link

## Tools Available to Agents

Agents can use these tools when allowlisted in their `permission:` map. Prefer
split least-privilege wrappers in new/updated agent definitions; use legacy
compatibility wrappers only when required by compatibility windows and always
label them as compatibility-only per [AGENTS compatibility policy](../../../AGENTS.md#compatibility-window-retirement-gates).

Examples of preferred split wrappers:
- `adw_spec_read` / `adw_spec_write` / `adw_spec_messages` (instead of monolithic `adw_spec`)
- `git_diff` / `git_stage` / `git_commit` / `git_branch` / `git_merge` / `git_worktree` (instead of `git_operations`)
- `platform_pr_write` / `platform_pr_read` / `platform_issue_read` / `platform_issue_write` / `platform_label_write` / `platform_rate_limit_read` (instead of broad `platform_operations` for migrated commands)

Agents can also use utility tools such as:
- **`get_version`**: Get project version information
- **`get_datetime`**: Get current date/time for timestamps (UTC by default, America/Denver when `localtime` is true)
- **`run_pytest`**: Execute tests with coverage (Python projects)
- **`adw`**: ADW workflow operations (compatibility surface; use focused wrappers where available)

## Repository Conventions

All agents should reference these guides:
- `.opencode/guides/architecture_reference.md` - Design principles and patterns
- `.opencode/guides/code_style.md` - Coding conventions
- `.opencode/guides/testing_guide.md` - Test framework and patterns
- `.opencode/guides/linting_guide.md` - Code quality standards
- `.opencode/guides/docstring_guide.md` - Documentation format
- `.opencode/guides/documentation_guide.md` - Doc file standards
- `.opencode/guides/review_guide.md` - Review criteria

## See Also

- **OpenCode Agent Documentation**: https://opencode.ai/docs/agents/
- **Agent Templates**: `adw/templates/opencode_config/agent/`
- **Active Agents**: `.opencode/agent/` directory
- **Repository Guides**: `.opencode/guides/` directory

## Contributing

To add a new agent:
1. Create agent definition in `.opencode/agent/[name].md`
2. Create documentation in `.opencode/guides/agents/[name].md`
3. Update this README with new agent entry
4. Test agent with example scenarios
5. Share with team and gather feedback
