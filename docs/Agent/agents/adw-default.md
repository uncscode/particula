# ADW Default Agent - Usage Guide

## Overview

The `adw_default` agent is the general-purpose fallback agent for the ADW (AI Developer Workflow) system. It handles custom slash commands, serves as a fallback when specialized agents are not needed, and provides comprehensive access to ADW tools and repository context.

## When to Use

### Primary Use Cases

**1. Custom Slash Commands**
When users invoke custom slash commands that need execution:
- `/analyze-dependencies` - Analyze project dependencies
- `/check-security` - Security vulnerability scan
- `/optimize-performance` - Performance analysis
- Any repository-specific custom command

**2. Fallback Agent**
When no specialized agent is needed:
- Simple implementations that don't require `implementor` agent
- General workflow tasks outside specialized agent scope
- Coordination between workflow phases

**3. State Management Tasks**
Operations involving workflow state:
- Reading implementation plans
- Updating progress logs
- Tracking workflow metadata
- Querying workflow status

**4. Repository Analysis**
Understanding codebase structure and conventions:
- Architecture queries
- Codebase navigation
- Convention lookups
- Documentation exploration

## Permissions

- **Mode**: Uses repository base permissions from `.opencode` configuration
- **Read Access**: All repository files, workflow state via `adw_spec` tool
- **Write Access**: Code, documentation, configuration files (based on repo permissions)
- **Tool Access**: Full ADW tool suite (`adw_spec`, `adw`, `run_pytest`, standard OpenCode tools)

## Core Capabilities

### 1. Workflow State Management

The agent interacts with workflow state exclusively through the `adw_spec` tool:

```bash
# List all state fields
adw_spec --command list --adw-id abc12345

# Read implementation plan
adw_spec --command read --adw-id abc12345 --field spec_content

# Update workflow logs
adw_spec --command write --adw-id abc12345 --field workflow_logs --content "Progress update" --append
```

**Key State Fields**:
- `spec_content` - Implementation plan/specification
- `issue_number` - GitHub issue being processed
- `branch_name` - Git branch name
- `worktree_path` - Absolute path to isolated worktree
- `plan_file` - Path to plan markdown file
- `issue_class` - Issue type (`/chore`, `/bug`, `/feature`)
- `workflow_logs` - Timestamped log messages

### 2. Repository Context Awareness

The agent reads repository context from:
- **`README.md`** - System overview, installation, architecture
- **`docs/Agent/README.md`** - Complete agent documentation index

And references specific guides as needed:
- Code style, testing, linting, documentation conventions
- Commit and PR formatting standards
- Architecture patterns and design decisions

### 3. ADW Tool Integration

**Primary Tool: adw_spec**
- Read/write workflow state
- Access implementation plans
- Update progress logs
- Query workflow metadata

**Supporting Tools**:
- `adw` - CLI operations (status, health, workflows)
- `run_pytest` - Execute tests with coverage
- `get_version` - Project version information
- `get_date` - Timestamps for logs

## Usage Examples

### Example 1: Execute Custom Slash Command

**Scenario**: User wants to analyze test coverage with custom command

**Command**: `/analyze-test-coverage`

**Expected Behavior**:
1. Agent reads testing guide to understand framework (pytest)
2. Runs `run_pytest` with coverage options
3. Analyzes coverage report
4. Updates workflow logs with findings
5. Provides detailed coverage summary

**Output**:
```markdown
Test coverage analysis complete:

**Coverage Metrics**:
- Overall: 85.2%
- adw/core/: 92%
- adw/workflows/: 78%
- adw/github/: 81%

**Recommendations**:
- Increase coverage in adw/workflows/ module
- Add integration tests for workflow orchestration
- Focus on error handling paths (lines 123-145 in dispatcher.py)

I've updated the workflow logs with these findings.
```

### Example 2: Fallback Implementation Task

**Scenario**: Build phase needs implementation without specialized `implementor` agent

**Context**: ADW ID `a1b2c3d4`, issue #123, feature implementation

**Process**:
1. Read implementation plan from `spec_content`
2. Read worktree path for file operations
3. Implement changes following code style guide
4. Write tests following testing guide
5. Update spec content with implementation notes

**Output**:
```markdown
Implementation complete for issue #123:

**Files Modified**:
- `adw/core/new_feature.py` - Core logic (150 lines)
- `adw/core/tests/new_feature_test.py` - Test coverage (80 lines)

**Tests**: ✅ All passing (15 tests, 95% coverage)

**Code Quality**:
- ✅ Linting passed (ruff)
- ✅ Type checking passed (mypy)
- ✅ Follows repository code style

**Next Steps**:
1. Run `/review` to validate quality
2. Run `/document` to generate docs
3. Run `/ship` to create PR

Updated workflow state with implementation details.
```

### Example 3: State Management

**Scenario**: Query current workflow state and implementation plan

**Command**: "Show me the current implementation plan for workflow a1b2c3d4"

**Process**:
1. List available state fields
2. Read `spec_content` field
3. Parse and summarize plan
4. Present structured overview

**Output**:
```markdown
Implementation Plan for Workflow `a1b2c3d4`:

**Issue**: #123 - Add user authentication system

**Type**: /feature

**Phases**:
1. ✅ Planning - Complete
2. ⚙️ Building - In Progress
3. ⏳ Testing - Pending
4. ⏳ Review - Pending
5. ⏳ Documentation - Pending

**Implementation Tasks**:
1. Create `auth.py` module with login/logout functions
2. Add user session management
3. Integrate with existing user model
4. Add authentication middleware
5. Write comprehensive tests (target: 90% coverage)

**Current Progress**: Tasks 1-2 complete, working on task 3

**Worktree**: `/home/kyle/Code/Agent/trees/a1b2c3d4`
**Branch**: `feature-123-a1b2c3d4-user-authentication`
```

### Example 4: Repository Architecture Query

**Scenario**: New developer wants to understand system architecture

**Command**: "Explain the ADW repository architecture"

**Process**:
1. Read `README.md` for overview
2. Read `docs/Agent/architecture_reference.md` for details
3. Analyze key modules in `adw/` package
4. Provide comprehensive architecture summary

**Output**:
```markdown
ADW Repository Architecture:

**Core Package Structure** (`adw/`):

1. **Core Components** (`adw/core/`):
   - `models.py` - Data types (WorkflowContext, WorkflowResult)
   - `agent.py` - OpenCode agent interface
   - `exceptions.py` - Custom exceptions

2. **Workflow Orchestration** (`adw/workflows/`):
   - `dispatcher.py` - Workflow coordination
   - `complete.py`, `patch.py` - Workflow types
   - `plan.py`, `build.py`, `test.py`, etc. - Workflow phases

3. **GitHub Integration** (`adw/github/`):
   - `operations.py` - API operations
   - `status.py` - Status comment management
   - `rate_limit.py` - API rate limiting

4. **State Management** (`adw/state/`):
   - `manager.py` - Persistent state (ADWState)

5. **Git Operations** (`adw/git/`):
   - `worktree.py` - Isolated environment management
   - `operations.py` - Git command operations

**Key Architectural Patterns**:
- **Isolated Worktrees**: Each workflow runs in `trees/{adw_id}/`
- **State-Driven**: Workflow state in `agents/{adw_id}/adw_state.json`
- **Modular Phases**: Plan → Build → Test → Review → Document → Ship
- **Concurrent Execution**: Unlimited parallel workflows

See `docs/Agent/architecture_reference.md` for complete details.
```

## Best Practices

### 1. Always Load Repository Context

Before any task, read:
- `README.md` - System overview
- `docs/Agent/README.md` - Agent documentation index
- Specific guides as needed (code_style.md, testing_guide.md, etc.)

### 2. Use adw_spec Tool Correctly

**✅ Do**:
```bash
adw_spec --command read --adw-id abc12345 --field spec_content
```

**❌ Don't**:
```bash
cat agents/abc12345/adw_state.json  # Never directly access state files
```

### 3. Work in Isolated Worktrees

Always perform file operations within the worktree:

```bash
# Read worktree path first
adw_spec --command read --adw-id abc12345 --field worktree_path

# Use worktree path for operations
cd /path/to/trees/abc12345
# Then modify files
```

### 4. Update Workflow State

After significant actions, update workflow logs:

```bash
adw_spec --command write --adw-id abc12345 --field workflow_logs \
  --content "[2024-01-15T10:30:00Z] Implementation completed" --append
```

### 5. Follow Repository Conventions

Apply standards from guides:
- Code style: snake_case, type hints, Google docstrings
- Testing: pytest, `*_test.py` suffix, 50% coverage minimum
- Linting: ruff check + format, mypy type checking
- Commits: Semantic format with issue linking

### 6. Provide Clear Output

Summarize actions with:
- What was done
- What files changed
- Test results
- Next steps
- Relevant links/references

## Limitations

### What This Agent Cannot Do

1. **Directly access state files**: Must use `adw_spec` tool
2. **Modify main repository during workflows**: Must work in worktrees
3. **Override repository permissions**: Uses base `.opencode` permissions
4. **Select models dynamically**: Uses OpenCode defaults
5. **Replace specialized agents entirely**: Serves as fallback, not replacement

### When to Use Specialized Agents Instead

- **Planning complex features**: Use `complete_planner` agent
- **Large-scale implementations**: Use `implementor` agent
- **Comprehensive code review**: Use code review agents
- **Documentation generation**: Use `documenter` agent
- **Test debugging**: Use `tester` agent

## Troubleshooting

### Issue: "ADW ID not found"

**Cause**: Invalid or non-existent workflow ID

**Solution**:
```bash
# Check active workflows
adw status

# Verify ADW ID exists
ls agents/ | grep abc12345
```

### Issue: "Cannot read spec_content"

**Cause**: Field doesn't exist or wrong ADW ID

**Solution**:
```bash
# List all available fields
adw_spec --command list --adw-id abc12345

# Check field names (case-sensitive)
```

### Issue: "Worktree not found"

**Cause**: Worktree doesn't exist or was removed

**Solution**:
```bash
# Check worktree path from state
adw_spec --command read --adw-id abc12345 --field worktree_path

# List all worktrees
git worktree list

# Verify directory exists
ls -la trees/abc12345/
```

### Issue: "Tool execution failed"

**Cause**: Incorrect tool parameters or permissions

**Solution**:
1. Check tool documentation for required parameters
2. Verify permissions allow tool usage
3. Check tool output for specific error messages
4. Ensure all prerequisites met (e.g., GitHub auth for `adw` tool)

## Integration with Other Agents

### Relationship to Specialized Agents

**adw_default serves as:**
- **Fallback**: When specialized logic not needed
- **Coordinator**: Between specialized agent phases
- **General-purpose**: For tasks outside specialized scope

**Works with:**
- `complete_planner` - Reads plans, executes general implementation
- `implementor` - Fallback for simple implementations
- `tester` - Executes tests, analyzes results
- `reviewer` - Provides general code quality checks
- `documenter` - Handles basic documentation tasks

### Workflow Phase Integration

**Entry Points** (can start workflows):
- Plan phase - Create implementation plans
- Custom commands - Execute user requests

**Mid-Workflow** (can continue workflows):
- Build phase - General implementation
- Test phase - Test execution and analysis
- Review phase - Quality validation

**Exit Points** (can complete workflows):
- Ship phase - Push changes and create PRs
- State updates - Record completion

## See Also

### Documentation
- **ADW System README**: `/home/kyle/Code/Agent/README.md` - Complete system guide
- **Agent Documentation Index**: `/home/kyle/Code/Agent/docs/Agent/README.md` - All guides
- **Architecture Reference**: `docs/Agent/architecture_reference.md` - System design
- **Code Style Guide**: `docs/Agent/code_style.md` - Python conventions
- **Testing Guide**: `docs/Agent/testing_guide.md` - Test framework and patterns

### Tools
- **adw_spec**: Workflow state management (use `--help` for details)
- **adw**: CLI operations (use `adw --help` for commands)
- **run_pytest**: Python test execution

### Related Agents
- **complete_planner**: Planning agent for complex features
- **implementor**: Specialized implementation agent
- **tester**: Test execution and validation agent
- **reviewer**: Code review and quality agent
- **documenter**: Documentation generation agent

### Configuration
- **OpenCode Config**: `.opencode/` - Agent and command configuration
- **Environment Variables**: `.env` - API keys and settings
- **ADW State**: `agents/{adw_id}/adw_state.json` - Workflow state (access via tool)
