# Execute-Plan Agent - Usage Guide

## Overview

The execute-plan agent converts implementation plans into executable todos and systematically implements them with full automation, parallel execution where possible, and automatic test validation.

## When to Use

- **After plan-work agent**: Execute the implementation plan created by plan-work
- **Automated workflows**: Part of ADW complete/patch workflows  
- **Zero-touch implementation**: When you need fully autonomous code implementation
- **Parallel execution**: When plan has independent steps that can run concurrently

## Invocation

**CLI Command:**
```bash
# Execute plan for specific ADW workflow
uv run adw workflow run execute-plan <issue-number> --adw-id <adw-id>

# Example
uv run adw workflow run execute-plan 123 --adw-id abc12345
```

**Requirements:**
- `issue-number`: GitHub issue number
- `adw-id`: ADW workflow identifier (from plan-work agent)
- Workspace must exist with `spec_content` populated
- Must be in project root (agent navigates to worktree)

## What It Does

### Phase 1: Setup & Context Loading
1. Parses arguments to get `issue_number` and `adw_id`
2. Loads workspace context from `adw_state.json`:
   - `worktree_path` - isolated workspace location
   - `spec_content` - implementation plan to execute
   - Issue metadata and branch information
3. **Moves to worktree** before any implementation work
4. Verifies worktree is valid git workspace

### Phase 2: Plan Parsing & Todo Creation
1. Reads implementation plan from `spec_content`
2. **Automatically converts** plan steps to todo list
3. Analyzes dependencies between steps
4. Assigns priorities based on:
   - Critical path items (high priority)
   - Dependent steps (medium priority)
   - Documentation/cleanup (low priority)
5. Identifies steps that can run in parallel

### Phase 3: Task Execution
1. Groups independent high-priority tasks for parallel execution
2. Executes dependent tasks sequentially
3. For each task:
   - Marks as `in_progress` in todo list
   - Implements changes following repository conventions
   - Searches codebase for patterns if unclear
   - Makes autonomous decisions (never asks questions)
   - Marks as `completed` when done
4. Tracks progress through todo list updates

### Phase 4: Automated Validation
1. **Calls tester agent** automatically after implementation
2. Tester runs full test suite using `run_pytest`
3. Tester attempts to fix failures (up to 3 retries)
4. Returns pass/fail result

### Phase 5: Retry Logic (If Needed)
- **Attempt 1:** Direct implementation following plan
- **Attempt 2:** Enhanced context gathering from codebase
- **Attempt 3:** Minimal viable solution focused on passing tests
- **After 3 attempts:** Reports failure with detailed diagnostics

### Phase 6: Commit Changes
1. **Calls git-commit subagent** automatically after validation
2. Git-commit generates conventional commit message
3. Creates commit with proper format and issue linking
4. Handles pre-commit hook failures (up to 3 retries)
5. Reports commit hash and stats

### Phase 7: Completion Reporting
1. Verifies 100% task completion
2. Validates acceptance criteria from plan
3. Includes commit hash and message in report
4. Outputs completion signal or failure report

## Output

### Success Output
```
EXECUTE_PLAN_COMPLETE

Task Completion: 5/5 tasks completed (100%)

Summary:
- Added input validation to data parser
- Implemented comprehensive error handling
- Created unit tests covering edge cases
- Updated docstrings following repository style
- All tests passing (52/52)

Commit: a1b2c3d - feat: add data validation module
Files changed: 2 (+70/-0)

Test Results: All tests passed successfully
```

### Failure Output
```
EXECUTE_PLAN_FAILED: Test failures after 3 retry attempts

Retry Attempts: 3/3 exhausted

Summary:
- Completed: 4/5 tasks
- Failed tasks: ["Add integration tests (test framework mismatch)"]
- Error details: pytest expects *_test.py but found test_*.py pattern

Last attempt:
- Renamed test files to *_test.py pattern
- Tests still fail due to import path issues

Commit Status: Uncommitted (tests must pass before commit)
Files changed: 1 (+25/-0)

Recommendation: Manual intervention required for test framework configuration
```

## Example Workflows

### Example 1: Bug Fix Implementation

**Scenario:** Fix IndexError in data parser (Issue #123)

**Plan (from plan-work):**
```markdown
## Steps
### Step 1: Add Input Validation
Files: adw/utils/parser.py
Details: Add bounds checking before array access
### Step 2: Add Error Tests
Files: adw/utils/tests/parser_test.py
Details: Test empty input and short arrays
```

**Execution:**
```bash
uv run adw workflow run execute-plan 123 --adw-id abc12345
```

**What Happens:**
1. Loads context, moves to `trees/abc12345/`
2. Creates todos:
   - Task 1: Add validation to parser.py (priority: high)
   - Task 2: Add tests to parser_test.py (priority: high)
3. **Executes in parallel** (different files, no dependencies)
4. Both tasks complete
5. Calls tester agent → all tests pass
6. Outputs `EXECUTE_PLAN_COMPLETE`

### Example 2: Feature Implementation with Dependencies

**Scenario:** Add new data loader module (Issue #456)

**Plan:**
```markdown
## Steps
### Step 1: Create Module Skeleton
Files: adw/loaders/data_loader.py
### Step 2: Implement Core Function
Files: adw/loaders/data_loader.py
Dependencies: Step 1
### Step 3: Add Helper Functions
Files: adw/loaders/helpers.py
### Step 4: Write Unit Tests
Files: adw/loaders/tests/data_loader_test.py
Dependencies: Step 2
### Step 5: Write Integration Tests
Files: adw/loaders/tests/integration_test.py
Dependencies: Step 3
```

**Execution:**
```bash
uv run adw workflow run execute-plan 456 --adw-id def67890
```

**What Happens:**
1. Creates todos with dependency analysis:
   - Task 1: Skeleton (high, no deps)
   - Task 2: Core function (high, depends on 1)
   - Task 3: Helpers (high, no deps)
   - Task 4: Unit tests (medium, depends on 2)
   - Task 5: Integration tests (medium, depends on 3)
2. **Parallel execution:** Tasks 1 & 3 run concurrently
3. **Sequential execution:** Task 2 waits for 1, task 4 waits for 2
4. All tasks complete
5. Tester validates → all pass
6. Git-commit creates commit → `feat: add data loader module`
7. Outputs `EXECUTE_PLAN_COMPLETE`

### Example 3: Retry After Test Failure

**Scenario:** Implementation passes but tests fail

**Execution:**
```bash
uv run adw workflow run execute-plan 789 --adw-id ghi11223
```

**What Happens:**
1. All implementation tasks complete successfully
2. Tester runs tests → 2 failures detected
3. **Attempt 1:** Tester tries to fix failures
   - Fixes syntax error, re-runs tests
   - 1 test still failing (logic error)
4. **Attempt 2:** Execute-plan reviews code, finds logic bug
   - Fixes logic error
   - Tester re-runs tests → all pass
5. Git-commit creates commit → `fix: resolve data loader logic error`
6. Outputs `EXECUTE_PLAN_COMPLETE` (after 2 attempts)

## Autonomous Decision-Making

The execute-plan agent **never asks questions** and makes decisions autonomously:

### When Requirements Are Unclear
- **Searches codebase** for similar patterns
- **Reads related files** for context
- **Infers intent** from plan and acceptance criteria
- **Chooses simplest approach** that satisfies requirements

### When Multiple Approaches Exist
- **Follows repository conventions** from `docs/Agent/` guides
- **Prioritizes consistency** with existing code patterns
- **Defaults to simpler solution** over complex optimizations

### When Encountering Errors
- **Attempt 1:** Fix directly, retry
- **Attempt 2:** Gather more context, adjust approach
- **Attempt 3:** Minimal viable solution

## Parallel Execution

### Safe for Parallel Execution
- ✅ Tasks modifying different files
- ✅ Tasks with no dependencies
- ✅ Independent test file creation
- ✅ Documentation updates in separate files

### Must Be Sequential
- ❌ Tasks modifying the same file
- ❌ Tasks with explicit dependencies (Step X depends on Step Y)
- ❌ Tasks requiring output from previous task
- ❌ Database schema migrations (order matters)

### Performance Benefits
- **Single file modification:** Sequential execution
- **Multiple independent files:** ~2-3x faster with parallel execution
- **Large feature with modules:** Up to 5x faster for independent components

## Integration with ADW Workflows

### Complete Workflow
```bash
# Full workflow: plan → execute-plan → commit → test → review → document → ship
uv run adw complete 123

# Internally runs:
# 1. plan-work agent creates implementation plan
# 2. execute-plan agent implements the plan (automatic)
# 3. git-commit subagent commits changes (called by execute-plan)
# 4. tester agent validates (called by execute-plan)
# 5. review, document, ship phases follow
```

### Manual Two-Phase Workflow
```bash
# Phase 1: Create plan
uv run adw workflow run plan 123

# Phase 2: Execute plan (you specify adw_id from phase 1)
uv run adw workflow run execute-plan 123 --adw-id abc12345
# This automatically:
# - Implements all plan steps
# - Commits changes with conventional format
# - Runs tests for validation
```

## Git Commit Integration

Execute-plan automatically calls the **git-commit subagent** after completing implementation tasks and validating with tests.

### Commit Flow

1. **All tasks complete** → execute-plan marks all todos as completed
2. **Tests pass** → tester agent reports success
3. **Git-commit called** → Automatically creates commit with:
   - Conventional commit format (`feat:`, `fix:`, `docs:`, etc.)
   - Context from issue title and git diff
   - Issue linking with `Fixes #<number>`
   - Pre-commit hook handling (auto-retry if hooks modify files)

### Commit Type Selection

Git-commit determines commit type from:
- **Workflow type:** complete→feat, patch→fix, document→docs
- **Issue labels:** bug→fix, chore→chore, test→test
- **File analysis:** Only tests→test, only docs→docs

### Commit Failure Handling

If git-commit fails (e.g., pre-commit hooks fail):
- **Attempt 1-3:** Git-commit retries with hook modifications
- **After 3 failures:** Execute-plan reports `EXECUTE_PLAN_FAILED` with uncommitted changes
- **Manual fix needed:** Developer must fix linting/hook errors manually

### Example Commit Messages

Generated by git-commit subagent:
```
feat: add user authentication module
fix: resolve parser IndexError on empty input
docs: expand API documentation with examples
test: add integration tests for workflow engine
chore: update dependencies to latest versions
```

See `docs/Agent/agents/git-commit.md` for complete git-commit documentation.

## Troubleshooting

### "EXECUTE_PLAN_FAILED: No worktree found"
**Cause:** Workspace not created or adw_id invalid

**Solution:**
```bash
# Verify workspace exists
cat agents/abc12345/adw_state.json | jq .worktree_path
ls trees/abc12345/

# If missing, run plan-work first
uv run adw workflow run plan 123
```

### "EXECUTE_PLAN_FAILED: No implementation plan found"
**Cause:** `spec_content` field empty in `adw_state.json`

**Solution:**
```bash
# Check if plan exists
adw_spec read abc12345 spec_content

# If empty, run plan-work to generate plan
uv run adw workflow run plan 123 --adw-id abc12345
```

> **Note:** `uv run adw spec read|write|delete --help` now introspects
> `ADWStateData`, surfaces the curated common fields list, and still reminds you
> to run [`adw spec list`](../../../README.md#spec-state-commands) for the full
> schema. `read` and `write` keep their `spec_content` default, while `delete`
> remains `--field`-required but benefits from the same contextual guidance.

### "Test failures after 3 retry attempts"
**Cause:** Tests fail and agent couldn't fix automatically

**Solution:**
```bash
# Review what was attempted
cat agents/abc12345/adw_state.json | jq .workflow_logs

# Manual investigation
cd trees/abc12345/
pytest -v  # See specific test failures

# Fix manually, then resume workflow
```

### Agent doesn't detect parallel opportunities
**Cause:** Plan steps have implicit dependencies not specified

**Solution:**
- Update plan to explicitly mark dependencies
- Or steps may actually need sequential execution (modifying same file)

## Best Practices

### For Plan Authors (plan-work)
- **Mark dependencies explicitly** in plan steps
- **Break large files into modules** to enable parallel work
- **Order steps by dependencies** (prerequisites first)
- **Be specific about file paths** to avoid ambiguity

### For Execute-Plan Usage
- **Always run after plan-work** to ensure plan exists
- **Monitor first execution** to verify parallel detection works
- **Check logs if failures occur** at `agents/{adw_id}/adw_state.json`
- **Trust retry logic** - agent attempts 3 fixes before failing

### For Repository Setup
- **Keep `docs/Agent/` guides current** - agent relies on these
- **Use consistent patterns** across codebase for easier inference
- **Maintain test conventions** so tester agent works smoothly

## Performance Characteristics

| Scenario | Sequential Time | Parallel Time | Speedup |
|----------|----------------|---------------|---------|
| Single file modification | 5 min | 5 min | 1x (no parallel) |
| 3 independent files | 15 min | 6 min | 2.5x |
| Large feature (10 modules) | 50 min | 12 min | 4x |
| Bug fix (1 file + tests) | 8 min | 4 min | 2x |

*Times are approximate and depend on complexity*

## References

- **ADW System**: `README.md` - Complete ADW workflow documentation
- **Plan-Work Agent**: `docs/Agent/agents/plan-work.md` - Creates plans that execute-plan implements
- **Tester Agent**: `.opencode/agent/tester.md` - Called automatically for validation
- **Architecture**: `docs/Agent/architecture_reference.md` - Patterns agent follows
- **Code Style**: `docs/Agent/code_style.md` - Conventions agent adheres to
- **Testing Guide**: `docs/Agent/testing_guide.md` - Test framework details

## See Also

- **Dynamic Workflow System**: `adw/cli_dynamic.py` - CLI implementation with retry logic
- **Workflow Engine**: `adw/workflows/engine/` - Execution and resume capabilities
- **State Management**: `adw/state/manager.py` - How workflow state is tracked
