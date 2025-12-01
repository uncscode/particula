# Git-Commit Subagent - Usage Guide

## Overview

The git-commit subagent creates properly formatted git commits following conventional commit format and repository conventions. It's designed to be called by primary workflow agents (execute-plan, implementor, etc.) to commit implementation changes automatically.

## When to Use

- **Called by execute-plan**: After completing all implementation tasks
- **Called by implementor**: After IMPLEMENTATION_COMPLETE signal
- **Called by patch agents**: After applying quick fixes
- **Called by document agents**: After updating documentation
- **Not invoked directly**: Subagent designed for workflow automation

## Invocation

**From Primary Agent:**
```python
# Minimal invocation (loads context from state)
task({
  "description": "Commit implementation changes",
  "prompt": "Create commit for workflow. Arguments: adw_id=abc12345",
  "subagent_type": "git-commit"
})

# With explicit worktree path
task({
  "description": "Commit changes in worktree",
  "prompt": "Create commit. Arguments: adw_id=def67890 worktree_path=/trees/def67890",
  "subagent_type": "git-commit"
})

# With commit type override
task({
  "description": "Commit bug fix",
  "prompt": "Create commit. Arguments: adw_id=ghi11223 commit_type=fix",
  "subagent_type": "git-commit"
})
```

**Arguments:**
- `adw_id` (required): 8-character workflow identifier
- `worktree_path` (optional): Path to worktree (loaded from state if not provided)
- `commit_type` (optional): Override commit type (feat/fix/docs/chore/test/refactor/style)

## What It Does

### Phase 1: Context Loading
1. Parses input arguments to extract `adw_id`, `worktree_path`, `commit_type`
2. Loads workflow context from `adw_state.json` via `adw_spec` tool:
   - Worktree path for git operations
   - Issue details (number, title, labels)
   - Workflow type (complete/patch/document)
   - Implementation plan for context
   - Branch name

### Phase 2: Change Analysis
1. Navigates to worktree directory
2. Verifies valid git repository
3. Analyzes changes with `git diff --stat HEAD`
4. Determines:
   - Number of files changed
   - Lines added/removed
   - File types modified (code, tests, docs)
   - Whether changes exist to commit

### Phase 3: Commit Type Determination
Priority order for determining commit type:
1. **Explicit `commit_type` argument** (highest priority)
2. **Workflow type from state:**
   - `complete`/`feature` → `feat:`
   - `patch` → `fix:`
   - `document` → `docs:`
3. **Issue labels:**
   - `bug` → `fix:`
   - `chore` → `chore:`
   - `documentation` → `docs:`
   - `test` → `test:`
4. **Git diff analysis:**
   - Only test files → `test:`
   - Only markdown files → `docs:`
   - Only config/build files → `chore:`
5. **Default:** `feat:`

### Phase 4: Message Generation
1. Reads commit conventions from `docs/Agent/commit_conventions.md`
2. Generates subject line (≤50 chars):
   - Present tense imperative ("add" not "added")
   - Lowercase (unless proper noun)
   - No period at end
   - Context from issue title and file changes
3. Optionally adds body (if complex changes)
4. Adds footer with `Fixes #<issue_number>`

### Phase 5: Commit Creation
1. Stages all changes: `git add -A`
2. Creates commit: `git commit -m "<message>"`
3. Handles pre-commit hook failures:
   - If hooks modify files, re-stages and retries
   - Up to 3 retry attempts
   - Reports hook errors after 3 failures

### Phase 6: Result Reporting
Reports back to primary agent with:
- Success: Commit hash, message, file stats
- Failure: Error details, retry attempts
- Skip: If no changes to commit

## Output Signals

### Success
```
GIT_COMMIT_SUCCESS

Commit: a1b2c3d
Message: feat: add user authentication module
Files changed: 5
Insertions: 120
Deletions: 15
Branch: feature-issue-123-add-authentication
```

### Failure
```
GIT_COMMIT_FAILED: Pre-commit hooks failed after 3 attempts

Details:
- Attempted message: feat: add new feature
- Files to commit: 8
- Error: ruff check failed with 5 linting errors
- Retry attempts: 3/3
```

### Skip
```
GIT_COMMIT_SKIPPED: No changes to commit

Status: Working tree clean
Branch: feature-issue-456-existing-work
```

## Git Command Permissions

### Allowed Commands ✅
- `git checkout *` - Switch branches
- `git add *` - Stage all changes
- `git commit *` - Create commits
- `git push -u origin <feature-branch>` - Push to feature branches
- `git diff *` - View changes
- `git status` - Check repository state
- `git branch` - Manage branches
- `git log` - View commit history

### Denied Commands ❌
- `git push origin main` - No direct push to main branch
- `git push -u origin main` - No direct push to main branch
- `git push --force` - No force push (prevents rewriting history)
- `git push -f` - No force push (short form)
- `git rebase -i` - No interactive rebase
- `git reset --hard` - No destructive resets

**Rationale:** These restrictions preserve PR-based workflow and prevent accidental history rewrites.

## Examples

### Example 1: Feature Implementation Commit

**Scenario:** Execute-plan completes feature implementation

**Primary Agent Call:**
```python
task({
  "description": "Commit feature implementation",
  "prompt": "Create commit for workflow. Arguments: adw_id=abc12345",
  "subagent_type": "git-commit"
})
```

**State Context:**
```json
{
  "adw_id": "abc12345",
  "worktree_path": "/trees/abc12345",
  "issue_number": "123",
  "issue_title": "Add user authentication module",
  "workflow_type": "complete",
  "branch_name": "feature-issue-123-add-authentication"
}
```

**Git Changes:**
```
 adw/auth/login.py           | 45 +++++++++++++++++++
 adw/auth/tests/login_test.py | 30 +++++++++++++
 docs/auth-api.md             | 20 ++++++++
```

**Generated Commit:**
```
feat: add user authentication module

Implements JWT-based authentication with password hashing
and session management. Includes comprehensive unit tests
and API documentation.

Fixes #123
```

**Subagent Output:**
```
GIT_COMMIT_SUCCESS

Commit: a1b2c3d
Message: feat: add user authentication module
Files changed: 3
Insertions: 95
Deletions: 0
Branch: feature-issue-123-add-authentication
```

### Example 2: Bug Fix with Override

**Scenario:** Implementor creates bug fix, explicitly specifies commit type

**Primary Agent Call:**
```python
task({
  "description": "Commit bug fix",
  "prompt": "Create commit. Arguments: adw_id=def67890 commit_type=fix",
  "subagent_type": "git-commit"
})
```

**Git Changes:**
```
 adw/utils/parser.py           | 5 +++--
 adw/utils/tests/parser_test.py | 12 ++++++++++++
```

**Generated Commit:**
```
fix: resolve parser IndexError on empty input

Fixes #456
```

**Subagent Output:**
```
GIT_COMMIT_SUCCESS

Commit: b2c3d4e
Message: fix: resolve parser IndexError on empty input
Files changed: 2
Insertions: 17
Deletions: 2
Branch: bug-issue-456-fix-indexerror
```

### Example 3: Pre-commit Hook Modification

**Scenario:** Commit triggers pre-commit hooks that auto-format code

**Primary Agent Call:**
```python
task({
  "description": "Commit implementation",
  "prompt": "Arguments: adw_id=ghi11223",
  "subagent_type": "git-commit"
})
```

**Execution Flow:**
```
1. Stage changes: git add -A
2. Attempt commit: git commit -m "feat: add data loader"
3. Pre-commit hook runs: ruff format
4. Hook modifies files: adw/loaders/data_loader.py
5. Retry: git add -A (re-stage formatted files)
6. Retry commit: git commit -m "feat: add data loader"
7. Success!
```

**Subagent Output:**
```
GIT_COMMIT_SUCCESS

Commit: c3d4e5f
Message: feat: add data loader module
Files changed: 4
Insertions: 85
Deletions: 10
Branch: feature-issue-789-data-loader
Retries: 1 (pre-commit hook modified files)
```

### Example 4: No Changes to Commit

**Scenario:** Primary agent calls commit but working tree is clean

**Primary Agent Call:**
```python
task({
  "description": "Commit changes",
  "prompt": "Arguments: adw_id=jkl44556",
  "subagent_type": "git-commit"
})
```

**Git Status:**
```bash
$ git status --porcelain
# (no output - clean working tree)
```

**Subagent Output:**
```
GIT_COMMIT_SKIPPED: No changes to commit

Status: Working tree clean
Branch: feature-issue-999-already-committed
```

**Primary Agent Handling:**
```python
if "GIT_COMMIT_SKIPPED" in output:
  # No changes - this is fine, continue workflow
  log("No changes to commit, proceeding")
  continue_to_next_step()
```

## Integration Patterns

### Execute-Plan Agent

After completing all implementation tasks, execute-plan calls git-commit:

```python
# In execute-plan agent, Step 10: Commit Changes
todos_complete = all(task["status"] == "completed" for task in todos)

if todos_complete:
  # All tasks done, commit the changes
  result = task({
    "description": "Commit implementation",
    "prompt": f"Create commit. Arguments: adw_id={adw_id}",
    "subagent_type": "git-commit"
  })
  
  # Parse result
  if "GIT_COMMIT_SUCCESS" in result:
    commit_hash = extract_commit_hash(result)
    state.update(last_commit_sha=commit_hash)
    log(f"Committed changes: {commit_hash}")
    proceed_to_testing()
  elif "GIT_COMMIT_FAILED" in result:
    handle_commit_failure(result)
  elif "GIT_COMMIT_SKIPPED" in result:
    # No changes is fine, continue
    proceed_to_testing()
```

### Implementor Agent

After implementation complete signal:

```python
# After outputting IMPLEMENTATION_COMPLETE
# Call git-commit to save work
commit_result = task({
  "description": "Commit implementation",
  "prompt": f"Arguments: adw_id={adw_id} worktree_path={worktree_path}",
  "subagent_type": "git-commit"
})

# Parse and report in completion signal
if "GIT_COMMIT_SUCCESS" in commit_result:
  output(f"Implementation committed: {extract_commit_hash(commit_result)}")
```

## Troubleshooting

### "GIT_COMMIT_FAILED: Not in git repository"
**Cause:** Invalid worktree path or not in git workspace

**Solution:**
```bash
# Verify worktree exists
cat agents/abc12345/adw_state.json | jq .worktree_path
ls /trees/abc12345/

# Verify it's a git repository
cd /trees/abc12345/
git rev-parse --is-inside-work-tree
```

### "GIT_COMMIT_FAILED: Pre-commit hooks failed after 3 attempts"
**Cause:** Pre-commit hooks (linting, formatting) fail validation

**Solution:**
```bash
# Check hook errors
cd /trees/abc12345/
git commit -m "test"  # See hook output

# Fix linting errors manually
ruff check adw/ --fix
ruff format adw/

# Or skip hooks temporarily (not recommended)
git commit -m "message" --no-verify
```

### "GIT_COMMIT_FAILED: Missing adw_id parameter"
**Cause:** Primary agent didn't provide adw_id argument

**Solution:**
```python
# Correct invocation - include adw_id
task({
  "description": "Commit changes",
  "prompt": "Create commit. Arguments: adw_id=abc12345",  # Include this!
  "subagent_type": "git-commit"
})
```

### Commit message too generic
**Cause:** Limited context from issue title or git diff

**Solution:**
```json
// Improve issue title to be more descriptive
{
  "issue_title": "Add user authentication module with JWT",  // Good
  "issue_title": "Add feature"  // Too generic
}

// Or use commit_type override with explicit context
task({
  "prompt": "Arguments: adw_id=abc12345 commit_type=feat"
})
```

## Best Practices

### For Primary Agents Calling Git-Commit

1. **Always provide adw_id**: Required parameter for context loading
2. **Call after work completion**: Commit only when implementation/changes are done
3. **Check for skip signal**: Handle `GIT_COMMIT_SKIPPED` gracefully (not an error)
4. **Parse commit hash**: Extract commit SHA from success output for state tracking
5. **Don't retry on failure**: If git-commit fails after 3 retries, escalate to user

### For Repository Setup

1. **Keep commit conventions current**: Update `docs/Agent/commit_conventions.md`
2. **Configure pre-commit hooks**: Use `.pre-commit-config.yaml` for automation
3. **Set up branch protection**: Enforce PR workflow, prevent direct main pushes
4. **Use descriptive issue titles**: Helps generate better commit messages

### For Workflow Design

1. **Commit frequently**: After each logical unit of work
2. **Test before commit**: Ensure code compiles/passes basic checks
3. **Include issue context**: Link commits to issues with `Fixes #<number>`
4. **Use conventional commits**: Enables automated changelog generation

## Commit Message Quality

### Good Commit Messages ✅
```
feat: add JWT-based authentication
fix: resolve memory leak in parser
docs: expand API documentation with examples
test: add integration tests for workflow engine
chore: update dependencies to latest versions
refactor: simplify error handling logic
```

### Bad Commit Messages ❌
```
feat: stuff                        # Too vague
fixed bug                          # Missing type prefix
Add new feature.                   # Capital letter, period at end
feat: added authentication         # Past tense (should be "add")
update code                        # Missing type, too generic
```

## Performance Characteristics

| Scenario | Typical Time | Notes |
|----------|--------------|-------|
| Simple commit (no hooks) | <1 second | Instant staging and commit |
| With pre-commit hooks | 2-5 seconds | Linting/formatting takes time |
| Hook modifications (retry) | 3-10 seconds | Re-staging and retry |
| Large changeset (100+ files) | 5-15 seconds | Git operations scale with size |

## Security Considerations

### Protected Operations
- **No force push**: Prevents rewriting shared history
- **No direct main push**: Enforces PR review workflow
- **No hard reset**: Prevents accidental data loss

### Safe Operations
- **Feature branch commits**: Isolated, safe to push
- **Staged commits**: All changes reviewed before commit
- **Conventional messages**: Clear audit trail

### Worktree Isolation
- Each workflow gets isolated git worktree
- Commits don't affect main working directory
- Safe parallel execution of multiple workflows

## References

- **Commit Conventions Guide**: `docs/Agent/commit_conventions.md` - Format specification
- **Original Slash Command**: `.opencode/command/commit.md` - Commit command reference
- **Execute-Plan Agent**: `.opencode/agent/execute-plan.md` - Primary caller
- **Implementor Agent**: `.opencode/agent/implementor.md` - Another caller
- **Conventional Commits**: https://www.conventionalcommits.org/ - Format standard

## See Also

- **ADW Workflow System**: `README.md` - Complete workflow documentation
- **State Management**: `adw/state/manager.py` - How workflow context is stored
- **Git Operations**: `adw/git/operations.py` - Low-level git command wrappers
- **Worktree Management**: `adw/git/worktree.py` - Isolated workspace handling
