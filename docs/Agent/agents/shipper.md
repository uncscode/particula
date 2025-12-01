# Shipper Agent - Usage Guide

## Overview

The shipper agent is a primary workflow agent that finalizes implementation changes by performing final linting validation, creating properly formatted commits, pushing branches to remote, and creating pull requests. It's the final step in the ADW workflow pipeline, ensuring code quality and proper GitHub integration before submitting changes for review.

## When to Use

- **Called by complete workflow**: After build, test, and review phases complete successfully
- **Called by patch workflow**: After quick fix implementation finishes
- **Called by document workflow**: After documentation updates are complete
- **Manual invocation**: When developer wants to ship their feature branch
- **Not for unfinished work**: Only invoke when implementation is fully complete and validated

## Permissions

- **Mode**: `write` - Full read and write access to repository
- **Read Access**: 
  - Git repository status and history
  - Workflow state from `adw_state.json`
  - Repository convention guides
  - CI configuration files
- **Write Access**: 
  - Stage and commit changes
  - Push branches to remote
  - Create GitHub pull requests
  - Update workflow state
- **Tool Access**:
  - `run_linters` - Execute linting with auto-fix
  - `adw_spec` - Load and update workflow state
  - `bash` - Git operations and GitHub CLI

## Required Context Files

The agent consults these repository-specific guides:
- `docs/Agent/linting_guide.md` - Linting tools, configuration, standards
- `docs/Agent/commit_conventions.md` - Commit message format and rules
- `docs/Agent/pr_conventions.md` - Pull request title, body, linking format
- `docs/Agent/code_style.md` - General code style conventions

## Usage Examples

### Example 1: Complete Workflow Ship

**Context:** Complete workflow has finished build, test, and review phases

**Command:**
```bash
# Invoked by complete workflow orchestrator
opencode --agent shipper --prompt "Ship implementation. Arguments: adw_id=abc12345"
```

**Expected Behavior:**
1. Load workflow state from `agents/abc12345/adw_state.json`
2. Change to worktree: `/trees/abc12345/`
3. Run linters (ruff, mypy) with auto-fix
4. Create commit: `feat: add user authentication module`
5. Push branch: `feature-issue-123-add-authentication`
6. Create PR: `feat: #123 - Add user authentication module`
7. Output success signal with PR URL

**Output:**
```
SHIPPER_SUCCESS

Pull Request: https://github.com/owner/repo/pull/456
Branch: feature-issue-123-add-authentication
Commit: a1b2c3d
Linting: passed
Files changed: 8

Summary:
- Linting: ✓ passed (0 fixes applied)
- Commit: ✓ created (feat: add user authentication module)
- Push: ✓ succeeded
- PR: ✓ created (#456)

Next steps:
- Review the PR: https://github.com/owner/repo/pull/456
- Request reviews from team members
- Monitor CI/CD pipeline status
- Merge when approved
```

### Example 2: Patch Workflow with Auto-Fixes

**Context:** Quick fix implemented, code has minor linting issues

**Command:**
```bash
opencode --agent shipper --prompt "Ship patch. Arguments: adw_id=def67890"
```

**Expected Behavior:**
1. Load workflow state
2. Run linters → Find 3 auto-fixable issues (unused imports, long lines)
3. Auto-fix issues and re-validate
4. Create commit: `fix: resolve parser IndexError on empty input`
5. Push and create PR

**Output:**
```
SHIPPER_SUCCESS

Pull Request: https://github.com/owner/repo/pull/457
Branch: fix-issue-456-indexerror
Commit: b2c3d4e
Linting: passed
Files changed: 2

Summary:
- Linting: ✓ passed (3 fixes applied)
- Commit: ✓ created (fix: resolve parser IndexError on empty input)
- Push: ✓ succeeded
- PR: ✓ created (#457)
```

### Example 4: Linting Failure Scenario

**Context:** Implementation has complex type errors that can't be auto-fixed

**Command:**
```bash
opencode --agent shipper --prompt "Ship implementation. Arguments: adw_id=ghi11223"
```

**Expected Behavior:**
1. Load workflow state
2. Run linters → Find type errors
3. Apply 5 auto-fixes (imports, formatting)
4. Re-run linters → 2 complex type errors remain
5. Fail with detailed error

**Output:**
```
SHIPPER_FAILED: Linting validation failed

Details:
- Phase: Linting
- Error: Type errors require manual intervention
- Attempted: 5 auto-fixes applied, 2 errors remain
- State: Changes staged but not committed

Remaining issues:
- adw/core/models.py:89 - Incompatible type in assignment
- adw/workflows/dispatcher.py:134 - Cannot infer type argument

To retry:
1. Fix type errors manually in worktree: /trees/ghi11223
2. Run shipper again with same adw_id
```

### Example 5: No Changes to Commit

**Context:** Working tree is clean, PR from existing commits

**Command:**
```bash
opencode --agent shipper --prompt "Ship changes. Arguments: adw_id=mno77889"
```

**Expected Behavior:**
1. Load workflow state
2. Run linters → Pass (no changes)
3. Check for changes → Working tree clean
4. Skip commit phase
5. Push branch (already up to date)
6. Create PR from existing commits

**Output:**
```
SHIPPER_SUCCESS

Pull Request: https://github.com/owner/repo/pull/459
Branch: feature-issue-999-existing-work
Commit: (no new commit)
Linting: passed
Files changed: 0 (new), 8 (existing)

Summary:
- Linting: ✓ passed (0 fixes applied)
- Commit: ⊘ skipped (working tree clean)
- Push: ✓ succeeded (already up to date)
- PR: ✓ created (#459)
```

## Best Practices

### For Workflow Orchestrators

1. **Call after validation**: Only invoke shipper after all validation phases (tests, reviews) pass
2. **Always provide adw_id**: Required parameter for loading workflow context
3. **Check success signal**: Parse output for `SHIPPER_SUCCESS` vs `SHIPPER_FAILED`
4. **Extract PR information**: Parse PR URL and number from output for state tracking
5. **Don't retry automatically**: If shipper fails, report to user for manual intervention
6. **Use skip flags carefully**: Only skip linting for documentation changes

### For Repository Configuration

1. **Keep linter config current**: Update `.github/workflows/lint.yml` with all linters
2. **Configure pre-commit hooks**: Use `.pre-commit-config.yaml` for local validation
3. **Set up branch protection**: Enforce PR reviews and status checks
4. **Ensure authentication**: Make GITHUB_PAT available in environment
5. **Document conventions**: Keep `docs/Agent/*.md` guides up to date

### For Development Workflow

1. **Ship only complete work**: Don't ship work-in-progress or partial implementations
2. **Review before shipping**: Ensure implementation matches specification
3. **Test before shipping**: Run tests locally first to catch issues early
4. **Document changes**: Update relevant documentation before shipping
5. **Clean worktree**: Ensure no unrelated changes are included

## Limitations

- **Cannot fix complex linting issues**: Type errors and architectural issues require manual fixes
- **No force push**: Cannot overwrite remote history (by design)
- **No direct main push**: Cannot push directly to main/master (enforces PR workflow)
- **Requires authentication**: Needs GITHUB_PAT or `gh auth login` for PR creation
- **Linear workflow**: Must complete linting before committing, commit before pushing

## Integration with Other Agents

### Complete Workflow Orchestrator

```python
# After build, test, review phases complete
result = invoke_agent("shipper", f"Ship implementation. Arguments: adw_id={adw_id}")

if "SHIPPER_SUCCESS" in result:
    pr_url = extract_pr_url(result)
    log(f"PR created: {pr_url}")
    mark_workflow_complete()
elif "SHIPPER_FAILED" in result:
    report_failure(result)
```

### Patch Workflow Orchestrator

```python
# After quick fix implementation
result = invoke_agent("shipper", f"Ship patch. Arguments: adw_id={adw_id}")

if "SHIPPER_SUCCESS" in result:
    log("Patch shipped successfully")
else:
    handle_failure(result)
```

### Document Workflow Orchestrator

```python
# After documentation updates
result = invoke_agent("shipper", f"Ship documentation. Arguments: adw_id={adw_id} skip_lint=true")

if "SHIPPER_SUCCESS" in result:
    pr_url = extract_pr_url(result)
    notify_documentation_team(pr_url)
```

## Troubleshooting

### "SHIPPER_FAILED: Linting validation failed"

**Cause:** Code has linting issues that couldn't be auto-fixed

**Solution:**
1. Review linting output in error details
2. Navigate to worktree: `cd /trees/<adw_id>/`
3. Fix issues manually: `ruff check adw/ --fix` and `mypy adw/`
4. Re-run shipper with same adw_id
5. Or use `skip_lint=true` if absolutely necessary (not recommended)

### "SHIPPER_FAILED: Commit creation failed"

**Cause:** Pre-commit hooks failed after 3 retry attempts

**Solution:**
1. Check pre-commit hook output in error details
2. Test hooks locally: `cd /trees/<adw_id>/ && git commit -m "test"`
3. Fix hook issues (linting, formatting, etc.)
4. Re-run shipper

### "SHIPPER_FAILED: Push rejected (non-fast-forward)"

**Cause:** Remote branch has commits not in local branch

**Solution:**
1. Navigate to worktree: `cd /trees/<adw_id>/`
2. Pull latest changes: `git pull origin <branch_name>`
3. Resolve any conflicts
4. Re-run shipper

### "SHIPPER_FAILED: GitHub authentication failed"

**Cause:** GITHUB_PAT not set or expired, gh CLI not authenticated

**Solution:**
1. Check environment: `echo $GITHUB_PAT`
2. Authenticate gh CLI: `gh auth login`
3. Verify authentication: `gh auth status`
4. Re-run shipper

### "SHIPPER_FAILED: PR creation failed - PR already exists"

**Cause:** A PR already exists for this branch

**Solution:**
1. Check existing PRs: `gh pr list --head <branch_name>`
2. Update existing PR description if needed: `gh pr edit <pr_number>`

### Commit message too generic

**Cause:** Issue title is vague or git diff provides limited context

**Solution:**
1. Update issue title to be more descriptive
2. Or manually create commit with better message
3. Re-run shipper with `skip_commit=true`

## Security Considerations

### Protected Operations

- **No force push**: Prevents rewriting shared history (git push --force denied)
- **No direct main push**: Enforces PR review workflow (git push origin main denied)
- **Branch validation**: Ensures not pushing to protected branches
- **PR-based workflow**: All changes go through review process

### Safe Operations

- **Feature branch commits**: Isolated, safe to push
- **Conventional commits**: Clear audit trail for all changes
- **Linting validation**: Catches security issues early (via configured linters)
- **Pre-commit hooks**: Additional validation layer

### Worktree Isolation

- Each workflow operates in isolated worktree under `trees/<adw_id>/`
- Changes don't affect main working directory
- Safe parallel execution of multiple workflows
- No cross-contamination between concurrent shipments

## Related Documentation

### Primary References
- **Agent Definition**: `.opencode/agent/shipper.md` - Complete agent specification
- **Linting Guide**: `docs/Agent/linting_guide.md` - Linting standards and tools
- **Commit Conventions**: `docs/Agent/commit_conventions.md` - Commit message format
- **PR Conventions**: `docs/Agent/pr_conventions.md` - Pull request format

### Command References
- **Lint Command**: `.opencode/command/lint.md` - Linting command documentation
- **Commit Command**: `.opencode/command/commit.md` - Commit command documentation
- **PR Command**: `.opencode/command/pull_request.md` - PR creation command

### Subagent References
- **Linter Subagent**: `docs/Agent/agents/linter.md` - Linting subagent guide
- **Git-Commit Subagent**: `docs/Agent/agents/git-commit.md` - Commit subagent guide

### System Documentation
- **ADW Workflow System**: `README.md` - Complete ADW documentation
- **State Management**: `adw/state/manager.py` - Workflow state handling
- **Git Operations**: `adw/git/operations.py` - Git command wrappers
- **Worktree Management**: `adw/git/worktree.py` - Isolated workspace handling
