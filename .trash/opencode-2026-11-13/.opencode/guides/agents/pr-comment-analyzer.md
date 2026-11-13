# PR Comment Analyzer - Usage Guide

## Overview

The PR Comment Analyzer agent analyzes actionable PR review comments and generates an adw-build-compatible implementation plan. It's the first step in the `fix` workflow (formerly `pr-fix`), replacing the need to create intermediate GitHub issues when fixing PR feedback directly.

## When to Use

- **Invoked by fix workflow**: First step when PR has `request:fix` label
- **Triggered automatically**: Cron detects `request:fix` label on PR
- **Manual invocation**: `adw workflow fix <pr-number>`
- **Not invoked directly**: Primary agent designed for workflow automation

## Invocation

**From fix Workflow:**
```json
{
  "type": "agent",
  "name": "Analyze PR Comments",
  "agent": "pr-comment-analyzer",
  "prompt": "Analyze PR review comments and generate fix plan.\n\npr_number=$ARGUMENTS\nadw_id=${adw_id}",
  "model": "base",
  "timeout": 600
}
```

**Arguments:**
- `pr_number` (required): PR number to analyze
- `adw_id` (context): Workflow identifier for spec writes
- `prefer_scope` (optional): Routes to fork/upstream; defaults to `ADW_TARGET_REPO`

## What It Does

### Phase 1: Argument Parsing
1. Extracts `pr_number` from workflow arguments (required)
2. Extracts optional `prefer_scope` for platform routing
3. Validates that `adw_id` is available in workflow context

### Phase 2: Fetch Actionable Comments
1. Calls `platform pr-comments <PR#> --actionable-only --format json`
2. Passes `--prefer-scope` when provided
3. Receives JSON with `pr.head_branch` and `comments` list
4. Filters out any resolved comments defensively

### Phase 3: Group and Classify
1. Groups comments by file and line number
2. Captures reviewer attribution for each comment
3. Classifies intent into categories:
   - **bug**: crash, incorrect behavior, exceptions
   - **test**: add tests, assertions, coverage
   - **refactor**: cleanup, duplication removal
   - **doc**: docstrings, comments
   - **style**: formatting, naming
   - **review**: default for unclassified

### Phase 4: Generate Implementation Plan
1. Creates adw-build-compatible plan with:
   - Overview (PR number, head branch)
   - Findings grouped by file/line with reviewer intent
   - Steps: one per file grouping with details and validation
   - Tests to write based on intent
   - Error handling and acceptance criteria
2. Follows plan template for adw-build compatibility

### Phase 5: Write and Verify
1. Writes plan to `spec_content` via `adw_spec write`
2. Reads back to verify non-empty and contains required sections
3. Retries once on verification failure
4. Reports final status

## Output Signals

### SUCCESS
```
PR_COMMENT_ANALYZER_SUCCESS

PR: #1450
Branch: feature/add-validation
Actionable Comments: 5
Files Affected: 3

Plan Summary:
- 3 bug fixes in adw/core/parser.py
- 1 test addition in adw/workflows/build.py
- 1 style fix in adw/utils/helpers.py

Plan written to spec_content and verified.
```

### NO_ACTIONABLE
```
PR_COMMENT_ANALYZER_NO_ACTIONABLE

PR: #1450
No actionable comments found.

Plan written noting no actions required.
```

### FAILURE
```
PR_COMMENT_ANALYZER_FAILURE: Spec write verification failed after retry

Details:
- PR: #1450
- Comments fetched: 5
- Error: spec_content empty after write
```

## Plan Template

The agent generates plans compatible with adw-build:

```markdown
# Implementation Plan: Fix actionable PR review comments for PR #1450

**PR:** #1450
**Branch:** feature/add-validation

## Overview
Summary of actionable scope (5 comments across 3 files).

## Findings (grouped by file/line)
- `adw/core/parser.py:45` - intent: bug - reviewer: @user1 - Fix null handling
- `adw/core/parser.py:67` - intent: bug - reviewer: @user1 - Add error check
- `adw/workflows/build.py:123` - intent: test - reviewer: @user2 - Add coverage

## Steps

### Step 1: Address findings in `adw/core/parser.py`
**Files:** `adw/core/parser.py`
**Details:**
- Fix null handling at line 45
- Add error check at line 67
**Validation:** Run parser tests

### Step 2: Address findings in `adw/workflows/build.py`
**Files:** `adw/workflows/build.py`
**Details:**
- Add test coverage as requested by @user2
**Validation:** Verify new tests pass

## Tests to Write
- Test null handling edge case in parser
- Test error check behavior

## Error Handling
- Fetch failures: Fail with message
- No actionable: Return NO_ACTIONABLE
- Missing head_branch: Warn but continue
- Spec write failure: Retry once then fail

## Acceptance Criteria
- [ ] Null handling fixed in parser.py:45
- [ ] Error check added in parser.py:67
- [ ] Test coverage added for build.py
- [ ] All new tests pass
```

## Error Handling

### Recoverable Errors
- **Missing head_branch**: Continue with warning (non-fatal)
- **Spec write failure**: Retry once before failing

### Unrecoverable Errors
- **Missing pr_number**: Fail immediately
- **Fetch failure**: Fail with API error details
- **Spec verification failure**: Fail after retry

## Integration with fix Workflow

The fix workflow orchestrates 6 steps:

```
pr-comment-analyzer  -->  adw-build  -->  adw-validate  -->  test  -->  adw-format  -->  ship
```

1. **pr-comment-analyzer**: Generates plan from review comments (this agent)
2. **adw-build**: Implements fixes based on plan
3. **adw-validate**: Validates implementation against plan
4. **test**: Runs test suite
5. **adw-format**: Formats code and adds docstrings
6. **ship**: Pushes changes to PR branch

## Examples

### Example 1: PR with Bug Fixes

**Scenario:** PR #1450 has 3 unresolved review comments about bugs

**Workflow Trigger:**
```bash
# Automatic: Cron detects request:fix label
# Manual: adw workflow fix 1450
```

**Agent Execution:**
```
1. Fetch: platform pr-comments 1450 --actionable-only --format json
2. Result: 3 comments, all intent=bug, in adw/core/parser.py
3. Generate: Plan with 1 step covering all 3 fixes
4. Write: spec_content updated
5. Verify: Read-back confirms Overview/Steps present
6. Exit: SUCCESS
```

### Example 2: No Actionable Comments

**Scenario:** PR #1451 has only resolved comments

**Agent Execution:**
```
1. Fetch: platform pr-comments 1451 --actionable-only --format json
2. Result: 0 comments (all resolved filtered out)
3. Generate: Minimal plan noting no actions
4. Write: spec_content with "No actionable comments"
5. Exit: NO_ACTIONABLE
```

### Example 3: Mixed Intent Comments

**Scenario:** PR #1452 has bug, test, and style comments

**Agent Execution:**
```
1. Fetch: 5 comments across 3 files
2. Classify: 2 bug, 2 test, 1 style
3. Group: By file (parser.py, build.py, helpers.py)
4. Generate: Plan with 3 steps, tests section for intent=test
5. Write and verify
6. Exit: SUCCESS
```

## Troubleshooting

### "FAILURE: Missing pr_number"
**Cause:** Workflow did not pass PR number in arguments

**Solution:**
```bash
# Verify workflow invocation
adw workflow fix <pr-number>
```

### "FAILURE: Fetch error"
**Cause:** Platform API error or authentication issue

**Solution:**
```bash
# Verify platform access
platform pr-comments <pr-number> --format json

# Check authentication
gh auth status
```

### "FAILURE: Spec verification failed"
**Cause:** `adw_spec write` succeeded but read-back empty

**Solution:**
```bash
# Check workflow state
adw spec read --adw-id <id>

# Verify worktree exists
ls -la trees/<id>/
```

### "NO_ACTIONABLE but comments exist"
**Cause:** All comments marked as resolved

**Solution:**
```bash
# Check comment status without filter
platform pr-comments <pr-number> --format json

# Verify unresolved comments exist on PR
```

## Permissions

**Tools Available:**
- `read`, `list`, `glob`, `grep`: File system exploration
- `todoread`, `todowrite`: Task tracking
- `adw_spec`: Read/write workflow state
- `platform_operations`: Fetch PR comments
- `get_datetime`: Timestamps

**Not Available:**
- `edit`, `write`, `move`: No file modifications
- `task`: No subagent invocation
- `bash`: No shell access
- `run_pytest`, `run_linters`: No test/lint execution

## References

- **Agent Definition**: `.opencode/agent/pr-comment-analyzer.md`
- **Workflow Definition**: `.opencode/workflow/fix.json`
- **Feature Plan**: `adw-docs/dev-plans/features/F18-direct-pr-fix-workflow.md`
- **Platform CLI**: `platform pr-comments --help`
- **adw-build Agent**: `adw-docs/agents/adw-build-family.md`

## See Also

- **Linter Subagent**: `adw-docs/agents/linter.md` - Code quality validation
- **ADW Build Family**: `adw-docs/agents/adw-build-family.md` - Implementation agents
- **Testing Guide**: `adw-docs/testing_guide.md` - Test patterns
- **README**: `README.md` - Complete workflow documentation
