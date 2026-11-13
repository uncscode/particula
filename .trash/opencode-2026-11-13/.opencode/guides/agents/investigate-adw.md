# Investigate-ADW Agent - Usage Guide

## Overview

The **investigate-adw** agent diagnoses ADW workflow issues by analyzing state files, logs, worktrees, and platform context. It provides root cause analysis and actionable recommendations, with the ability to fix issues upon user approval.

## When to Use

- Workflow failed or stuck at a step
- PR wasn't shipped when expected
- Tools failed during workflow execution
- State appears corrupted or inconsistent
- Worktree issues (missing, diverged, orphaned)
- Need to understand what went wrong

## Quick Start

```
Investigate adw abc12345
```

Or ask naturally:
```
Why did workflow abc12345 fail?
Check what's wrong with adw xyz98765
```

## Tool Configuration

| Tool | Enabled | Rationale |
|------|---------|-----------|
| `read`, `list`, `glob`, `grep` | ✅ | Core investigation |
| `edit`, `write` | ✅ | Can fix issues with approval |
| `task` | ✅ | Invoke subagents (tester, linter) |
| `adw_spec` | ✅ | Read/write workflow state |
| `adw` | ✅ | Check status, health |
| `git_diff` | ✅ | Check worktree status and inspect refs/diffs |
| `platform_operations` | ✅ | Fetch issue/PR context |
| `run_pytest` | ✅ | Diagnose test failures |
| `run_linters` | ✅ | Check linting issues |
| `bash` | ❌ | Security (always disabled) |
| `webfetch`, `websearch` | ❌ | Not needed |

## What It Investigates

| Location | Purpose |
|----------|---------|
| `agents/{adw_id}/adw_state.json` | Workflow state, checkpoints, logs |
| `agents/{adw_id}/logs/` | Phase execution logs |
| `trees/{adw_id}/` | Isolated git worktree |
| GitHub/GitLab | Issue and PR status |

## Common Issues It Detects

| Issue | Symptoms | Typical Fix |
|-------|----------|-------------|
| **Stuck at step** | `failed_step` set, retries exhausted | Fix code, clear checkpoint, resume |
| **No PR shipped** | `pr_url` null | Push branch, create PR |
| **Test failures** | Tests fail in logs | Fix implementation or tests |
| **State corruption** | Missing fields | Repair or recreate state |
| **Missing worktree** | Path doesn't exist | Recreate from branch |

## Output Format

The agent produces a structured report:

```markdown
## Investigation Report: {adw_id}

### Summary
- **Issue:** {one-line description}
- **Workflow:** {type} for issue #{number}
- **Failed At:** {step}

### Root Cause
{detailed explanation}

### Evidence
- {supporting data}

### Recommended Actions
1. {action}
2. {action}

### Can I Fix This?
{Yes/No with explanation}
```

## Examples

### Example 1: Failed Tests

**Input:** `Investigate adw abc12345`

**Output:**
```
## Investigation Report: abc12345

### Summary
- **Issue:** Testing step failed after 3 retries
- **Workflow:** complete for issue #42
- **Failed At:** Testing

### Root Cause
Test `test_feature_validation` fails - implementation doesn't handle empty input.

### Recommended Actions
1. Fix edge case in `feature.py:45`
2. Resume: `adw workflow test 42 --adw-id abc12345 --resume`

### Can I Fix This?
Yes - I can update the code. Want me to proceed?
```

### Example 2: Missing PR

**Input:** `Why didn't xyz98765 ship?`

**Output:**
```
## Investigation Report: xyz98765

### Summary
- **Issue:** PR not created
- **Workflow:** patch for issue #99

### Root Cause
Branch exists but wasn't pushed (auth issue).

### Recommended Actions
1. Push branch manually
2. Check auth: `gh auth status`

### Can I Fix This?
Yes - I can push and create the PR.
```

### Example 3: Quick Health Check

**Input:** `Check adw def11111`

**Output:**
```
## Investigation Report: def11111

### Summary
- **Issue:** No issues - workflow running normally
- **Current Step:** Build (in progress)

### Recommended Actions
1. Wait for completion
2. Monitor with `adw status`
```

## Finding Your ADW ID

If you don't know the ADW ID:

```bash
# Check active workflows
adw status

# Look in agents directory
ls agents/

# Search git branches
git branch | grep adw
```

## Related Agents

| Agent | Relationship |
|-------|-------------|
| `tester` | Invoked for deep test analysis |
| `linter` | Invoked for lint fixes |
| `adw-commit` | Invoked to commit fixes |
| `adw-build` | The build agent that may have failed |

## See Also

- [ADW Architecture](../architecture/architecture_outline.md)
- [Workflow Engine Guide](../workflow-engine.md)
- [Testing Guide](../testing_guide.md)
