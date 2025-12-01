# Linter Subagent - Usage Guide

## Overview

The linter subagent runs configured linters and auto-fixes code quality issues following repository conventions. It's designed to be called by primary workflow agents (execute-plan, implementor, etc.) to validate code quality before committing changes.

## When to Use

- **Called by execute-plan**: Before creating git commit
- **Called by implementor**: After implementation tasks complete
- **Called by patch agents**: Before finalizing quick fixes
- **Called by review agents**: During code quality validation
- **Not invoked directly**: Subagent designed for workflow automation

## Invocation

**From Primary Agent:**
```python
# Minimal invocation (loads context from state)
task({
  "description": "Run linters on implementation",
  "prompt": "Lint code. Arguments: adw_id=abc12345",
  "subagent_type": "linter"
})

# With explicit worktree path
task({
  "description": "Lint code in worktree",
  "prompt": "Lint code. Arguments: adw_id=def67890 worktree_path=/trees/def67890",
  "subagent_type": "linter"
})

# With target directory override
task({
  "description": "Lint specific directory",
  "prompt": "Lint code. Arguments: adw_id=ghi11223 target_dir=adw/workflows",
  "subagent_type": "linter"
})
```

**Arguments:**
- `adw_id` (required): 8-character workflow identifier
- `worktree_path` (optional): Path to worktree (loaded from state if not provided)
- `target_dir` (optional): Directory to lint (default: adw)

## What It Does

### Phase 1: Context Loading
1. Parses input arguments to extract `adw_id`, `worktree_path`, `target_dir`
2. Loads workflow context from `adw_state.json` via `adw_spec` tool
3. Changes to worktree directory for linting operations
4. Reads `.github/workflows/lint.yml` to understand configured linters

### Phase 2: Initial Linting
1. Executes `run_linters` tool with auto-fix enabled
2. Runs configured linters:
   - **ruff check --fix**: Linting with auto-fix
   - **ruff format**: Code formatting
   - **mypy**: Type checking
3. Parses results to determine pass/fail status

### Phase 3: Fix Management (If Linting Failed)
1. Parses linter error output to extract issues
2. Creates comprehensive todo list using `todowrite`:
   - File path and line numbers
   - Error codes and descriptions
   - Prioritization (high/medium/low)
   - Groups similar errors
3. Processes each todo item systematically:
   - Mark as `in_progress`
   - Read affected file
   - Apply fix using `edit` tool
   - Mark as `completed`
   - Move to next fix

### Phase 4: Re-validation
1. After all manual fixes applied, re-run linters
2. Verify all issues resolved
3. If still failing, document remaining unfixable issues

### Phase 5: Result Reporting
Reports back to primary agent with:
- Success: Which linters passed, number of fixes applied
- Failure: Remaining errors, manual intervention needed

## Output Signals

### Success
```
LINTING_SUCCESS

Linters: ruff (passed), mypy (passed)
Target: adw/
Fixes applied: 5
- Fixed F401: unused imports (3 files)
- Fixed E501: line too long (2 files)
```

### Failure
```
LINTING_FAILED: Type errors require manual intervention

Details:
- Linters failed: mypy
- Errors remaining: 3
- Fixes attempted: 7
- Manual intervention needed: Complex type incompatibilities in adw/core/models.py
```

## Linter Configuration

### Configuration Sources
The subagent reads linter configuration from:
- `.github/workflows/lint.yml` - CI workflow definition
- `pyproject.toml` - ruff and mypy settings
- `.ruff.toml` - Additional ruff configuration (if present)

### Default Linters (ADW Repository)
- **ruff check**: Fast Python linter
  - Line length: 100 characters
  - Enabled rules: F (pyflakes), E (pycodestyle errors), W (pycodestyle warnings)
  - Auto-fixes: unused imports, formatting issues
- **ruff format**: Code formatter
  - Black-compatible formatting
  - Auto-applies consistent style
- **mypy**: Static type checker
  - Checks type annotations
  - Requires manual fixes for type errors

### Permissions

**Allowed Operations ✅**
- Run `ruff check --fix` to auto-fix linting issues
- Run `ruff format` to format code
- Run `mypy` for type checking
- Read source files to understand context
- Edit source files to apply fixes
- Read CI configuration files

**Denied Operations ❌**
- Modify `pyproject.toml`, `.ruff.toml`, or `mypy.ini`
- Skip linting checks or suppress errors
- Disable specific linter rules

**Rationale:** Config changes require human review. Fixes should address root causes, not suppress warnings.

## Examples

### Example 1: All Linters Pass

**Scenario:** Execute-plan calls linter after implementation

**Primary Agent Call:**
```python
task({
  "description": "Validate code quality",
  "prompt": "Lint code. Arguments: adw_id=abc12345",
  "subagent_type": "linter"
})
```

**Linter Execution:**
```
$ ruff check adw/ --fix
All checks passed! ✅

$ ruff format adw/ --check
Already formatted ✅

$ mypy adw/ --ignore-missing-imports
Success: no issues found ✅
```

**Subagent Output:**
```
LINTING_SUCCESS

Linters: ruff (passed), mypy (passed)
Target: adw/
Fixes applied: 0
```

### Example 2: Auto-fixable Issues

**Scenario:** Implementation has unused imports and long lines

**Primary Agent Call:**
```python
task({
  "description": "Lint and fix issues",
  "prompt": "Arguments: adw_id=def67890",
  "subagent_type": "linter"
})
```

**Initial Linting:**
```
ruff check adw/ --fix
Found 5 errors:
- F401: unused import in adw/workflows/complete.py:5
- F401: unused import in adw/workflows/build.py:3
- F401: unused import in adw/core/agent.py:10
- E501: line too long (105 > 100) in adw/core/models.py:45
- E501: line too long (108 > 100) in adw/utils/helpers.py:78

Auto-fixed 5 errors ✓
```

**Subagent Output:**
```
LINTING_SUCCESS

Linters: ruff (passed), mypy (passed)
Target: adw/
Fixes applied: 5
- Fixed F401: unused imports (3 files)
- Fixed E501: line too long (2 files)
```

### Example 3: Type Errors Requiring Manual Fixes

**Scenario:** Implementation has type annotation errors

**Primary Agent Call:**
```python
task({
  "description": "Lint implementation",
  "prompt": "Arguments: adw_id=ghi11223",
  "subagent_type": "linter"
})
```

**Initial Linting:**
```
mypy adw/
adw/workflows/build.py:67: error: Argument 1 has incompatible type "str"; expected "int"
adw/core/models.py:45: error: Missing return statement
```

**Todo List Created:**
```json
[
  {
    "id": "1",
    "content": "Fix type error: Argument 1 has incompatible type str; expected int in adw/workflows/build.py:67",
    "status": "pending",
    "priority": "high"
  },
  {
    "id": "2",
    "content": "Fix missing return statement in adw/core/models.py:45",
    "status": "pending",
    "priority": "high"
  }
]
```

**Fix Process:**
```
1. Mark todo #1 as in_progress
2. Read adw/workflows/build.py
3. Apply fix: Convert str to int at line 67
4. Mark todo #1 as completed
5. Mark todo #2 as in_progress
6. Read adw/core/models.py
7. Apply fix: Add return statement at line 45
8. Mark todo #2 as completed
9. Re-run linters
```

**Final Linting:**
```
mypy adw/
Success: no issues found ✅
```

**Subagent Output:**
```
LINTING_SUCCESS

Linters: ruff (passed), mypy (passed)
Target: adw/
Fixes applied: 2
- Fixed type error in adw/workflows/build.py:67
- Added return statement in adw/core/models.py:45
```

### Example 4: Unfixable Issues

**Scenario:** Complex type errors requiring architectural changes

**Primary Agent Call:**
```python
task({
  "description": "Lint code",
  "prompt": "Arguments: adw_id=jkl44556",
  "subagent_type": "linter"
})
```

**Linting Issues:**
```
mypy adw/
adw/core/models.py:89: error: Incompatible types in assignment (complex generic)
adw/workflows/dispatcher.py:134: error: Cannot infer type argument (requires protocol redesign)
```

**After Fix Attempts:**
```
Fixes attempted: 7 (imports, formatting, simple type fixes)
Remaining errors: 2 (complex type incompatibilities)
```

**Subagent Output:**
```
LINTING_FAILED: Type errors require manual intervention

Details:
- Linters failed: mypy
- Errors remaining: 2
- Fixes attempted: 7
- Manual intervention needed: Complex type incompatibilities in adw/core/models.py:89 and adw/workflows/dispatcher.py:134 require protocol redesign
```

## Integration Patterns

### Execute-Plan Agent

After completing implementation tasks, before committing:

```python
# Step 9: Validate Code Quality (before Step 10: Commit)
lint_result = task({
  "description": "Validate code quality",
  "prompt": f"Lint code. Arguments: adw_id={adw_id}",
  "subagent_type": "linter"
})

if "LINTING_SUCCESS" in lint_result:
  log("Code quality validated")
  proceed_to_commit()
elif "LINTING_FAILED" in lint_result:
  handle_linting_failure(lint_result)
  # Stop workflow, report to user
```

### Implementor Agent

After implementation complete:

```python
# Before outputting IMPLEMENTATION_COMPLETE
lint_result = task({
  "description": "Lint implementation",
  "prompt": f"Arguments: adw_id={adw_id}",
  "subagent_type": "linter"
})

if "LINTING_FAILED" in lint_result:
  output("IMPLEMENTATION_FAILED: Linting issues detected")
  output(lint_result)
  exit()
```

## Troubleshooting

### "LINTING_FAILED: Linter tool not found"
**Cause:** `run_linters` tool not available or worktree environment not set up

**Solution:**
```bash
# Verify tool exists
which ruff
which mypy

# Verify virtual environment active
cd /trees/abc12345/
source .venv/bin/activate
```

### "LINTING_FAILED: Permission denied writing to file"
**Cause:** File permissions issue in worktree

**Solution:**
```bash
# Check file permissions
ls -la adw/workflows/build.py

# Fix permissions if needed
chmod 644 adw/workflows/build.py
```

### Linter takes too long
**Cause:** Large codebase or slow type checking

**Solution:**
```python
# Use target_dir to lint specific directory
task({
  "prompt": "Arguments: adw_id=abc12345 target_dir=adw/workflows",
  "subagent_type": "linter"
})
```

### Auto-fixes break functionality
**Cause:** Aggressive auto-fix removed necessary code

**Solution:**
```bash
# Review changes before committing
git diff HEAD

# Revert specific auto-fix if needed
git checkout -- adw/file.py
```

## Best Practices

### For Primary Agents Calling Linter

1. **Call before commit**: Ensure code quality before creating commit
2. **Always provide adw_id**: Required for context loading
3. **Stop on failure**: Don't commit if linting fails
4. **Parse output carefully**: Check for LINTING_SUCCESS vs LINTING_FAILED signals
5. **Report fixes to user**: Communicate what was auto-fixed

### For Repository Setup

1. **Keep lint.yml current**: Update `.github/workflows/lint.yml` with all linters
2. **Configure pre-commit hooks**: Catch issues locally before pushing
3. **Set reasonable limits**: Balance strictness with productivity
4. **Document style guide**: Keep `docs/Agent/linting_guide.md` up to date

### For Workflow Design

1. **Lint early and often**: Catch issues before they compound
2. **Auto-fix first**: Let tools handle formatting and simple issues
3. **Test after fixing**: Ensure auto-fixes don't break functionality
4. **Document unfixable issues**: Help developers understand complex problems

## Performance Characteristics

| Scenario | Typical Time | Notes |
|----------|--------------|-------|
| All linters pass | 5-10 seconds | Quick validation |
| Auto-fixes only | 10-20 seconds | ruff fixes and re-runs |
| Manual fixes (5 issues) | 30-60 seconds | File reads, edits, re-validation |
| Large codebase (1000+ files) | 30-120 seconds | mypy type checking scales with size |

## References

- **Linting Guide**: `docs/Agent/linting_guide.md` - Repository linting standards
- **Original Slash Command**: `.opencode/command/lint.md` - Lint command reference
- **CI Workflow**: `.github/workflows/lint.yml` - Linter configuration
- **Execute-Plan Agent**: `.opencode/agent/execute-plan.md` - Primary caller
- **Ruff Documentation**: https://docs.astral.sh/ruff/ - Linter reference
- **Mypy Documentation**: https://mypy.readthedocs.io/ - Type checker reference

## See Also

- **Git-Commit Subagent**: `.opencode/agent/git-commit.md` - Called after linting
- **Testing Guide**: `docs/Agent/testing_guide.md` - Testing standards
- **Code Style Guide**: `docs/Agent/code_style.md` - Coding conventions
- **ADW Workflow System**: `README.md` - Complete workflow documentation
