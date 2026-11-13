# ADW Build Agent Family - Usage Guide

## Overview

The ADW Build agent family provides comprehensive implementation orchestration with per-task validation. This family replaces the monolithic `execute-plan` approach with a modular, validation-focused architecture.

For auto workflows, the family now splits original implementation from trailing review-fix work:
- `adw-build` executes the original plan from `spec_content`
- `adw-build-fix` executes the review-driven fix plan from `fix_spec_content`

## Agent Family Structure

```
adw-build (primary)
  │
  ├── adw-build-tests (subagent)      # Test validation & writing
  ├── adw-build-docstrings (subagent) # Docstrings & linting
  ├── adw-build-validate (subagent)   # Spec compliance check
  └── adw-commit (subagent)           # Commit with pre-commit hooks
```

## When to Use

### Use `adw-build` When:
- Implementing features from a spec/plan
- You need comprehensive validation (tests, docstrings, linting)
- Quality assurance is critical
- Working in an automated CI/CD pipeline

### Use `adw-build-fix` When:
- Running the trailing fix pass in `complete-auto` or `patch-auto`
- Review has already persisted `request_fix`, `review_feedback`, and `review_findings`
- A dedicated fix plan exists in `fix_spec_content`

### Use Individual Subagents When:
- Running targeted validation on specific files
- Debugging test or docstring issues
- Manual intervention in a partially complete workflow

## Agents

### 1. adw-build (Primary)

**Purpose:** Orchestrate implementation with per-task validation.

**Mode:** `primary`

**Invocation:**
```bash
adw workflow run build <issue-number> --adw-id <id>
```

**What it does:**
1. Reads implementation plan from `spec_content`
2. Converts plan steps to todos
3. Implements each task with per-task validation
4. Calls subagents for tests, docstrings, and final validation
5. Commits changes with pre-commit handling

**Boundary:** This agent no longer owns auto-workflow fix gating. Trailing review-fix work is handled by `adw-build-fix`.

---

### 1b. adw-build-fix (Primary)

**Purpose:** Execute the dedicated review-fix plan for auto workflows.

**Mode:** `primary`

**Invocation:**
```bash
adw workflow run patch-auto <issue-number> --adw-id <id>
adw workflow run complete-auto <issue-number> --adw-id <id>
```

**What it does:**
1. Reads `fix_spec_content`
2. Reads persisted review context from `review_feedback` and `review_findings`
3. Verifies `current_step == "Fix"` and `request_fix == True`
4. Marks `fix_completed=True` with explicit field writes
5. Implements the fix plan and runs fast validation

**Output Signals:**
- `ADW_BUILD_FIX_COMPLETE` - Fix tasks complete
- `ADW_BUILD_FIX_FAILED` - Could not complete fix implementation

**Output Signals:**
- `ADW_BUILD_COMPLETE` - All tasks complete, committed
- `ADW_BUILD_FAILED` - Could not complete implementation

---

### 2. adw-build-tests (Subagent)

**Purpose:** Validate test coverage and write missing tests.

**Mode:** `subagent`

**Invocation:**
```python
task({
  "description": "Validate tests for changed files",
  "prompt": "Validate tests.\n\nArguments: adw_id=abc12345 file=adw/utils/parser.py\n\nContext: Added input validation",
  "subagent_type": "adw-build-tests"
})
```

**Scope Options:**
- `file=<path>` - Single file
- `module=<path>` - Module directory
- `dir=<path>` - Directory (recursive)
- `files=<path1,path2>` - Comma-separated list

**What it does:**
1. Identifies all functions needing tests
2. Checks existing test coverage
3. Writes missing tests (public AND private functions)
4. Runs tests and fixes failures (3 retries)
5. Enforces 80% coverage threshold

**Output Signals:**
- `ADW_BUILD_TESTS_SUCCESS` - Tests validated and passing
- `ADW_BUILD_TESTS_FAILED` - Could not achieve passing tests

---

### 3. adw-build-docstrings (Subagent)

**Purpose:** Add/update docstrings and run linting.

**Mode:** `subagent`

**Invocation:**
```python
task({
  "description": "Add docstrings and run linting",
  "prompt": "Add docstrings and lint.\n\nArguments: adw_id=abc12345 file=adw/utils/parser.py\n\nContext: Added input validation",
  "subagent_type": "adw-build-docstrings"
})
```

**Scope Options:**
- `file=<path>` - Single file
- `module=<path>` - Module directory  
- `dir=<path>` - Directory (recursive)
- `files=<path1,path2>` - Comma-separated list

**What it does:**
1. Identifies missing/outdated docstrings
2. Adds Google-style docstrings to all functions/classes
3. Runs ruff check, ruff format, mypy
4. Auto-fixes linting issues (3 retries)

**Output Signals:**
- `ADW_BUILD_DOCSTRINGS_SUCCESS` - Docstrings complete, linting passes
- `ADW_BUILD_DOCSTRINGS_FAILED` - Could not complete

---

### 4. adw-build-validate (Subagent)

**Purpose:** Validate implementation against spec and issue requirements.

**Mode:** `subagent` (READ-ONLY)

**Invocation:**
```python
task({
  "description": "Validate implementation against spec",
  "prompt": "Validate implementation.\n\nArguments: adw_id=abc12345",
  "subagent_type": "adw-build-validate"
})
```

**What it does:**
1. Reads spec_content and issue requirements
2. Compares actual changes against requirements
3. Runs full test suite (catches unintended failures)
4. Reports structured list of gaps

**Output Signals:**
- `ADW_BUILD_VALIDATE_SUCCESS` - All requirements met
- `ADW_BUILD_VALIDATE_INCOMPLETE` - Gaps found (with actionable list)

**Note:** This agent does NOT fix issues - it reports them for `adw-build` to fix.

---

### 5. adw-commit (Subagent)

**Purpose:** Commit changes with pre-commit hook handling.

**Mode:** `subagent`

**Invocation:**
```python
task({
  "description": "Commit implementation changes",
  "prompt": "Commit changes.\n\nArguments: adw_id=abc12345",
  "subagent_type": "adw-commit"
})
```

**What it does:**
1. Analyzes git diff for commit message
2. Generates conventional commit message
3. Stages all changes
4. Commits with pre-commit hooks
5. Fixes pre-commit failures (3 retries)

**Output Signals:**
- `ADW_COMMIT_SUCCESS` - Commit created
- `ADW_COMMIT_SKIPPED` - No changes to commit
- `ADW_COMMIT_FAILED` - Could not commit

---

## Execution Flow

### Per-Task Validation

```
For each implementation task:
  ┌──────────────────────────────────────┐
  │ 1. Implement code change             │
  └──────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────┐
  │ 2. adw-build-tests (file scope)      │
  │    - Write missing tests             │
  │    - Run tests, fix failures         │
  │    - Ensure 80% coverage             │
  └──────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────┐
  │ 3. adw-build-docstrings (file scope) │
  │    - Add missing docstrings          │
  │    - Run linters, fix issues         │
  └──────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────┐
  │ 4. Mark task complete                │
  └──────────────────────────────────────┘
```

### Final Validation

```
After all tasks complete:
  ┌──────────────────────────────────────┐
  │ adw-build-validate (whole package)   │
  │    - Compare spec vs actual          │
  │    - Run full test suite             │
  │    - Check acceptance criteria       │
  └──────────────────────────────────────┘
           ↓                    ↓
     [SUCCESS]            [INCOMPLETE]
           ↓                    ↓
  ┌────────────────┐   ┌────────────────┐
  │ adw-commit     │   │ Fix gaps       │
  │                │   │ Re-validate    │
  └────────────────┘   └────────────────┘
```

### Auto Review-Fix Path

```text
Review
  -> review-state-writer persists request_fix/review_feedback/review_findings
  -> review-fix-planner writes fix_spec_content
  -> adw-build-fix executes fix_spec_content
  -> adw-validate checks against fix_spec_content during Fix-* steps
```

## Quality Standards

### Test Requirements
- All public functions: ≥1 test
- All private functions: ≥1 test
- Changed code coverage: ≥80%
- Meaningful assertions (not just `assert True`)

### Docstring Requirements
- Module-level docstrings
- All functions have docstrings
- Google-style format (Args, Returns, Raises)
- Line length ≤100 characters

### Linting Requirements
- Ruff check: No errors
- Ruff format: Properly formatted
- Mypy: No type errors

## Usage Examples

### Full Workflow
```bash
# Run complete build workflow
adw workflow run build 123 --adw-id abc12345
```

### Targeted Test Validation
```python
# Validate tests for a specific file
task({
  "description": "Validate tests",
  "prompt": "Arguments: adw_id=abc12345 file=adw/utils/parser.py",
  "subagent_type": "adw-build-tests"
})
```

### Targeted Docstring/Lint Check
```python
# Add docstrings and lint a module
task({
  "description": "Docstrings and lint",
  "prompt": "Arguments: adw_id=abc12345 module=adw/utils",
  "subagent_type": "adw-build-docstrings"
})
```

### Validate Against Spec
```python
# Check if implementation matches spec
task({
  "description": "Validate against spec",
  "prompt": "Arguments: adw_id=abc12345",
  "subagent_type": "adw-build-validate"
})
```

## Comparison with execute-plan

| Feature | execute-plan | adw-build |
|---------|--------------|-----------|
| Per-task test validation | ❌ | ✅ |
| Per-task docstring validation | ❌ | ✅ |
| Writes missing tests | ❌ | ✅ |
| Writes missing docstrings | ❌ (vague instruction) | ✅ |
| Spec compliance check | ❌ | ✅ |
| Pre-commit hook handling | ✅ (via git-commit) | ✅ (via adw-commit) |
| Scoped validation | ❌ | ✅ (file/module/dir) |
| Coverage enforcement | ❌ | ✅ (80% threshold) |

## Troubleshooting

### Tests Keep Failing
1. Check if implementation bug vs test bug
2. Review test output for specific assertion failures
3. Ensure test follows repository patterns

### Docstrings Not Being Added
1. Verify file is not in `tests/` directory (excluded)
2. Check if function already has docstring (update needed, not add)
3. Review linting errors that may block docstring formatting

### Validation Reports Gaps But They're Done
1. Ensure changes are saved (not just in editor)
2. Check if correct file paths in spec match actual files
3. Verify tests run against correct scope

### Pre-commit Hooks Keep Failing
1. Check hook output for specific errors
2. Run linters manually to see issues
3. Some hooks auto-fix; re-stage files

## See Also

- [Testing Guide](../testing_guide.md)
- [Docstring Guide](../docstring_guide.md)
- [Linting Guide](../linting_guide.md)
- [Commit Conventions](../commit_conventions.md)
- [Code Style](../code_style.md)
