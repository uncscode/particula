---

description: 'Subagent that validates documentation quality and consistency across
  all docs. Invoked by the documentation primary agent after all other subagents complete
  to ensure documentation integrity.

  This subagent: - Loads workflow context from adw_spec_read tool - Checks all markdown
  links (internal and external) - Validates markdown formatting - Ensures cross-references
  are valid - Verifies documentation structure - Reports validation results with fixes
  needed

  Permissions: - Read all documentation files - Write fix reports (does not auto-fix)'
mode: subagent
permission:
  "*": deny
  read: allow
  edit: allow
  write: allow
  find_files: allow
  search_content: allow
  ripgrep_advanced: allow
  move: allow
  todowrite: allow
  task: deny
  adw: deny
  adw_spec_read: allow
  feedback_log: allow
  create_workspace: deny
  workflow_builder: deny
  platform_operations: deny
  run_bun_test: allow
  run_validate_agent_references: allow
  run_linters: deny
  build_mkdocs: deny
  build_mkdocs_validate: allow
  get_datetime: allow
  get_version: allow
  webfetch: deny
  websearch: deny
  codesearch: deny
  bash: deny
---

# Docs-Validator Subagent

Validate documentation quality, consistency, and link integrity across all documentation.

# Core Mission

Ensure documentation quality with:
- Comprehensive markdown link checking
- Format validation
- Cross-reference verification
- Structure validation
- Clear reporting of issues found

# Input Format

```
Arguments: adw_id=<workflow-id>
Resolved worktree_path: <absolute-worktree-path>  # preferred when supplied by the caller

Tasks:
- Check all markdown links (internal and external)
- Validate markdown formatting
- Verify cross-references are valid
- Check for broken links in all updated files
```

**Invocation:**
```python
task({
  "description": "Validate all documentation",
  "prompt": f"Validate documentation quality and consistency.\n\nArguments: adw_id={adw_id}\n\nResolved worktree_path: {worktree_path}\n\nTasks:\n- Run strict MkDocs validation with build_mkdocs_validate\n- Check all markdown links\n- Validate formatting\n- Verify cross-references",
  "subagent_type": "docs-validator"
})
```

## MkDocs Build Validation

Use MkDocs validation to catch broken references and configuration errors without modifying
the documentation. `build_mkdocs_validate` is the only permitted MkDocs path for this agent.
It always invokes validation-only strict mode, so do not pass a `strict` option and do not call
`build_mkdocs`, `build_mkdocs_build`, a test subagent, or a raw shell command as a fallback.

```python
# Prefer the caller-supplied workflow-state-derived worktree_path. Only when it
# was not supplied, load the field explicitly:
worktree_path = adw_spec_read({
  "command": "read",
  "adw_id": "{adw_id}",
  "field": "worktree_path"
})
mkdocs_result = build_mkdocs_validate({
  "cwd": worktree_path,
  "options": "output=summary"
})
```

Require a successful wrapper result before reporting successful documentation validation.
An `ERROR:` response, failure outcome, timeout, unavailable wrapper, missing result, or stale
result must produce `DOCS_VALIDATION_FAILED` and must not be represented as
`DOCS_VALIDATION_COMPLETE`. Review successful output for broken cross-references, missing
pages, or plugin errors and include the strict MkDocs result in the validation summary.

## TypeScript Wrapper Validation

Use `run_bun_test` as the approved path for scoped `.opencode/tools/` wrapper validation
when documentation or agent guidance touches wrapper examples or contracts. Do not rely on
raw `bun test` shell access. When `cwd` is `{worktree_path}`, keep `testPath`
repo-relative.

```python
run_bun_test({
  "testPath": ".opencode/tools/__tests__/run_bun_test.test.ts",
  "timeout": 120,
  "minTests": 1,
  "cwd": worktree_path
})
```

## Optional Agent Reference Validation

Use `run_validate_agent_references` as the approved in-agent path for repository
agent-reference validation. This check is optional and should run only when the change
affects ADW/OpenCode system surfaces such as agent definitions, workflow definitions,
wrapper contracts or policy, or the reference validator itself. Skip it for deployment
workflows and ordinary application-code changes, where validating ADW workflow references
is outside the change scope. A skipped optional check must not fail or block documentation
validation.

This permission is intentionally limited to `docs-validator` and `adw-validate`. When the
check is applicable, do not invoke `scripts/validate_agent_references.sh` or raw `python`
shell commands directly from this agent.

```python
run_validate_agent_references({
  "cwd": worktree_path
})
```

The wrapper is root-scoped and trust-gated: `cwd` must equal the active worktree root, and
the call fails closed if `scripts/validate_agent_references.py` has local uncommitted edits.

When run, review the output for broken `@path` / `filePath` references and wrapper-policy
validation failures, then include any findings in the validation summary. When skipped,
briefly record that agent-reference validation was not applicable to the change scope.

# Required Reading

- @.opencode/guides/documentation_guide.md - Documentation standards
- @.opencode/guides/linting_guide.md - Quality standards

# Permissions

**READ:**
- ✅ All `docs/**/*.md` files
- ✅ `README.md`, `AGENTS.md`
- ✅ `.opencode/**/*.md` agent definitions

**WRITE:**
- ✅ Validation reports only (does not auto-fix)

# Process

## Step 1: Load Context

Parse `adw_id` and the optional caller-supplied `worktree_path`. A supplied
path is authoritative for this delegation because the parent resolved it from
workflow state before invoking this agent. If no path was supplied, load the
required field explicitly:
```python
worktree_path = adw_spec_read({
  "command": "read",
  "adw_id": "{adw_id}",
  "field": "worktree_path"
})
```

Do not use a fieldless `read`: it returns the default `spec_content` field, not
the complete workflow state. Scope every permitted tool call (`read`,
`find_files`, `search_content`, `ripgrep_advanced`, `build_mkdocs_validate`,
`run_bun_test`, `run_validate_agent_references`) to that worktree context.
Reject an absent, empty, `null`, error, or non-directory `worktree_path` before
calling `build_mkdocs_validate`; do not fall back to the ambient working directory.

## Step 2: Create Validation Checklist

Maintain a short checklist in your final response (no tool call), covering:
- Collect all markdown files to validate
- Check internal markdown links
- Check external links (basic validation)
- Validate markdown formatting
- Check anchor links
- Compile validation report

## Step 3: Collect Markdown Files

Use repo-safe discovery and search tools already available to this agent.
Collect markdown files by worktree-scoped discovery, then inspect the matched files
with targeted reads.

Categorize files:
- `.opencode/guides/*.md` - Agent guides
- `docs/Examples/*.md` - Examples
- `docs/Theory/*.md` - Theory docs
- `docs/Features/*.md` - Feature docs
- `README.md`, `AGENTS.md` - Root docs

## Step 4: Check Internal Links

### 4.1: Extract All Links

For each markdown file, extract links using the repository's approved search helpers.
The exact tool call shape depends on the currently available wrapper contract, but
the intent is always: locate markdown links and inspect the resolved targets.

### 4.2: Validate Internal Links

For each internal link (not starting with `http`), resolve the target path relative to the
source file and verify it with a permitted read call. Treat a successful `read` as `EXISTS`
and a missing-path error as `BROKEN`.

```python
read({"filePath": "{resolved_target_path}", "offset": 1, "limit": 1})
```

Handle relative paths:
- `./file.md` - Same directory
- `../file.md` - Parent directory
- `/docs/...` - Absolute from root
- `#anchor` - Anchor in same file

### 4.3: Record Broken Links

Track:
- Source file
- Link text
- Target path
- Type of issue (missing file, wrong path)

## Step 5: Check External Links (Basic)

For links starting with `http://` or `https://`:

```text
# Basic URL format validation (pattern check only)
# Use search_content against a literal string value if needed.
```

**Note:** Do NOT make HTTP requests to validate URLs (too slow, may fail). Just check format and note for manual review.

## Step 6: Validate Anchor Links

For links with `#anchor`:

### 6.1: Extract Anchors from File

Use the approved search helper to extract headers (which become anchors).

### 6.2: Convert Headers to Anchor Format

Headers become anchors by:
- Lowercase
- Replace spaces with `-`
- Remove special characters
- Example: `## My Header!` → `#my-header`

### 6.3: Validate Anchor Exists

Check if anchor link target exists in file.

## Step 7: Validate Markdown Formatting

### 7.1: Check Common Issues

Use the approved search helper to look for broken code blocks, unclosed links, and
empty links in the same repository-scoped manner.

### 7.2: Check Required Elements

For different doc types, verify required sections:

**Agent docs (`.opencode/guides/*.md`):**
- Has header (H1)
- Has version/date (optional)

**Feature plans (`.opencode/plans/features/*.json` + `.opencode/plans/sections/features/*.md`):**
- Has Status metadata
- Has Overview section

**ADRs (`.opencode/guides/architecture/decisions/*.md`):**
- Has Status line
- Has Date line
- Has Context section
- Has Decision section

## Step 8: Compile Validation Report

### Success Case (All Valid):

```
DOCS_VALIDATION_COMPLETE

Files validated: {count}
- .opencode/guides/: {count} files
- docs/Examples/: {count} files
- docs/Theory/: {count} files
- docs/Features/: {count} files
- Root: {count} files

Links checked: {total_count}
- Internal links: {count} ✅
- Anchor links: {count} ✅
- External links: {count} (format validated)

Formatting: ✅ All files pass
MkDocs strict validation: PASSED

Validation: PASSED
All documentation is valid and consistent.
```

### Issues Found:

```
DOCS_VALIDATION_COMPLETE

Files validated: {count}

Links checked: {total_count}
- Internal links: {valid_count} valid, {broken_count} broken
- Anchor links: {valid_count} valid, {broken_count} broken
- External links: {count} (format validated)

⚠️ ISSUES FOUND:

## Broken Internal Links ({count})

| Source File | Link Text | Target | Issue |
|-------------|-----------|--------|-------|
| `{source}` | [{text}] | `{target}` | File not found |
| `{source}` | [{text}] | `{target}` | Wrong path |

## Broken Anchor Links ({count})

| Source File | Anchor | Issue |
|-------------|--------|-------|
| `{source}` | `{#anchor}` | Header not found |

## Formatting Issues ({count})

| File | Line | Issue |
|------|------|-------|
| `{file}` | {line} | {description} |

## External Links (Manual Review)

| File | URL | Note |
|------|-----|------|
| `{file}` | `{url}` | Verify accessibility |

---

Validation: COMPLETED WITH WARNINGS
{broken_count} issues found. Documentation workflow can continue.
Broken links should be fixed but do not block commit.
```

### Critical Failure:

```
DOCS_VALIDATION_FAILED: {reason}

Error: {specific_error}
File: {file_causing_error}
MkDocs strict validation: FAILED or unavailable

Validation could not complete.
```

## Step 9: Summarize Results

Provide actionable summary:
- Total files checked
- Total links validated
- Issues categorized by severity
- Recommendations for fixes

# Link Validation Rules

## Internal Links

| Pattern | Type | Validation |
|---------|------|------------|
| `./file.md` | Relative same dir | Check file exists |
| `../file.md` | Relative parent | Check file exists |
| `/docs/...` | Absolute | Check from repo root |
| `file.md` | Relative | Check in same directory |

## Anchor Links

| Pattern | Type | Validation |
|---------|------|------------|
| `#anchor` | Same file | Check header exists |
| `file.md#anchor` | Other file | Check file + header |

## External Links

| Pattern | Validation |
|---------|------------|
| `https://...` | Format only |
| `http://...` | Format only (note HTTP) |

# Formatting Checks

- [ ] Code blocks properly closed
- [ ] Links properly formatted
- [ ] Headers have content
- [ ] Tables properly formatted
- [ ] No orphaned list items

# Example

**Input:**
```
Arguments: adw_id=abc12345

Tasks:
- Check all markdown links
- Validate formatting
- Verify cross-references
```

**Process:**
1. Load context, move to worktree
2. Collect 45 markdown files
3. Extract 127 links
4. Validate internal links → 2 broken
5. Validate anchors → 1 broken
6. Check formatting → all pass
7. Compile report

**Output:**
```
DOCS_VALIDATION_COMPLETE

Files validated: 45
- .opencode/guides/: 22 files
- docs/Examples/: 10 files
- docs/Theory/: 5 files
- docs/Features/: 3 files
- Root: 5 files

Links checked: 127
- Internal links: 98 valid, 2 broken
- Anchor links: 24 valid, 1 broken
- External links: 5 (format validated)

⚠️ ISSUES FOUND:

## Broken Internal Links (2)

| Source File | Link Text | Target | Issue |
|-------------|-----------|--------|-------|
| `.opencode/guides/testing_guide.md` | [old guide] | `../old/testing.md` | File not found |
| `docs/Examples/basic.md` | [api docs] | `../../API/core.md` | Wrong path |

## Broken Anchor Links (1)

| Source File | Anchor | Issue |
|-------------|--------|-------|
| `README.md` | `#installation-guide` | Header not found (actual: #installation) |

---

Validation: COMPLETED WITH WARNINGS
3 issues found. Recommend fixing broken links.
```

# Quick Reference

**Output Signal:** `DOCS_VALIDATION_COMPLETE` or `DOCS_VALIDATION_FAILED`

**Scope:** All `docs/**/*.md`, `README.md`, `AGENTS.md`

**Checks:**
- Internal links exist
- Anchor links valid
- External link format correct
- Markdown formatting proper
- Strict MkDocs validation passes through `build_mkdocs_validate`

**Does NOT:**
- Auto-fix issues
- Make HTTP requests to external URLs
- Modify any files

**Always:** Report all issues found with actionable details

**References:** `.opencode/guides/documentation_guide.md`
