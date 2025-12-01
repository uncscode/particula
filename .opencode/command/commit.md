---
description: "Generate Git Commit"
---

# Generate Git Commit

Based on the `Instructions` below, take the `Variables` follow the `Run` section to create a git commit with a properly formatted message. Then follow the `Report` section to report the results of your work.

## Variables

adw_id: $ARGUMENT (8-character hexadecimal workflow ID, e.g., "abc12345")

## Instructions

### 1. Load Workflow Data

Use the `adw_spec` tool to load all workflow data from state if adw_id is provided:

**If adw_id is provided:**
```
adw_spec({
  command: "read",
  adw_id: "{adw_id}"
})
```

**If adw_id is not provided:** Skip to step 2 and generate commit based only on git diff.

**IMPORTANT**: The `adw_id` parameter must be the 8-character hexadecimal workflow ID string, NOT a file path.

The spec contains the complete workflow state including:
- `issue`: Complete issue object with number, title, body, labels
- `workflow_type`: Type of workflow (complete, patch, document) - determines prefix
- `spec_content`: Implementation plan to understand what was done

### 2. Generate Commit Message
- Generate a concise commit message in the format: `<workflow_type>: <commit message>`
- The `<commit message>` should be:
  - Present tense (e.g., "add", "fix", "update", not "added", "fixed", "updated")
  - 50 characters or less
  - Descriptive of the actual changes made (use `git diff HEAD` to see changes)
  - No period at the end
- Workflow type prefixes:
  - `feature:` for new features (workflow_type = "complete" or "feature")
  - `fix:` for bug fixes (workflow_type = "patch" or issue has "bug" label)
  - `docs:` for documentation (workflow_type = "document")
  - `chore:` for maintenance (issue has "chore" label)
- Examples:
  - `feature: add user authentication module`
  - `fix: resolve login validation error`
  - `docs: expand API documentation`
- Extract context from the issue object and git diff to make the commit message relevant
- Don't include any 'Generated with...' or 'Authored by...' in the commit message. Focus purely on the changes made.

## Run

1. Run `git diff HEAD` to understand what changes have been made
2. Run `git add -A` to stage all changes
3. Run `git commit -m "<generated_commit_message>"` to create the commit

### Error Handling

- If the pre-commit checks fail, then fix or address the issues and stage the changes again before committing.
- Then re-run the commit command.

## Report

Return ONLY the commit message that was used (no other text)