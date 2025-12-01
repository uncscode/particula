---
description: "Create Pull Request"
---

# Create Pull Request

Based on the `Instructions` below, take the `Variables` follow the `Run` section to create a pull request. Then follow the `Report` section to report the results of your work.

## Variables

adw_id: $ARGUMENT (8-character hexadecimal workflow ID, e.g., "abc12345")

## Instructions

### 1. Load Workflow Data
Use the `adw_spec` tool to load all workflow data from state:

```
adw_spec({
  command: "read",
  adw_id: "{adw_id}"
})
```

**IMPORTANT**: The `adw_id` parameter must be the 8-character hexadecimal workflow ID string, NOT a file path.

The spec contains the complete workflow state including:
- `branch_name`: The git branch to push
- `issue`: Complete issue object with number, title, body, labels
- `spec_content`: Implementation plan/specification
- `workflow_type`: Type of workflow (complete, patch, document, etc.)

### 2. Generate Pull Request
- IMPORTANT: Code must be linted before creating the PR. The ADW workflow automatically runs linting checks before committing.
- Generate a pull request title in the format: `<issue_type>: #<issue_number> - <issue_title>`
- The PR body should include:
  - A summary section with the issue context
  - Link to the implementation plan (spec_content) if available
  - Reference to the issue (Closes #<issue_number>)
  - ADW tracking ID
  - A checklist of what was done
  - A summary of key changes made
- Extract issue number, type, and title from the issue object in spec
- Examples of PR titles:
  - `feature: #123 - Add user authentication`
  - `chore: #789 - Update dependencies`

## Run

1. Run `git diff origin/main...HEAD --stat` to see a summary of changed files
2. Run `git log origin/main..HEAD --oneline` to see the commits that will be included
3. Run `git diff origin/main...HEAD --name-only` to get a list of changed files
4. Run `git push -u origin <branch_name>` to push the branch
5. Set GH_TOKEN environment variable from GITHUB_PAT if available, then run `gh pr create --title "<pr_title>" --body "<pr_body>" --base main --web=false` to create the PR
6. Capture the PR URL from the output

## Report

Return ONLY the PR URL that was created (no other text)