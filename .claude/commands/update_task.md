# Update Task

Update the task list with the result of agent work.

## Variables
adw_id: $1
worktree_name: $2
task_description: $3
status: $4
commit_hash: $5
error_message: $ARGUMENT

## Instructions

1. Read the current `tasks.md` file
2. Find the task matching the provided `adw_id`, `worktree_name` and `task_description`
3. Update the task status based on the result:
   - If `status` is "success", update to `[✅ <commit_hash>, <adw_id>] <description>`
   - If `status` is "failed", update to `[❌, <adw_id>] <description> // Failed: <error_message>`
4. Preserve all other content and formatting in the file
5. Write the updated content back to `tasks.md`

## Task Update Rules

1. **Success Status**:
   - Format: `[✅ <commit_hash>, <adw_id>] <description>`
   - Must include the git commit hash
   - Example: `[✅ abc123def, adw_12345678] Implement user authentication`

2. **Failed Status**:
   - Format: `[❌, <adw_id>] <description> // Failed: <error_message>`
   - Include the error message after `// Failed:` if provided
   - If no error_message is provided, use format: `[❌, <adw_id>] <description>`
   - Example: `[❌, adw_12345678] Implement user authentication // Failed: Module 'auth' not found`

3. **Matching Tasks**:
   - Find tasks with status `[🟡, <adw_id>]` in the specified worktree
   - The adw_id must match exactly
   - Only update the first matching task

## Error Handling

- If no matching task is found, report an error
- If the task is not in progress status `[🟡]`, report an error
- Ensure the file write is atomic to prevent corruption

## File Format Preservation

- Maintain exact formatting of the markdown file
- Preserve whitespace and line breaks
- Keep task descriptions unchanged
- Retain any tags in the format `{tag1, tag2}`

## Example Update

Before:
```
## Git Worktree feature-auth
[🟡, adw_12345678] Implement user authentication
```

After (success):
```
## Git Worktree feature-auth
[✅ abc123def, adw_12345678] Implement user authentication
```

After (failure with error message):
```
## Git Worktree feature-auth
[❌, adw_12345678] Implement user authentication // Failed: Tests failed with 3 errors
```

After (failure without error message):
```
## Git Worktree feature-auth
[❌, adw_12345678] Implement user authentication
```

## Report

After updating the task, report:
- The task that was updated
- The new status
- Success or failure of the update operation