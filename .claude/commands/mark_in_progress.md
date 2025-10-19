# Mark Task In Progress

Mark tasks as in-progress in the task list file.

## Variables
task_file_path: $1
worktree_name: $2
task_description: $3
adw_id: $4

## Instructions

1. Read the task file at `task_file_path`
2. Locate the worktree section matching `worktree_name`
3. Find the task matching `task_description` that has status `[]` or `[⏰]`
4. Update the task status to `[🟡, <adw_id>]`
5. Preserve all tags and formatting
6. Write the updated content back to the file

## Task Format

Before:
```
[] <task_description> {optional_tags}
```
or
```
[⏰] <task_description> {optional_tags}
```

After:
```
[🟡, <adw_id>] <task_description> {optional_tags}
```

## Error Handling

- If the task file doesn't exist, report error
- If the worktree section is not found, report error
- If the task is not found or already in progress, report error
- Ensure atomic file write to prevent corruption

## Report

Report the following:
- Task that was updated
- New status with ADW ID
- Success or failure of the operation