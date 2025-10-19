# Process Tasks

Analyze the current task list and identify tasks that are ready to be picked up by agents.

## Instructions

1. Read the `tasks.md` file to understand the current state of all tasks
2. Identify tasks that are eligible for pickup:
   - Tasks with status `[]` (not started) are always eligible
   - IMPORTANT: Tasks with status `[⏰]` (blocked) are eligible ONLY if ALL tasks above them in the same worktree have status `[✅]` (success)
3. Group eligible tasks by their worktree
4. Return a JSON array with the structure specified below
5. DO NOT modify the task list - only analyze and return eligible tasks

## Task Status Guide

- `[]` - Not started (ready for pickup)
- `[⏰]` - Not started and blocked (can only start when all tasks above in the worktree are successful)
- `[🟡]` - Work in progress (skip these)
- `[✅]` - Success (completed)
- `[❌]` - Failed (terminal state)

## Rules

1. Only include worktrees that have eligible tasks
2. IMPORTANT: For blocked tasks `[⏰]`, check that ALL tasks above them in the same worktree are successful aka `[✅]`
3. Extract tags from task descriptions - tags are in the format `{tag1, tag2}`
4. Return an empty array `[]` if no tasks are eligible
5. Tasks are processed top to bottom within each worktree

## Examples

### Example 1: Task in progress blocks dependent task
Given this task list:
```
## Git Worktree feature-auth
[✅] Task 1
[🟡] Task 2
[] Task 3 {api, auth}
[⏰] Task 4
```

The blocked task (Task 4) is NOT eligible because Task 2 is still in progress.
Only Task 3 would be returned as eligible.

### Example 2: Failed task prevents blocked task from running
Given this task list:
```
## Git Worktree create-topic-filter
[❌, 17d16d17] Generate filtered dataset at data/tweets_tech_topics.csv containing only technology and entertainment topics from tweets_v1.csv
[⏰] Add 30 new tweets about sports and recreation to expand topic diversity in tweets_v1.csv
```

The blocked task (Add 30 new tweets) will NOT be eligible for pickup because the task above it failed. 
Blocked tasks require ALL preceding tasks in the same worktree to be successful (`[✅]`) before they can run.
No tasks would be returned as eligible from this worktree.

## Task

Read `tasks.md` and return eligible tasks in the specified JSON format.


## Output Format

IMPORTANT: Return a JSON array with this structure:

```json
[
  {
    "worktree_name": "worktree_name",
    "tasks_to_start": [
      {
        "description": "task description",
        "tags": ["tag1", "tag2"]
      }
    ]
  }
]
```