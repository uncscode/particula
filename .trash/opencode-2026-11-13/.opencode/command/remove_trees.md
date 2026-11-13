---
description: "List and remove git worktrees"
---

# Remove Git Worktrees

List active git worktrees in the `trees/` directory and optionally remove them.

## Purpose

Clean up git worktrees created by ADW workflows. Each workflow runs in an isolated
worktree under `trees/<adw_id>/`. Use this command to view and remove completed
or abandoned worktrees.

## Instructions

1. **List active worktrees**
   - Use the `list` tool to enumerate directories in `trees/`
   - For each worktree, use `git_operations` to check status:
     ```
     git_operations({ command: "status", worktree_path: "trees/<adw_id>" })
     ```
   - Display each worktree with its ADW ID and any uncommitted changes

2. **Ask for confirmation**
   - Show the list of worktrees found in `trees/`
   - Ask the user which worktrees to remove (all, specific IDs, or none)

3. **Remove selected worktrees**
   - For each worktree to remove:
     - Use `git_operations` to restore any staged changes first:
       ```
       git_operations({ command: "restore", staged: true, worktree_path: "trees/<adw_id>" })
       ```
     - Use the `move` tool with `trash: true` to move the worktree directory:
       ```
       move({ source: "trees/<adw_id>", destination: "", trash: true })
       ```
     - Use the `move` tool to trash the corresponding state directory:
       ```
       move({ source: "agents/<adw_id>", destination: "", trash: true })
       ```
   - Report success or failure for each removal

## Tool Examples

### List worktree status
```json
{
  "command": "status",
  "worktree_path": "trees/abc12345",
  "porcelain": true
}
```

### Check for uncommitted changes
```json
{
  "command": "diff",
  "worktree_path": "trees/abc12345",
  "stat": true
}
```

### Restore staged changes before removal
```json
{
  "command": "restore",
  "staged": true,
  "worktree_path": "trees/abc12345"
}
```

## Example Output

```
Found 3 worktrees in trees/:

  1. abc12345 - clean
  2. def67890 - 2 uncommitted changes
  3. ghi11111 - clean

Which worktrees would you like to remove?
- Enter "all" to remove all worktrees
- Enter specific IDs (e.g., "abc12345 def67890")
- Enter "none" to cancel
```

## Notes

- Worktrees are moved to `.trash/` for safe removal with audit trail
- Associated state in `agents/<adw_id>/` is also trashed
- Use `adw status` to check for active workflows before cleanup
- Trashed directories can be restored by moving them back out of `.trash/`
