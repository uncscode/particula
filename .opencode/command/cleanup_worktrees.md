---
description: "Cleanup ADW Worktrees"
---

# Cleanup ADW Worktrees

Clean up isolated ADW worktrees and their associated resources.

## Variables

action: $ARGUMENT (all|specific|list)
adw_id: $ARGUMENT (optional, required if action is "specific")

## Instructions

Manage git worktrees created by isolated ADW workflows:
- If action is "list": Show all worktrees under trees/ directory
- If action is "specific": Remove the specific worktree for the given adw_id
- If action is "all": Remove all worktrees under trees/ directory

## Run

Based on the action:

### List worktrees
If action is "list":
- Run `git worktree list | grep "trees/"` to show isolated worktrees
- List the contents of the trees/ directory with sizes

### Remove specific worktree
If action is "specific" and adw_id is provided:
- Check if trees/{adw_id} exists
- Run `git worktree remove trees/{adw_id}` to remove it
- Report success or any errors

### Remove all worktrees
If action is "all":
- First list all worktrees that will be removed
- For each worktree under trees/, run `git worktree remove`
- Clean up any remaining directories under trees/
- Run `git worktree prune` to clean up any stale entries

## Report

Report the results of the cleanup operation:
- Number of worktrees removed
- Any errors encountered
- Current status after cleanup