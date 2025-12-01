---
description: "Install Worktree"
---

# Install Worktree

This command sets up an isolated worktree environment for Python development.

## Parameters
- Worktree path: {0}

## Steps

1. **Navigate to worktree directory**
   ```bash
   cd {0}
   ```

2. **Install Python dependencies**
   ```bash
   uv pip install -e ".[dev]"
   ```

## Error Handling
- If uv is not available, suggest installing it first
- Report any installation errors clearly

## Report
- Confirm Python dependencies installed successfully
- Show worktree path
- Note any errors encountered