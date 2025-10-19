# Initialize Worktree with Sparse Checkout

Create a new git worktree for an agent to work in isolation, with only the specified directory checked out.

## Variables
worktree_name: $1
target_directory: $2

## Instructions

1. Create a new git worktree in the `trees/<worktree_name>` directory with sparse checkout
2. Configure sparse checkout to only include `<target_directory>`
3. Base the worktree on the main branch
4. Copy the `.env` file from the root directory to the worktree (if it exists)
5. Create an initial commit in the worktree to establish the branch
6. Report the successful creation of the worktree

## Git Worktree Setup with Sparse Checkout

Execute these steps in order:

1. **Create the trees directory** if it doesn't exist:
   ```bash
   mkdir -p trees
   ```

2. **Check if worktree already exists**:
   - If `trees/<worktree_name>` already exists, report that it exists and stop
   - Otherwise, proceed with creation

3. **Create the git worktree without checkout**:
   ```bash
   git worktree add --no-checkout trees/<worktree_name> -b <worktree_name>
   ```

4. **Configure sparse checkout for the target directory**:
   ```bash
   cd trees/<worktree_name>
   
   # Initialize sparse checkout
   git sparse-checkout init --cone
   
   # Set sparse checkout to only include the target directory
   git sparse-checkout set <target_directory>
   
   # Now checkout the files
   git checkout
   ```

5. **Copy environment file** (if exists):
   Copy the .env from the root directory into `trees/<worktree_name>/<target_directory>/.env`

6. **Create initial commit with no changes**:
   ```bash
   git commit --allow-empty -m "Initial worktree setup for <worktree_name> with sparse checkout of <target_directory>"
   ```

## Error Handling

- If the worktree already exists, report this and exit gracefully
- If git worktree creation fails, report the error
- If sparse-checkout configuration fails, report the error
- If .env doesn't exist in root or target directory, continue without error (it's optional)

## Verification

After setup, verify the sparse checkout is working:
```bash
cd trees/<worktree_name>
ls -la  # Should only show <target_directory> directory (plus .git)
git sparse-checkout list  # Should show: <target_directory>
```

## Report

Report one of the following:
- Success: "Worktree '<worktree_name>' created successfully at trees/<worktree_name> with only <target_directory> checked out"
- Already exists: "Worktree '<worktree_name>' already exists at trees/<worktree_name>"
- Error: "Failed to create worktree: <error message>"

## Notes

- Git worktrees with sparse checkout provide double isolation:
  - **Worktree isolation**: Separate branch and working directory
  - **Sparse checkout**: Only the relevant app directory is present
- This reduces clutter and prevents accidental modifications to other apps
- The agent only sees and works with `<target_directory>`
- Full repository history is still available but only the specified directory is in the working tree
- Each worktree maintains its own sparse-checkout configuration