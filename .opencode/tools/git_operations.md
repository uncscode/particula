# Git Operations Tool Reference

Compatibility reference for the legacy broad `git_operations` wrapper.

For new and updated workflows, prefer the atomic split wrappers:
`git_diff`, `git_stage`, `git_commit`, `git_branch`, `git_merge`, and
`git_worktree`.

Use this page when maintaining compatibility call paths that still depend on
`git_operations`.

## Preferred Routing (Split Wrappers)

- Read-only inspection (`status`, `diff`, `log`, `show`) → `git_diff`
- Staging (`add`, `restore`) → `git_stage`
- Commit-only flows (`commit`) → `git_commit`
- Branch pointer moves (`checkout`, `push`, `push-force-with-lease`) → `git_branch`
- Merge/rebase lifecycle (`merge`, `rebase`, `fetch`, `sync`, `accumulate`, `abort`, `continue`, `reset`) → `git_merge`
- Worktree lifecycle (`worktree-list`, `worktree-prune`, `worktree-remove`) → `git_worktree`

## Commands

| Command               | Required params              | Description                              |
|-----------------------|------------------------------|------------------------------------------|
| `commit`              | `summary`                    | Create commit with message               |
| `push`                | `branch`                     | Push branch to origin                    |
| `checkout`            | `branch`                     | Checkout or create branch                |
| `status`              | —                            | Show git status                          |
| `diff`                | —                            | Show git diff                            |
| `log`                 | —                            | Show git log                             |
| `show`                | `ref`                        | Show commit or file at ref               |
| `add`                 | `stage_all` or `files`       | Stage changes                            |
| `restore`             | —                            | Restore/unstage changes                  |
| `merge`               | `source`                     | Merge branch                             |
| `rebase`              | `branch`                     | Rebase onto branch                       |
| `fetch`               | —                            | Fetch from remote                        |
| `sync`                | —                            | Sync branch from remote                  |
| `accumulate`          | `slice_branch`, `tracking_branch` | Rebase-accumulate into tracking     |
| `abort`               | —                            | Abort merge/rebase                       |
| `continue`            | —                            | Continue merge/rebase                    |
| `reset`               | `ref`                        | Reset to ref                             |
| `push-force-with-lease` | `branch`                   | Safe force push (blocks main/master)     |
| `worktree-list`       | —                            | List worktrees                           |
| `worktree-prune`      | —                            | Prune stale worktrees                    |
| `worktree-remove`     | `adw_id`                     | Remove worktree by ID                    |

## Split-Wrapper-First Examples

```jsonc
// Read-only status
git_diff({ "command": "status", "porcelain": true, "worktree_path": "./trees/abc" })

// Stage all files
git_stage({ "command": "add", "stage_all": true, "worktree_path": "./trees/abc" })

// Commit changes
git_commit({ "summary": "Add feature", "stage_all": true, "worktree_path": "./trees/abc" })

// Push branch pointer
git_branch({ "command": "push", "branch": "feature-123", "worktree_path": "./trees/abc" })

// Merge lifecycle command
git_merge({ "command": "merge", "source": "main", "target": "develop", "worktree_path": "./trees/abc" })

// Worktree lifecycle command
git_worktree({ "command": "worktree-list" })
```

## Legacy Compatibility Examples (`git_operations`)

```jsonc
// Commit with staging
{ "command": "commit", "summary": "Add feature", "stage_all": true, "worktree_path": "./trees/abc" }

// Push branch
{ "command": "push", "branch": "feature-123", "worktree_path": "./trees/abc" }

// Check status
{ "command": "status", "porcelain": true, "worktree_path": "./trees/abc" }

// View diff
{ "command": "diff", "stat": true, "worktree_path": "./trees/abc" }

// Stage all files
{ "command": "add", "stage_all": true, "worktree_path": "./trees/abc" }

// View log
{ "command": "log", "max_count": 5, "oneline": true, "worktree_path": "./trees/abc" }
```

## Advanced Compatibility Examples (`git_operations`)

```jsonc
// Checkout with create from source
{ "command": "checkout", "branch": "feature-123", "create": true, "source": "origin/develop" }

// Merge with no-ff
{ "command": "merge", "source": "main", "target": "develop", "no_ff": true, "worktree_path": "./trees/abc" }

// Rebase onto main
{ "command": "rebase", "branch": "develop", "onto": "main", "worktree_path": "./trees/abc" }

// Diff against base branch (PR review)
{ "command": "diff", "base": "main", "stat": true, "worktree_path": "./trees/abc" }

// Diff between two branches
{ "command": "diff", "base": "main", "target": "feature", "stat": true, "worktree_path": "./trees/abc" }

// Show file at ref
{ "command": "show", "ref": "main", "path": "src/app.py", "worktree_path": "./trees/abc" }

// Commit with traceability
{ "command": "commit", "summary": "Fix bug", "description": "Extended details", "adw_id": "a17d3f8a", "worktree_path": "./trees/abc" }

// Accumulate slice into tracking branch
{ "command": "accumulate", "slice_branch": "issue-123-adw-abc", "tracking_branch": "feature/epic-x", "worktree_path": "./trees/abc" }

// Safe force push after rebase
{ "command": "push-force-with-lease", "branch": "feature-123", "worktree_path": "./trees/abc" }

// Fetch with prune
{ "command": "fetch", "remote": "upstream", "prune": true, "worktree_path": "./trees/abc" }

// Restore staged files
{ "command": "restore", "staged": true, "files": ["src/main.py"], "worktree_path": "./trees/abc" }

// Remove worktree
{ "command": "worktree-remove", "adw_id": "abc12345", "force": true }
```

## Parameter Reference

Sparse-call normalization runs before command validation and command assembly.
The wrapper treats these inert optional defaults as omitted:

- Empty or whitespace-only strings, such as `base: ""` or `branch: "   "`
- Empty arrays, such as `files: []`
- `false` booleans for opt-in flags, such as `stat: false` or `help: false`
- `0` numeric values when they arrive as part of a schema-flooded default payload

Sparse explicit false still has command-specific meaning for `force: false` on
`worktree-remove` and `abort_on_conflict: false` on `merge`/`rebase`. In noisy
schema-flooded calls those false values are treated as omitted to avoid
accidentally disabling defaults.

### Common

| Parameter       | Type   | Default | Description                                    |
|-----------------|--------|---------|------------------------------------------------|
| `command`       | enum   | —       | Git command to execute (required)              |
| `worktree_path` | string | —      | Target worktree directory (recommended always) |
| `help`          | boolean | false  | Show CLI help instead of executing             |

### Commit

| Parameter     | Type    | Default | Description                                      |
|---------------|---------|---------|--------------------------------------------------|
| `summary`     | string  | —       | Commit message summary (required)                |
| `description` | string  | —       | Extended commit body                             |
| `adw_id`      | string  | —       | ADW ID appended to commit for traceability       |
| `stage_all`   | boolean | false   | Stage all changes before committing              |
| `no_verify`   | boolean | false   | Commit-only explicit opt-in to bypass hooks with `--no-verify` |
| `max_retries` | number  | 3       | Commit-only retry attempts, integer range `0..10` |

### Branch Operations

| Parameter | Type    | Default | Description                                      |
|-----------|---------|---------|--------------------------------------------------|
| `branch`  | string  | —       | Branch name (required for push/checkout/rebase)  |
| `source`  | string  | —       | Source ref (required for merge, optional checkout)|
| `target`  | string  | —       | Target branch for merge/sync/diff                |
| `create`  | boolean | false   | Create branch without switching (checkout only)  |
| `no_ff`   | boolean | false   | Force merge commit (merge only)                  |
| `abort_on_conflict` | boolean | true | Abort on conflict (merge/rebase)         |
| `onto`    | string  | —       | Rebase --onto target                             |

### Remote Operations

| Parameter | Type    | Default  | Description                         |
|-----------|---------|----------|-------------------------------------|
| `remote`  | string  | "origin" | Remote to fetch from                |
| `prune`   | boolean | false    | Prune stale tracking branches       |

### Accumulate

| Parameter                | Type    | Default | Description                              |
|--------------------------|---------|---------|------------------------------------------|
| `slice_branch`           | string  | —       | Slice branch to accumulate from (required)|
| `tracking_branch`        | string  | —       | Tracking branch to accumulate into (required)|
| `recover_missing_worktree` | boolean | false | Recover from missing worktree path       |

### Inspection

| Parameter  | Type    | Default | Description                              |
|------------|---------|---------|------------------------------------------|
| `porcelain`| boolean | false   | Machine-readable status output           |
| `stat`     | boolean | false   | File change statistics (diff/show)       |
| `base`     | string  | —       | Base ref for three-dot diff              |
| `ref`      | string  | —       | Git ref (required for reset/show)        |
| `path`     | string  | —       | File path at ref (show only)             |
| `max_count`| number  | 10      | Max commits in log output, integer range `1..1000` |
| `oneline`  | boolean | false   | Compact log format                       |

### File Operations

| Parameter  | Type     | Default | Description                             |
|------------|----------|---------|-----------------------------------------|
| `stage_all`| boolean  | false   | Stage all files (`add`/`commit`)        |
| `files`    | string[] | —       | Specific files to stage/restore         |
| `staged`   | boolean  | false   | Unstage from index (restore only)       |

### Reset

| Parameter | Type    | Default | Description                     |
|-----------|---------|---------|---------------------------------|
| `ref`     | string  | —       | Ref to reset to (required)      |
| `hard`    | boolean | false   | Discard working tree changes    |

### Worktree

| Parameter | Type    | Default | Description                        |
|-----------|---------|---------|------------------------------------|
| `adw_id`  | string  | —       | ADW ID for worktree-remove (required)|
| `force`   | boolean | true    | Force removal even when dirty      |

`force: false` is honored only when supplied sparsely for `worktree-remove`.
Schema-flooded default payloads with many inert optional fields treat
`force: false` as omitted, preserving the wrapper default of force removal.

## Safety Guardrails

- `push-force-with-lease` and `checkout --create` block `main`/`master` branches
- Branch refs are validated for illegal characters (`..`, `@{`, control chars, etc.)
- `add` enforces mutual exclusivity between `stage_all` and `files`
- `accumulate` returns JSON payload for programmatic consumption
- `commit` retries automatically when pre-commit hooks modify staged files

## Commit Failure Diagnostics

Commit command failures now return a structured `ERROR:` envelope:

```text
ERROR: Failed to execute 'adw git commit'
command: commit
exit_code: <code>
stderr: <bounded snippet>
stdout: <bounded snippet>
hint: <actionable guidance>
```

- `stderr`/`stdout` snippets are clipped to 500 characters.
- Clipped snippets append the exact marker: `... [truncated]`.
- Hint precedence is deterministic:
  1. lock contention (`index.lock`, `.lock`, lock phrases)
  2. hook rejection (`pre-commit`, `hook`)
  3. clean tree (`nothing to commit`, `nothing added`)
  4. generic fallback
- Non-commit command failures keep the legacy envelope format.

## Worktree Isolation

Always set `worktree_path` when running inside ADW-managed trees to ensure
operations target the correct isolated worktree. The optional `adw_id` on
commits is appended to the commit body as `ADW-ID: <id>` for traceability.

## Help Mode

Set `help: true` on any command to bypass validation and view CLI help:
```jsonc
{ "command": "diff", "help": true }
```
