# Tools-to-Permission Migration Guide

Reference for migrating agent frontmatter from the deprecated `tools:` format
to the current `permission:` format in `.opencode/agent/*.md` files.

---

## Current Policy (Repository)

- Active `.opencode/agent/*.md` frontmatter uses `permission:` only.
- Active inventory is fail-closed and deny-by-default (`"*": deny` baseline).
- Required capabilities are explicitly allowlisted with `allow` (or `ask` where
  intentional human approval is required).
- Any retained `tools:` snippets are historical/deprecated migration context
  only and must not be presented as active guidance.

## Why Migrate

As of OpenCode v1.1.1, the `tools:` frontmatter key is deprecated. The
replacement is `permission:`, which provides finer-grained control
(`allow`/`ask`/`deny`) and supports glob-pattern matching for tool names and
bash commands. The old `tools` config still works for backwards compatibility,
but new and updated agents should use `permission:`.

See the upstream docs:
- [Agents — Permissions](https://opencode.ai/docs/agents#permissions)
- [Permissions reference](https://opencode.ai/docs/permissions)

---

## Key Differences

| Aspect | `tools:` (deprecated) | `permission:` (current) |
|---|---|---|
| Values | `true` / `false` | `allow` / `ask` / `deny` |
| Default | Each tool listed explicitly | Use `"*": deny` as a baseline |
| Granularity | Per tool name only | Per tool name, glob patterns, bash command patterns |
| Built-in grouping | None (each tool listed) | `edit` gates `write`, `edit`, `apply_patch`; `todowrite` gates `todowrite` and `todoread`; `grep` gates built-in `grep` only |
| Custom tools | Listed by tool name | Listed by tool name (wildcard patterns supported) |

---

## Built-in Permission Keys

These permission keys gate built-in OpenCode tools:

| Permission key | Tools it gates |
|---|---|
| `read` | `read` |
| `edit` | `write`, `edit`, `apply_patch` |
| `glob` | `glob` |
| `grep` | `grep` (built-in only) |
| `list` | `list` |
| `bash` | `bash` |
| `task` | `task` |
| `todowrite` | `todowrite`, `todoread` |
| `webfetch` | `webfetch` |
| `websearch` | `websearch` |
| `skill` | `skill` |
| `question` | `question` |
| `lsp` | `lsp` |
| `external_directory` | Any tool touching paths outside the project worktree |
| `doom_loop` | Recovery prompts when an agent appears stuck |

---

## Custom Tools — Use Split Wrappers, Not Glob Families

Custom tools defined in `.opencode/tools/` are **not** covered by built-in
permission keys. Each custom tool needs its own explicit permission entry.

**List each split wrapper individually.** Do not use glob patterns like
`"git_*": allow` or `"adw_*": allow` to blanket-allow tool families. Agents
should declare the minimum set of specific tools they actually use — this
enforces least-privilege and makes tool access auditable.

### Split wrapper lookup

Many tools have been split into focused wrappers. When migrating, replace
monolithic tool names with the specific split wrappers the agent needs:

| Monolithic (deprecated for permissions) | Split wrappers (use these) |
|---|---|
| `adw_spec` | `adw_spec_read`, `adw_spec_write`, `adw_spec_messages` |
| `adw_plans` | `adw_plans_read`, `adw_plans_mutate` |
| `adw_issues_spec` | `adw_issues_batch_init`, `adw_issues_batch_read`, `adw_issues_batch_write`, `adw_issues_batch_log`, `adw_issues_batch_summary` |
| `adw_notes` | `adw_notes_read`, `adw_notes_write` |
| `git_operations` | `git_diff`, `git_stage`, `git_commit`, `git_branch`, `git_merge`, `git_worktree` |
| `platform_operations` | `platform_pr_write`, `platform_pr_read`, `platform_issue_read`, `platform_issue_write`, `platform_label_write`, `platform_rate_limit_read` (**compatibility-only exception:** keep `platform_operations` only for unsplit commands `comment` and `pr-review` until dedicated wrappers land) |
| `run_pytest` | `run_pytest_basic`, `run_pytest_advanced` |
| `run_cmake` | `run_cmake_configure`, `run_cmake_build` |
| `run_sanitizers` | `run_sanitizers_basic`, `run_sanitizers_advanced` |
| `run_cpp_linters` | `run_cpp_lint_check`, `run_cpp_lint_fix` |
| `run_cpp_coverage` | `run_cpp_coverage_summary`, `run_cpp_coverage_advanced` |
| `refactor_astgrep` | `refactor_astgrep_preview`, `refactor_astgrep_apply` |
| `workflow_builder` | `workflow_builder_read`, `workflow_builder_mutate` |
| `validate_notebook` | `validate_notebook_readonly`, `convert_notebook_to_py`, `convert_py_to_notebook`, `sync_notebook_pair` |
| `move` | `move_safe`, `move_overwrite`, `move_trash` |
| `clear_build` | `clear_build_preview`, `clear_build_delete` |
| `ripgrep` | `find_files`, `search_content`, `ripgrep_advanced` |
| `build_mkdocs` | `build_mkdocs_validate`, `build_mkdocs_build` |

Only grant the specific split wrappers the agent actually calls. For example,
an agent that only reads spec state and writes messages needs:

```yaml
permission:
  adw_spec_read: allow
  adw_spec_messages: allow
```

Not `adw_spec: allow` and not `"adw_spec_*": allow`.

For compatibility-window boundaries and retirement gates, use the canonical
repository policy in [AGENTS.md](../../AGENTS.md#compatibility-window-retirement-gates).

### Other custom tools (no split wrappers)

These tools have a single wrapper and should be listed by exact name:

| Custom tool | Notes |
|---|---|
| `ripgrep` | Compatibility wrapper; prefer `find_files` / `search_content` / `ripgrep_advanced` |
| `feedback_log` | Feedback logging |
| `get_datetime` / `get_version` | Utility tools |
| `create_workspace` | Workspace creation |
| `auto_mode_manifest` | Auto-mode manifest |
| `run_linters` | Python linter runner |
| `run_ctest` | C++ test runner |
| `run_bun_test` | TypeScript test runner |
| `run_notebook` | Notebook/script execution |

---

## Required Permissions for All Agents

Every agent in this repository **must** have these two permissions regardless of
its role:

| Permission | Why |
|---|---|
| `feedback_log: allow` | All agents need to log friction, bugs, and workflow gaps reactively. |
| `adw_spec_messages: allow` | All agents need to read and write workflow context messages. |

Include both in every `permission:` block. When migrating from the `tools:`
format, add them explicitly — `adw_spec_messages` did not exist in the old
format, and `feedback_log` was not always present.

---

## Migration Steps

### 1. Read the current `tools:` block

Identify which tools are `true` (enabled) and which are `false` (disabled).

### 2. Replace `tools:` with `permission:`

Start with a deny-all baseline, then explicitly allow what's needed:

```yaml
permission:
  "*": deny
  # ... allowed tools below
```

### 3. Map `true` entries to `allow`

For each `tool_name: true` entry:

- If it's a **built-in** (`read`, `edit`, `list`, `bash`, `task`, etc.),
  use the built-in permission key.
- If it's a **custom tool** (`ripgrep`, `move`, `adw_spec`, etc.),
  add it by its tool name.

Note the grouping: `edit: allow` covers `write`, `edit`, and `apply_patch`
but does **not** cover custom tools like `move`. Similarly `grep: allow`
covers the built-in `grep` but not the custom `ripgrep`.

### 4. Drop `false` entries

The `"*": deny` baseline handles all disabled tools. You do not need to
list tools that should be denied.

### 5. Handle `todoread` / `todowrite`

The `todowrite` permission key gates **both** `todowrite` and `todoread`.
Replace both entries with a single `todowrite: allow`.

---

## Before / After Example

**Before** (deprecated `tools:` format):

```yaml
mode: subagent
tools:
  read: true
  edit: true
  write: true
  move: true
  list: true
  ripgrep: true
  todoread: true
  todowrite: true
  task: false
  adw: false
  adw_spec: true
  adw_plans: true
  adw_issues_spec: false
  feedback_log: true
  create_workspace: false
  workflow_builder: false
  git_operations: false
  platform_operations: false
  run_pytest: false
  run_linters: false
  get_datetime: true
  get_version: false
  refactor_astgrep: false
  webfetch: false
  websearch: false
  codesearch: false
  bash: false
```

**After** (current `permission:` format with split wrappers):

```yaml
mode: subagent
permission:
  "*": deny
  read: allow
  edit: allow
  move_safe: allow
  list: allow
  grep: allow
  find_files: allow
  search_content: allow
  todowrite: allow
  adw_spec_read: allow
  adw_spec_write: allow
  adw_spec_messages: allow
  adw_plans_read: allow
  adw_plans_mutate: allow
  feedback_log: allow
  get_datetime: allow
```

Key observations:
- `edit: allow` replaces both `edit: true` and `write: true` (built-in grouping)
- `move_safe: allow` uses the specific split wrapper needed (not `move` or `"move_*"`)
- `grep: allow` enables the built-in grep
- `find_files` / `search_content` replace monolithic `ripgrep` (custom tool, not covered by `grep`)
- `adw_spec_read` / `adw_spec_write` / `adw_spec_messages` replace monolithic `adw_spec`
- `adw_plans_read` / `adw_plans_mutate` replace monolithic `adw_plans`
- `todowrite: allow` replaces both `todoread: true` and `todowrite: true`
- All `false` entries are dropped — covered by `"*": deny`

---

## Using `ask` for Gated Approval

The `permission` format supports `ask` for interactive approval. This is
useful for tools that should be available but require human confirmation:

```yaml
permission:
  "*": deny
  read: allow
  edit: ask        # prompt before file modifications
  bash: ask        # prompt before shell commands
  task: allow
```

---

## Glob Patterns — When (Not) to Use Them

Glob patterns are supported by OpenCode's permission system, but **should be
avoided for custom tool families** in this repository. Instead, list each
split wrapper the agent needs by its exact name.

**Why not globs?**
- `"git_*": allow` grants `git_commit`, `git_merge`, `git_branch`, etc. —
  an agent that only needs `git_diff` would get write access it shouldn't have.
- `"adw_*": allow` grants every ADW tool including destructive ones like
  `adw_spec_write` when the agent may only need `adw_spec_read`.
- Glob patterns defeat least-privilege and make tool access harder to audit.

**When globs are acceptable:**
- The deny-all baseline `"*": deny` is always appropriate.
- Bash command patterns like `"bash(npm *)": allow` for scoped shell access.
- Rare cases where an agent genuinely needs every tool in a family AND the
  family has no destructive members.

**Preferred pattern — explicit split wrappers:**

```yaml
permission:
  "*": deny
  read: allow
  edit: allow
  list: allow
  grep: allow
  todowrite: allow
  # ADW state: only what this agent needs
  adw_spec_read: allow
  adw_spec_messages: allow
  # Git: read-only inspection
  git_diff: allow
  # Platform: read-only
  platform_issue_read: allow
  # Utilities
  feedback_log: allow
  get_datetime: allow
```

---

## Validation

After migration, verify the agent loads correctly:

```bash
opencode debug agent build
```

Check that:
1. The agent appears in the output with the correct permission configuration.
2. Tools that should be available are listed.
3. Tools that should be denied are not accessible.

---

## Migration Tracking

This repo has 97 agent files in `.opencode/agent/`.

**Migration status: COMPLETE.** All 97 agents use the `permission:` format
with `"*": deny` baseline. No agents use the deprecated `tools:` format.

The test suite (`adw/tests/agent_permission_validation_test.py`) enforces this
invariant: `test_active_agents_use_permission_not_tools` and
`test_active_agents_do_not_use_tools_frontmatter` fail closed if any agent
regresses to the deprecated format.
