# Backend Configuration Guide

**Last Updated:** 2026-03-05

> **Quick Start:** New to ADW? Start with the [Setup Guide](setup_guide.md) for a step-by-step walkthrough. This document covers advanced configuration options.

This guide explains how to configure repository routing for ADW workflows. ADW
supports both **GitHub** and **GitLab** platforms with automatic detection based
on repository URLs. Fork-based installs can read data from an upstream repository
while pushing branches to a contributor fork.

## Quick Start: Interactive Configuration Wizard

The easiest way to configure ADW is the interactive setup wizard:

```bash
adw setup env
```

The wizard guides you through:

1. **Platform selection** — Choose GitHub or GitLab
2. **Authentication** — Configure a PAT (with secure input masking)
3. **Repository URLs** — Set your primary repository and optional upstream
4. **Fork/upstream workflow** — Enable dual-repository routing if needed
5. **Model configuration** — Use defaults or customize model tiers

When you enable a fork/upstream workflow, the wizard now:
- Prompts for an upstream token (`GITHUB_UPSTREAM_PAT` or `GITLAB_UPSTREAM_TOKEN`) and
  lets you press Enter to reuse your fork token if it already has upstream access.
- Asks where to route issue/label operations (`ADW_TARGET_REPO`, default `upstream`).
- Guards against emitting upstream tokens when no upstream URL is provided.

The wizard generates a `.env` file with inline documentation and sets secure
file permissions (0600). For GitLab, it prompts for the required token scopes
(`api`, `read_repository`, `write_repository`).

**After running the wizard**, validate your configuration:

```bash
adw health
```

For manual configuration or advanced scenarios, continue reading below.

## Platform Selection

ADW automatically detects the platform from your repository URL:

- **GitHub**: URLs containing `github.com` (e.g., `https://github.com/owner/repo`)
- **GitLab**: URLs containing `gitlab.com` or `gitlab` in the domain (e.g., `https://gitlab.com/group/project` or `https://gitlab.company.com/team/repo`)

The platform detection logic uses URL pattern matching defined in
[`adw/platforms/factory.py`](https://github.com/Gorkowski/Agent/blob/main/adw/platforms/factory.py):

```python
# GitHub patterns
r"^https?://github\.com/"
r"^git@github\.com:"

# GitLab patterns (includes self-hosted)
r"^https?://gitlab\.com/"
r"^git@gitlab\.com:"
r"^https?://[^/]*gitlab[^/]*/"  # Self-hosted with "gitlab" in domain
r"^git@[^:]*gitlab[^:]*:"
```

For custom domains without `gitlab` in the name, use the `hint` parameter in code
or specify the platform explicitly via environment variables.

## Overview

ADW orchestrates declarative workflows stored under `.opencode/workflow/*.json`
and executed through `adw workflow <name>`. Every phase (plan → ship) needs to
know which GitHub repository should receive issue, label, and pull-request
traffic. By default ADW targets a single repository, but contributors can
optionally point read operations at an upstream project while keeping write
operations on their fork.

## OpenCode Configuration

`adw setup pull-opencode` is now the single CLI command that fetches the
canonical `.opencode/` directory from the source repository. Run it after your
initial environment configuration (or when workflows change) so every
workflow definition, tool, and template stored under `.opencode/` matches the
upstream source before you start running workflows.

```bash
adw setup pull-opencode --ref v2.3.0
```

The command uses sparse checkout to pull from `https://github.com/Gorkowski/Agent`
by default; override the source repository, ref, or path with
`--source-repo`, `--ref`, and `--source-path` as needed. Control local
overrides with `--preserve-manifest` (default `.opencode-preserve.yaml`) or
`--no-preserve`, preview the plan with `--dry-run`, and let the CLI back up the
existing directory when you let it merge changes.

This replaces the old template sync workflow. The legacy shipped template
directory has been removed, and the `adw sync` commands (for example `sync all`
and `sync commands`) were removed in v0.3.0; rely on `pull-opencode` to refresh
`.opencode/` from the canonical repo instead. Live automation templates and
refreshed automation guides now reside under `.opencode/` and
`.opencode/guides/`, while plan assets are refreshed through the related
multi-root `.opencode/plans/` paths.

## Environment Variables

Use `.env.example` or `adw/templates/.env.sample` as the canonical template when configuring ADW 3.0. The templates only list active variables; deprecated Claude/webhook settings were removed.

### Filesystem Permissions (ADW_PROJECT_ROOT)

OpenCode permission allowlists now resolve relative to `ADW_PROJECT_ROOT` with
`external_directory` set to `"deny"` by default. Set this environment variable to
an absolute path before running any workflow so read/write allowlists evaluate
correctly.

Recommended values:

- **Local development:** `export ADW_PROJECT_ROOT=$(pwd)` (repository root)
- **GitHub Actions / CI:** `export ADW_PROJECT_ROOT=$GITHUB_WORKSPACE`
- **Containers / devcontainers:** `export ADW_PROJECT_ROOT=/workspace`

#### Worktree and `cwd` Guidance (including nested worktrees)

- Keep `ADW_PROJECT_ROOT` anchored to the repository root, not an individual
  `trees/<adw_id>` directory.
- For tools that require an explicit `cwd` (for example plan-mutating wrapper
  calls), resolve it from ADW state first (`worktree_path`) and pass that
  worktree-local path directly.
- `cwd` validation is fail-closed: paths must exist, be directories, and
  resolve inside repository boundaries.
- Avoid hardcoding temporary or host-specific absolute paths; nested worktree
  layouts are supported only when canonical path resolution still maps under
  the configured project root.

### GitHub Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GITHUB_REPO_URL` | Yes | Repository ADW writes to (branches + PRs). For forked installs this must be **your fork**. | — |
| `GITHUB_PAT` | Yes | Authenticates GitHub API operations (PAT or `GITHUB_TOKEN` in Actions). | — |
| `OPENCODE_MODEL_LIGHT` | No | Optional model override for light tier. | `openai/gpt-5.1-codex-mini` |
| `OPENCODE_MODEL_BASE` | No | Optional model override for base tier. | `openai/gpt-5.1-codex-max` |
| `OPENCODE_MODEL_HEAVY` | No | Optional model override for heavy tier. | `opencode/claude-opus-4-5` |
| `GITHUB_UPSTREAM_URL` | No | Optional upstream repository that issue and label operations can target. | unset |
| `GITHUB_UPSTREAM_PAT` | No | Token for upstream operations (required when `GITHUB_UPSTREAM_URL` is set). | unset |
| `ADW_TARGET_REPO` | No | Routing selector (`fork` default) for issue/label operations. Accepts `fork`, `upstream`, or `both` (dual-scan with upstream-preferred merge; warns and falls back to fork when upstream is missing/invalid). | `fork` |
| `ADW_SYNC_FORK_UPSTREAM` | No | Fast-forward-only fork → upstream sync for default branches; skips on divergence and requires manual rebase. | `false` |
| `GITHUB_UPSTREAM_OPS` | No | Comma-delimited [`Operation`](https://github.com/Gorkowski/Agent/blob/main/adw/platforms/models.py) values that gate upstream tokens. | `DEFAULT_UPSTREAM_OPERATIONS` |
| `GITHUB_FORK_OPS` | No | Comma-delimited operation values for fork tokens (branch + PR writes). | `DEFAULT_FORK_OPERATIONS` |
| `GH_REPO` | No | Overrides repository *path* detection with an explicit `owner/repo`. Useful when the gh CLI cannot infer the right fork. | unset |
| `ADW_ENABLE_PR_SCANNING` | No | Enables PR polling alongside issues in the cron trigger. | `true` |
| `ADW_MAX_WORKFLOW_START` | No | Maximum workflows to spawn per cron poll. | `3` |
| `ADW_MAX_WORKFLOW_CONCURRENT` | No | Maximum concurrent workflows allowed by the cron trigger. | `5` |
| `ADW_HEARTBEAT_INTERVAL` | No | Minimum seconds between heartbeat status updates during agent execution. Set to 0 to disable. | `60` |
| `ADW_HEARTBEAT_STALL_THRESHOLD` | No | Seconds of no agent output before a stall warning appears in the status comment. | `120` |
| `ADW_TEST_MODE` | No | Short-circuits cron scheduling for tests. | `false` |

GitHub App credentials are no longer supported. Use a PAT (or `GITHUB_TOKEN` in GitHub Actions).

**Note:** Anthropic API key is managed by OpenCode directly (run `opencode auth` to configure your API key).

### Branch Targeting Labels

Use branch labels on issues to direct where workflow branches are created. Colors and descriptions
mirror the `BRANCH_LABELS` definitions in `adw/github/labels.py`.

| Label | Color | Description |
|-------|-------|-------------|
| `branch:develop` | `0e8a16` | Target develop branch for integration workflows |
| `branch:epic` | `1d76db` | Target epic branch for coordinated efforts |
| `branch:feature` | `5319e7` | Target feature branch for focused changes |

Precedence (highest first):
1. CLI `--base <branch>` flag
2. Issue branch label (table above)
3. `ADW_TARGET_BRANCH` environment variable
4. Default `main` branch

> **Warning:** Only one branch label is allowed. If multiple branch labels are present, ADW blocks
> processing and posts an error comment until the conflict is resolved.

### Branch Hierarchy Configuration

ADW supports an optional branch hierarchy to stage work before promotion to `main`.

- `ADW_BRANCH_HIERARCHY` (default: `true` where hierarchy is available): Enables hierarchy-aware defaults and validation. Set to `false` to operate main-only.
- `ADW_DEV_BRANCH` (default: `develop` when hierarchy is enabled): Integration branch for feature/phase work.
- Overrides: `--base <branch>` CLI flag or `branch:*` labels take precedence over defaults.

Set `ADW_BRANCH_HIERARCHY=false` to disable hierarchy temporarily and retarget bases to `main`; unset or reset `ADW_DEV_BRANCH` when hierarchy is off. For migration and rollback checklists, see the [Branch Hierarchy guide](architecture/branch-hierarchy.md#migration-from-main-only) and [rollback steps](architecture/branch-hierarchy.md#rollback).

Quick examples:

```bash
# Enable hierarchy with develop as integration branch
export ADW_BRANCH_HIERARCHY=true
export ADW_DEV_BRANCH=develop

# Run a workflow targeting develop explicitly
adw workflow complete 1317 --base develop

# Temporary rollback to main-only
export ADW_BRANCH_HIERARCHY=false
# base defaults to main; pass --base main when you want to be explicit
```

PR targeting guidance (see [Branch Hierarchy guide](architecture/branch-hierarchy.md)):

| Branch | Purpose | PR target |
|--------|---------|-----------|
| `main` | Release-ready, protected | From `develop` only |
| `develop` | Integration/staging | From `featNN-*` or `epicNN-*`; promotes to `main` |
| `epicNN-*` | Epic coordination (optional) | From feature/phase branches tied to the epic |
| `featNN-*` | Feature/phase branches | Target `develop` or active epic |

GitHub/GitLab parity: the same environment variables and branch rules apply; only the platform tokens and URLs differ. Combine hierarchy defaults with branch labels/flags above to keep base selection explicit.

### GitLab Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GITLAB_REPO_URL` | Yes (for GitLab) | GitLab repository URL. Supports both cloud and self-hosted. | — |
| `GITLAB_TOKEN` | Yes (for GitLab) | GitLab Personal Access Token (`glpat-xxx` format). | — |
| `GITLAB_UPSTREAM_URL` | No | Optional upstream GitLab project for fork workflows. | unset |
| `GITLAB_UPSTREAM_TOKEN` | No | Token for upstream project if different from main token. | unset |

**Token Scopes Required:**
- `api` - Full API access for issues, MRs, and labels
- `read_repository` - Read repository content
- `write_repository` - Push branches

#### GitLab CLI Activation (glab)

If label operations are routed through the GitLab CLI adapter, complete these steps and
set `ADW_GITLAB_USE_CLI=true` (explicit opt-in) in your environment.

1. **Install glab** and ensure it is on `PATH` (for example, `brew install glab`).
2. **Version check**: `glab >= 1.36.0` is required.
3. **Authenticate** using either `glab auth login` or by exporting `GITLAB_TOKEN` in
   your `.env` file (ADW passes the configured token to the CLI runner).
4. **Verify** the CLI can authenticate: `glab auth status` and `glab --version`.

The CLI adapter captures JSON output from `glab` and enforces a parsing size cap; if you
expect unusually large payloads, keep output sizes in mind when troubleshooting.

See [GitLab Configuration Tutorial](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/backends/gitlab-configuration.md)
for step-by-step setup instructions.

### Operation Permissions

`Operation` values describe every GitHub action ADW can perform
(`issue:read`, `label:write`, `pr:read`, etc.). Two optional environment variables
let you fine-tune which scopes each token may use:

- `GITHUB_UPSTREAM_OPS` (or `GITLAB_UPSTREAM_OPS`) limits capabilities for the
  upstream token. If unset, ADW falls back to `DEFAULT_UPSTREAM_OPERATIONS`
  (issue + label reads/writes).
- `GITHUB_FORK_OPS` (or `GITLAB_FORK_OPS`) limits fork-side tokens. Leaving it
  unset reuses `DEFAULT_FORK_OPERATIONS` (branch + PR writes and lightweight
  issue access).

The dual authentication system feeds these variables through
`Operation.parse_list`. Basic upstream routing works without setting `*_OPS`;
use them only when you need fine-grained splits per token. You can model the
behavior explicitly in code:

```python
import os

from adw.platforms import (
    DEFAULT_UPSTREAM_OPERATIONS,
    Operation,
    PlatformConfig,
    PlatformType,
    RepositoryScope,
)

raw = os.getenv("UPSTREAM_OPERATIONS")
allowed = Operation.parse_list(raw) or set(DEFAULT_UPSTREAM_OPERATIONS)

config = PlatformConfig(
    platform_type=PlatformType.GITHUB,
    scope=RepositoryScope.UPSTREAM,
    repo_url="https://github.com/org/repo",
    auth_token=os.environ["GITHUB_PAT"],
    allowed_operations=allowed,
)

config.validate_operation(Operation.ISSUE_WRITE)
```

## Dual Authentication (Fork + Upstream)

ADW supports dual-token workflows where contributor forks perform writes (branches,
PRs) while upstream repositories handle issue and label operations. This separation
allows fine-grained access control and follows the principle of least privilege.

### Dual Authentication Environment Variables

The following environment variables configure dual-token authentication. All
`*_OPS` variables accept comma-delimited [`Operation`](https://github.com/Gorkowski/Agent/blob/main/adw/platforms/models.py)
values (e.g., `issue:read,label:write,pr:read`). They are optional for basic
upstream routing; set them only when you need fine-grained token splits.

#### GitHub Dual Authentication

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GITHUB_REPO_URL` | Yes | Fork repository URL (branches + PRs created here). | — |
| `GITHUB_PAT` | Yes | Token for fork operations. | — |
| `GITHUB_FORK_OPS` | No | Comma-delimited operations for fork token. | `DEFAULT_FORK_OPERATIONS` |
| `GITHUB_UPSTREAM_URL` | No | Upstream repository URL (issues + labels read here). | unset |
| `GITHUB_UPSTREAM_PAT` | No | Token for upstream operations. Required if `GITHUB_UPSTREAM_URL` is set. | unset |
| `GITHUB_UPSTREAM_OPS` | No | Comma-delimited operations for upstream token. | `DEFAULT_UPSTREAM_OPERATIONS` |

#### GitLab Dual Authentication

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GITLAB_REPO_URL` | Yes (GitLab) | Fork project URL. | — |
| `GITLAB_TOKEN` | Yes (GitLab) | Token for fork operations. | — |
| `GITLAB_FORK_OPS` | No | Comma-delimited operations for fork token. | `DEFAULT_FORK_OPERATIONS` |
| `GITLAB_UPSTREAM_URL` | No | Upstream project URL. | unset |
| `GITLAB_UPSTREAM_TOKEN` | No | Token for upstream operations. Required if `GITLAB_UPSTREAM_URL` is set. | unset |
| `GITLAB_UPSTREAM_OPS` | No | Comma-delimited operations for upstream token. | `DEFAULT_UPSTREAM_OPERATIONS` |

### Operation Permissions Reference

Every platform action ADW can perform maps to an `Operation` enum value. Use these
identifiers in `*_OPS` environment variables to customize token capabilities.

| Operation | Description |
|-----------|-------------|
| `issue:read` | Fetch issue metadata, comments, and labels from the repository. |
| `issue:write` | Create or edit issues in the configured repository scope. |
| `issue:comment` | Post new comments or update existing issue comments. |
| `label:read` | List labels to map platform taxonomy locally. |
| `label:write` | Create, update, or delete labels. |
| `pr:read` | Read pull request metadata, comments, and reviews. |
| `pr:write` | Create pull requests or push review updates. |
| `pr:merge` | Merge pull requests when permitted by repository rules. |
| `pr:approve` | Approve pull requests by submitting reviews. |
| `status:read` | Read commit statuses and check run results. |
| `status:write` | Post or update commit statuses and check runs. |
| `branch:create` | Create new branches on the remote repository. |
| `branch:push` | Push commits to branches (fast-forward or force depending on policy). |

### Default Permissions

When `*_OPS` environment variables are not set, ADW uses sensible defaults:

**Fork Defaults (`DEFAULT_FORK_OPERATIONS`):**

```text
issue:read, issue:write, issue:comment, label:read, label:write, pr:read, pr:write, status:read, status:write, branch:create, branch:push
```

Fork tokens support full single-repository workflows: reading and writing issues,
managing labels, creating branches, pushing commits, and managing PR lifecycle.
This enables fork-only setups (without upstream configuration) to perform all
common operations including issue creation and label management.

**Upstream Defaults (`DEFAULT_UPSTREAM_OPERATIONS`):**

```text
issue:read, issue:write, issue:comment, label:read, label:write, status:read
```

Upstream tokens focus on issue triage: reading and writing issues, managing labels,
and reading commit statuses. PR operations are excluded since contributors create
PRs on their forks.

**Fork-Only vs. Fork+Upstream Routing:**

When only a fork is configured (no `*_UPSTREAM_URL`), all operations route to the
fork client. When both fork and upstream are configured, the router prefers the
fork client for operations that both support, with upstream as fallback for
operations the fork cannot perform.

## Platform Router

The Platform Router provides automatic routing of operations to the appropriate
platform client based on permissions and scope. It is configured automatically
from environment variables and handles the complexity of selecting the correct
client for each operation.

```python
from adw.platforms import get_platform_router

router = get_platform_router()  # Configured from environment
issue = router.fetch_issue("123")  # Automatic routing to correct client
```

The router:
- Routes write operations (PRs, branches) to fork by default
- Routes read operations to whichever client is configured
- Derives `prefer_scope` for `fetch_issue` calls from the workflow target (`ADW_TARGET_REPO`) and the
  resolved issue repo path when no flag is provided; the shared helper is used by the CLI, router,
  dispatcher, cron trigger, and GitHub operations to keep fork/upstream routing consistent.
- Supports explicit scope selection via `prefer_scope` and falls back to the available client when the
  preferred scope is missing, including dual-scan (`ADW_TARGET_REPO=both`) where fork is tried before
  upstream.
- Accepts explicit scope overrides in the CLI (`adw platform <command> --prefer-scope upstream`) and
  in `platform_operations` payloads (`prefer_scope: "upstream"`) to force routing when hints suggest it.
- Provides backward-compatible facades for deprecated `operations.py` functions

For complete documentation, see:
- [Platform Router Feature](https://github.com/Gorkowski/Agent/blob/main/docs/Features/platform-router.md) - Full API reference
- [Platform Router Examples](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/advanced/platform-router-usage.md) - Usage examples

### Configuration Examples

#### Example 1: Single Repository (Default)

For direct repository access without fork/upstream separation:

```bash
# All operations on one repository (backward compatible)
export GITHUB_REPO_URL=https://github.com/myorg/myproject
export GITHUB_PAT=ghp_your_token
```

No additional variables needed. ADW uses the single token for all operations with
full default permissions.

#### Example 2: Fork Contributor

Read issues from upstream, create PRs on your fork:

```bash
# Your fork (where branches and PRs are created)
export GITHUB_REPO_URL=https://github.com/you/fork
export GITHUB_PAT=ghp_fork_token

# Upstream repository (where issues live)
export GITHUB_UPSTREAM_URL=https://github.com/org/project
export GITHUB_UPSTREAM_PAT=ghp_upstream_token

# Optional: restrict upstream to read-only operations
export GITHUB_UPSTREAM_OPS=issue:read,issue:comment,label:read
```

With this configuration:
- Issue fetching uses `GITHUB_UPSTREAM_PAT` against `GITHUB_UPSTREAM_URL`
- Branch creation and PRs use `GITHUB_PAT` against `GITHUB_REPO_URL`
- Routing is automatic based on operation type

#### Example 3: Restricted CI/CD

Minimal permissions for automated workflows:

```bash
# CI/CD token with limited scope
export GITHUB_REPO_URL=https://github.com/org/project
export GITHUB_PAT=ghp_ci_token

# Only allow reading issues and PRs, writing status checks
export GITHUB_FORK_OPS=issue:read,pr:read,status:write
```

This prevents the CI token from creating issues, modifying labels, or merging PRs.

#### Example 4: GitLab Dual-Token Setup

Fork + upstream on GitLab (self-hosted or cloud):

```bash
# Your fork
export GITLAB_REPO_URL=https://gitlab.com/me/fork
export GITLAB_TOKEN=glpat-fork_token

# Upstream project
export GITLAB_UPSTREAM_URL=https://gitlab.com/org/upstream
export GITLAB_UPSTREAM_TOKEN=glpat-upstream_token

# Optional: customize operations
export GITLAB_FORK_OPS=pr:read,pr:write,branch:create,branch:push
export GITLAB_UPSTREAM_OPS=issue:read,label:read
```

## Fork Sync

`ADW_SYNC_FORK_UPSTREAM` enables a safety-first sync from your fork's default
branch to upstream via fast-forward only. When enabled:

- ADW attempts to fast-forward your fork's default branch from upstream on
  cron-driven syncs.
- Divergence is detected and skipped automatically—your fork is never rewritten.
- Network/API errors are retried on the next cycle; permission errors hard-fail
  so you can fix credentials.

Enable fork sync:

```bash
export ADW_SYNC_FORK_UPSTREAM=true
```

If divergence is reported, resolve manually:

```bash
git fetch upstream
# Rebase or merge as policy requires; example:
git rebase upstream/main
```

After resolving, rerun `adw health` and relaunch cron if stopped.

## Auto-Routing Behavior

When you set `ADW_TARGET_REPO=upstream`, issue and label operations automatically
route upstream while PRs/branches stay on your fork. Dual-scan mode
(`ADW_TARGET_REPO=both`) fetches issues from fork then upstream in one pass,
merging with upstream-preferred results. `*_OPS` variables are optional for this
basic routing; use them only for fine-grained token splits.

Env-derived routing for privileged write operations (issue updates, comments,
label writes, PR reviews) is disabled by default. Set
`ADW_ALLOW_ENV_WRITE_ROUTING=true` to enable env-derived write routing when you
intend to route writes based on `ADW_TARGET_REPO`.

### Routing Matrix

| Operation | Fork-only | `ADW_TARGET_REPO=upstream` | `ADW_TARGET_REPO=both` |
|-----------|-----------|----------------------------|------------------------|
| Issues | Fork | Upstream | Fork first, then upstream (merged) |
| Labels | Fork | Upstream | Fork first, then upstream (merged) |
| PRs | Fork | Fork | Fork |
| Branches | Fork | Fork | Fork |
| Fork sync (`ADW_SYNC_FORK_UPSTREAM`) | Disabled | Optional (fork → upstream fast-forward) | Optional (fork → upstream fast-forward) |

### Security Best Practices

Follow these recommendations when configuring dual authentication:

1. **Use Fine-Grained PATs**: Prefer GitHub fine-grained tokens over classic tokens.
   Fine-grained tokens allow precise repository and permission scoping, reducing
   blast radius if compromised.

2. **Least Privilege Principle**: Configure only the operations each token needs:
   - Fork token: `pr:write,branch:push,status:write` (no merge capability)
   - Upstream token: `issue:read,label:read` (read-only when possible)

3. **Avoid `pr:merge` on Fork Tokens**: Excluding merge permissions prevents
   accidental merges from automation. Require human review for merge operations.

4. **Use Separate Tokens**: Never reuse a powerful upstream token for fork
   operations. Different tokens isolate credential scope and simplify rotation.

5. **Environment Isolation**: Keep production and development tokens separate.
   Use different credentials for CI/CD versus local development.

6. **Token Rotation**: Rotate tokens periodically. Fine-grained PATs support
   expiration dates; set reasonable lifetimes and calendar reminders.

> **Warning**: Granting `pr:merge` or `pr:approve` to upstream tokens allows
> automation to merge PRs. Only enable these operations when intentional and
> properly reviewed.

**GitLab Token Scopes**: Ensure GitLab tokens include `api`, `read_repository`,
and `write_repository` scopes. Self-hosted instances may require additional
scopes depending on configuration.

## Repository Detection Order

Detection order: `GH_REPO` override → authenticated `gh` CLI context → local
`origin` remote.

Workspace creation and issue fetching rely on
`get_repo_context_with_fallback` to resolve the `owner/repo` path for GitHub
API calls. The helper evaluates the following strategies in order:

1. **`GH_REPO` override** — set `export GH_REPO=owner/repo` to pin the
   repository path when you already know which fork ADW should read from.
2. **Authenticated `gh` CLI context** — if the override is absent, ADW shells
   out to the authenticated GitHub CLI session
   (`gh repo view --json nameWithOwner`) so fork-aware routing follows your
   logged-in identity.
3. **Local `origin` remote** — as a final fallback ADW inspects
   `git config --get remote.origin.url` and extracts `owner/repo` from the URL.

Each successful detection emits a structured log entry you can grep for when
workspaces fail to initialize:

```text
Repository detection completed (repo=octo/agent, method=origin_remote)
```

The `method` field will be `env_override`, `gh_cli`, or `origin_remote` so you
can see exactly which strategy succeeded. If all strategies fail, workspace
creation surfaces an actionable ValueError before attempting to fetch the issue.
Issue fetch failures now include the detected repository, detection method, and
the standard troubleshooting checklist so you can validate authentication
quickly.

## Workflow Modes

### GitHub: Single-repository installs (default)

1. Set `GITHUB_REPO_URL` to the only repository you operate.
2. Leave `GITHUB_UPSTREAM_URL` unset; `ADW_TARGET_REPO` automatically stays on
   `fork`.
3. Issue, label, and PR operations all use the same repository.

### GitHub: Fork + upstream installs

1. Set `GITHUB_REPO_URL` to your fork so branch creation and PRs stay under your
   control.
2. Set `GITHUB_UPSTREAM_URL` to the canonical project where issues and labels
   live.
3. Choose the active routing target by either exporting
   `ADW_TARGET_REPO=upstream` or passing `--target upstream` to any
   `adw workflow <name>` invocation. Issue/label operations route upstream
   automatically; fork PR/branch operations stay on your fork. You do **not**
   need `GITHUB_UPSTREAM_OPS` for this basic routing; keep `*_OPS` only when you
   need fine-grained allowlists.
4. GitHub helper utilities (`get_target_repo*`) and workflow state management
   persist the choice so every `.opencode/workflow/*.json` definition that runs
   afterward inherits the same routing without extra flags.
5. Optional fork sync: set `ADW_SYNC_FORK_UPSTREAM=true` to fast-forward your
   fork's default branch from upstream when it is ahead; sync skips on divergence
   so you can rebase manually.

### GitHub: Dual-scan fork + upstream in one cycle

1. Set `GITHUB_REPO_URL` to your fork and `GITHUB_UPSTREAM_URL` to the canonical
   project.
2. Enable dual-scan with `ADW_TARGET_REPO=both` (or `--target both`).
3. The cron issue poller fetches fork issues first, then upstream, logging the
   source for each issue.
4. Results merge in O(n) by issue number with upstream entries winning conflicts;
   rate/concurrency gates still apply after merge. `*_OPS` variables remain
   optional unless you need fine-grained splits per token.
5. If the upstream URL is missing/invalid, ADW logs a single warning per cron
   cycle and scans the fork only. Upstream fetch failures also degrade
   gracefully to the fork list without stopping the cron run.

### GitLab: Single-project installs

1. Set `GITLAB_REPO_URL` to your GitLab project URL.
2. Set `GITLAB_TOKEN` to a Personal Access Token with required scopes.
3. Issue, label, and MR operations all use the same project.

```bash
# GitLab Cloud
export GITLAB_REPO_URL=https://gitlab.com/myorg/myproject
export GITLAB_TOKEN=glpat-xxxxxxxxxxxx

# Self-hosted GitLab
export GITLAB_REPO_URL=https://gitlab.company.com/team/project
export GITLAB_TOKEN=glpat-xxxxxxxxxxxx
```

### GitLab: Fork + upstream installs

1. Set `GITLAB_REPO_URL` to your forked project.
2. Set `GITLAB_TOKEN` with push access to your fork.
3. Set `GITLAB_UPSTREAM_URL` to the canonical project.
4. Optionally set `GITLAB_UPSTREAM_TOKEN` if the upstream requires different
   credentials.

```bash
# Fork + Upstream on GitLab
export GITLAB_REPO_URL=https://gitlab.com/me/fork
export GITLAB_TOKEN=glpat-fork-token
export GITLAB_UPSTREAM_URL=https://gitlab.com/org/upstream
export GITLAB_UPSTREAM_TOKEN=glpat-upstream-token  # Optional
```

## CLI Overrides and Persistence

You can override the routing target per workflow without editing `.env`. The
flag is honored by dedicated entries such as `adw workflow complete` and the
generic runner `adw workflow run`.

```bash
# Route issue + label operations upstream while PRs remain on your fork
adw workflow complete 123 --target upstream

# Invoke the JSON definition stored in .opencode/workflow/complete.json via the runner
adw workflow run complete 123 --target upstream

# Revert to fork routing for the next workflow
adw workflow complete 124 --target fork

# Dual-scan fork + upstream in one cron cycle (issues merged with upstream
# preference; falls back to fork-only if upstream missing/invalid)
adw workflow complete 125 --target both
```

ADW writes the chosen target to both the process environment and the workflow
state record so downstream phases (test, review, document, ship) and cron/dispatch
jobs reuse the same routing decision automatically.

## Routing Lifecycle

- **Issue + label operations**: Follow the selected target (`fork` or
  `upstream`).
- **Pull-request operations**: Always use `GITHUB_REPO_URL` (your fork) to keep
  branch ownership consistent, even when issues route upstream.
- **Dispatcher + cron trigger**: Honor the persisted `ADW_TARGET_REPO` value and
  call the same helper utilities used by the CLI, ensuring consistent behavior
  no matter which phase executes the JSON workflow definition.

## Warnings and Compatibility

- If `ADW_TARGET_REPO=upstream` or `both` but `GITHUB_UPSTREAM_URL` is
  missing/invalid, ADW emits a single warning per cron cycle and safely falls
  back to scanning the fork only so PR routing never breaks.
- Upstream fetch failures during dual-scan log an error but continue processing
  fork issues (rate/concurrency gates still apply after merge).
- Existing single-repo installs continue to function without setting new
  variables.
- `.env.example` and `README.md` mirror this guide; run `uv run adw health` after
  changing variables to verify connectivity.

## Platform Migration

Use this section when moving an existing ADW installation between GitHub and
GitLab or when switching from single-repo to fork+upstream routing. Always start
new workflows after changing platforms instead of resuming previous
`adw_state.json` files.

### GitHub → GitLab

- **Token scopes (GitLab):** `api`, `read_repository`, `write_repository`.
- **Environment mapping:**

| GitHub | GitLab |
|--------|--------|
| `GITHUB_REPO_URL` | `GITLAB_REPO_URL` |
| `GITHUB_PAT` | `GITLAB_TOKEN` |
| `GITHUB_UPSTREAM_URL` | `GITLAB_UPSTREAM_URL` |
| `GITHUB_UPSTREAM_PAT` | `GITLAB_UPSTREAM_TOKEN` |
| `GITHUB_UPSTREAM_OPS` | `GITLAB_UPSTREAM_OPS` |
| `GITHUB_FORK_OPS` | `GITLAB_FORK_OPS` |

- **State + workflows:** Start a fresh workflow; do not resume GitHub
  `adw_state.json` on GitLab. Workflow JSON definitions stay platform-agnostic.
- **Routing:** Mixed-platform operations within a single workflow are unsupported
  today.
- **Checklist:**
  - [ ] Create GitLab PAT with required scopes
  - [ ] Swap `GITHUB_*` variables for `GITLAB_*` equivalents
  - [ ] Unset any stale GitHub variables
  - [ ] Run `adw health` to confirm connectivity
  - [ ] Test with `adw workflow plan <issue-number>` on GitLab

### GitLab → GitHub

- **Token scopes (GitHub):** use fine-grained PAT covering issues, labels,
  branches/PRs, and status updates.
- **Environment mapping:**

| GitLab | GitHub |
|--------|--------|
| `GITLAB_REPO_URL` | `GITHUB_REPO_URL` |
| `GITLAB_TOKEN` | `GITHUB_PAT` |
| `GITLAB_UPSTREAM_URL` | `GITHUB_UPSTREAM_URL` |
| `GITLAB_UPSTREAM_TOKEN` | `GITHUB_UPSTREAM_PAT` |
| `GITLAB_UPSTREAM_OPS` | `GITHUB_UPSTREAM_OPS` |
| `GITLAB_FORK_OPS` | `GITHUB_FORK_OPS` |

- **State + workflows:** Start new workflows; do not reuse GitLab workflow IDs on
  GitHub. Workflow definitions remain portable.
- **Checklist:**
  - [ ] Create GitHub fine-grained PAT with required scopes
  - [ ] Swap `GITLAB_*` variables for `GITHUB_*` equivalents
  - [ ] Unset any stale GitLab variables
  - [ ] Run `adw health`
  - [ ] Test with `adw workflow plan <issue-number>` on GitHub

### Single-repo → fork+upstream (either platform)

- Set upstream URL + token when issues/labels live upstream but PRs live on your
  fork. Keep fork tokens for branch + PR writes.
- Select routing target with `ADW_TARGET_REPO` (`fork` default, `upstream`
  optional). If `ADW_TARGET_REPO=upstream` but the upstream URL is missing,
  ADW falls back to the fork safely.
- **Checklist:**
  - [ ] Set `*_REPO_URL` to your fork; set `*_PAT`/`*_TOKEN` with push access
  - [ ] Set `*_UPSTREAM_URL` (and upstream token) if reading issues upstream
  - [ ] Choose `ADW_TARGET_REPO` based on where issues should route
  - [ ] Validate with `adw health`
  - [ ] Run `adw workflow plan <issue-number>` to confirm detection

### Common Migration Issues

- **Stale environment variables** lead to 401/404 responses. Clear old variables
  before exporting new ones:

```bash
unset GITHUB_PAT GITHUB_REPO_URL GITHUB_UPSTREAM_URL \
  GITLAB_TOKEN GITLAB_REPO_URL GITLAB_UPSTREAM_URL
```

- **Issue vs MR numbering:** GitHub uses `#123`; GitLab issues use `#123` and
  merge requests use `!123`.
- **Self-hosted GitLab domains:** ensure the URL includes your custom domain;
  platform detection falls back to GitHub if `gitlab` is absent and no hint is
  provided.

### Migration Checklists

- **GitHub → GitLab:** tokens swapped, variables remapped, state restarted, router
  validated via `adw health`.
- **GitLab → GitHub:** PAT created with correct scopes, variables remapped, state
  restarted, router validated via `adw health`.
- **Single → fork+upstream:** upstream URLs/tokens configured, `ADW_TARGET_REPO`
  selected, fallback behavior understood.

### Mixed-Platform Workflows (Future)

Mixed-platform operations within the same workflow are not supported today. See
[Platform Router](https://github.com/Gorkowski/Agent/blob/main/docs/Features/platform-router.md) for current routing behavior
and future considerations.

## Additional Resources

- [Root README](https://github.com/Gorkowski/Agent/blob/main/README.md#backend-configuration) — Quick reference and
  onboarding instructions.
- [Backend Configuration Example](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/backends/configuration.md) —
  Step-by-step walkthrough for a forked contributor environment.
- [Platform Comparison](https://github.com/Gorkowski/Agent/blob/main/docs/Features/platform-comparison.md) — Differences across
  GitHub vs GitLab (Phase F4).
- [Platform Router Feature](https://github.com/Gorkowski/Agent/blob/main/docs/Features/platform-router.md) — Routing behavior
  and API reference.
- [GitLab Configuration Tutorial](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/backends/gitlab-configuration.md) —
  Complete GitLab setup guide.
- [GitLab Support Feature](https://github.com/Gorkowski/Agent/blob/main/docs/Features/gitlab-support.md) — GitLab platform
  capabilities and limitations.
- [Epic E1: Multi-platform Support](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/epics/completed/E1-multi-platform-support.md)
  — Track and phase details.
- [Architecture Outline](architecture/architecture_outline.md) — Explains how
  declarative `.opencode/workflow/*.json` definitions interact with backend
  routing at runtime.
