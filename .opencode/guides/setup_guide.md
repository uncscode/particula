# ADW Setup Guide

Single-source setup for ADW. For advanced routing see
[Backend Configuration](backend_configuration.md); troubleshooting will land in
[troubleshooting_setup.md](troubleshooting_setup.md).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Migration from Removed Commands](#migration-from-removed-commands)
- [Getting OpenCode Configuration](#getting-opencode-configuration)
- [Platform-Specific Setup](#platform-specific-setup)
  - [GitHub Personal Access Token](#github-personal-access-token)
  - [GitLab (cloud or self-hosted)](#gitlab-cloud-or-self-hosted)
- [Fork Workflow Setup](#fork-workflow-setup)
- [Validation](#validation)
- [Scaffolding Documentation](#scaffolding-documentation)
- [Next Steps](#next-steps)

## Prerequisites
- Python 3.12+, Git, and [uv](https://astral.sh/uv)
- GitHub or GitLab access (token)
- Optional: GitHub CLI (`gh`), `direnv`, WSL for Windows users
- Network: allow HTTPS to Anthropic and your platform; corporate firewalls may
require proxy settings
- Note: Anthropic API key is managed by OpenCode directly. Run `opencode auth`
  to configure after installation.
- Note: The `ast-grep` CLI is installed with the dev extras (`uv pip install -e ".[dev]"`).

## Installation
```bash
git clone https://github.com/Gorkowski/Agent.git
cd Agent
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
adw --help
```

## Quick Start
Run the wizard first (prefix with `uv run` if needed).
```bash
adw setup env
adw setup validate
adw workflow complete <issue-number>
```
```mermaid
graph TD
    A[adw setup] --> B{Platform?}
    B --|GitHub| C[GitHub PAT]
    B --|GitLab| D[GitLab Token]
    C --> G[Anthropic Key]
    D --> G
    G --> H[Repo URLs]
    H --> I{Fork?}
    I --|Yes| J[Upstream URL]
    I --|No| K[Model]
    J --> K
    K --> L[.env]
    L --> M[Validate]
    M --> N{Valid?}
    N --|Yes| O[Pull OpenCode Configuration]
    N --|No| P[Fix]
    P --> M
    O --> Q[Run Workflow]
```
Wizard sample:
```bash
$ adw setup env
Platform [GitHub/GitLab]: GitHub
GitHub PAT (ghp_...): ********
Saved .env with 0600 permissions
```

After the wizard and validation, run `adw setup pull-opencode` (for example `--ref v2.3.0`) to fetch the
canonical `.opencode/` contents before running workflows.

## Migration from Removed Commands

These commands were removed in v0.3.0. Use the supported replacements below to
stay on the current setup and workflow paths.

| Removed command | Replacement | Notes |
| ----- | ----- | ----- |
| `adw config init` / `adw config set ...` / `adw config validate` | `adw setup env`, edit `.env` directly, and `adw setup validate` | `setup env` generates `.env`; edit values manually, then validate. |
| `adw diagnose *` | `adw health` (dashboard) and/or `adw setup validate` (config validation) | Use `adw health` for the health dashboard and `adw setup validate` for configuration validation. |
| `adw init` | `adw setup pull-opencode` + `adw setup template apply` + `adw setup env` | Pull workflows, apply templates, then generate `.env`. |
| `adw tools *` | Manage `.mcp.json` manually | MCP tooling was removed from ADW; `.mcp.json` is no longer managed by ADW commands but may still be used by other MCP-aware tools. If you still rely on MCP, manage this file manually and keep any custom MCP config in version control. |
| `adw multiworkflow *` | `adw workflow <name>` | Use declarative JSON workflows registered under `.opencode/workflow/*.json`. |
| `adw health-check` | `adw health` | Duplicate removed in favor of `health`. |

## Getting OpenCode Configuration
Use `adw setup pull-opencode` to fetch the canonical `.opencode/` directory from
the ADW source repository before you run workflows or apply templates locally.
This command replaces the old template sync flow and the legacy shipped
templates; the `adw sync` commands (for example `sync all` and `sync commands`)
were removed in v0.3.0. Live automation templates and scaffolding/sync
documentation now resolve canonically under `.opencode/` (including
`.opencode/guides/`), with related multi-root plan assets under
`.opencode/plans/` and legacy fallback support for older repositories.
`pull-opencode` is the single CLI that brings those files into your local
workspace.

### Quick Start
```bash
# Pull from the default ADW repository
adw setup pull-opencode

# Pin to a specific ref (tag, branch, or commit)
adw setup pull-opencode --ref v2.3.0

# Pull from a custom repository
adw setup pull-opencode --source-repo https://github.com/myorg/my-adw-config
```

### Command Options
| Option | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `--source-repo` | `ADW_OPENCODE_SOURCE_REPO` | `https://github.com/Gorkowski/Agent` | Git repository to pull from |
| `--source-path` | `ADW_OPENCODE_SOURCE_PATH` | `.opencode` | Path inside the repository to sparse checkout |
| `--dest` | `ADW_OPENCODE_DEST_PATH` | `.opencode` | Local destination directory |
| `--ref` | – | `main` | Git ref to pull (branch, tag, or commit) |
| `--yes` | – | – | Skip prompts; default to backup when destination exists |
| `--dry-run` | – | – | Show planned actions without making changes |
| `--preserve-manifest` | – | `.opencode-preserve.yaml` | YAML manifest path (relative to the current directory unless absolute) listing files, directories (trailing `/`), or globs to keep |
| `--preserve / --no-preserve` | – | `--preserve` | Toggle preservation. Enabled stashes + restores matched paths so preserved content wins. Use `--no-preserve` to skip. |

### Preserve Custom Content
`adw setup pull-opencode` now honors a preserve manifest (`.opencode-preserve.yaml` by
default, relative to your current working directory unless you pass an absolute path or
`--preserve-manifest`). When preservation is enabled, matched files, directories (use a
trailing `/`), and glob patterns are stashed before the pull and restored afterward so
local overrides win.

- `--dry-run` reports which paths would be preserved without touching the filesystem.
- `--no-preserve` bypasses stash/restore entirely.
- Merge strategy already keeps existing files, so manifest preservation is skipped there.

Example manifest:
```yaml
# .opencode-preserve.yaml
preserve:
  - docs/local-overrides.md
  - scripts/local-*.sh
  - ops/hooks/
```

### Handling Existing Configuration
When `.opencode/` already exists, the command prompts for a strategy (or uses
`--yes` to auto-select backup):

1) **Backup and replace (default)** — moves the directory to a dated backup
2) **Merge** — copies only new files into the existing directory
3) **Overwrite** — deletes the current directory before pulling
4) **Cancel** — aborts without changes

Backups use the pattern `<dest>.backup.YYYY-MM-DD[.n]` and are restored
automatically if the pull fails.

### Environment Persistence
After the first successful pull, defaults are persisted to `.env` via the CLI
wizard (source repo/path/destination). Subsequent pulls reuse these values unless
flags override them.

### Troubleshooting
- **Invalid repository URL**: verify the URL/protocol; private repos require git
  credentials.
- **Path not found**: ensure `--source-path` exists on the chosen `--ref`.
- **Invalid ref**: confirm the branch, tag, or commit exists; git reports checkout
  failures when the ref is missing.
- **Git too old**: sparse checkout needs git ≥ 2.25; upgrade if you see version
  errors.

Integration + slow E2E tests cover the happy path, version pinning, invalid
repo/ref/path, and the backup workflow.

### Version Pinning
Use refs for reproducible configuration states:
```bash
adw setup pull-opencode --ref v2.3.0  # tag
adw setup pull-opencode --ref develop # branch
adw setup pull-opencode --ref <commit-sha>
```

Template Bootstrap (Step 7):
- Runs after project root selection and before the success summary.
- Default prompt is **Yes** for new projects (no `.opencode` content) and **No** when `.opencode` already has files.
- Accepting writes `.opencode/.adw-template-manifest.yaml`, ensures `.gitignore` contains ADW
  entries, and applies templates with substitutions. Default entries cover worktrees/state
  (`trees/`, `*trees/`, `agents/*`, `*.adw`, `.adw/`), local config
  (`.opencode/opencode.local.json`, `.opencode.backup.*/`, `.envrc`), and runtime logs (`agents/*`).
- Gitignore mode: the wizard asks whether to add entries as **active** (default, also used for
  non-interactive runs) or **commented** for review. Pick **commented** when onboarding in audited
  repositories so reviewers can enable patterns later, or **active** to immediately block
  accidental commits. The same choice is exposed on `adw setup template init --gitignore-mode
  active|commented`.
- Declining or template apply warnings never block `.env` generation.
- Flags: `--with-templates` forces bootstrap without prompting (uses defaults); `--skip-templates` removes the prompt entirely.

## Platform-Specific Setup
### GitHub Personal Access Token
- Fine-grained PAT with `repo`, `workflow`; set `ADW_PROJECT_ROOT` to repo path.
- GitHub App authentication is not supported; use PATs (or `GITHUB_TOKEN` in Actions).
```env
GITHUB_PAT=ghp_***
GITHUB_REPO_URL=https://github.com/<you>/<fork>
ADW_PROJECT_ROOT=/absolute/path/to/Agent
ADW_TARGET_REPO=fork
OPENCODE_MODEL_LIGHT=openai/gpt-5.1-codex-mini
OPENCODE_MODEL_BASE=openai/gpt-5.1-codex-max
OPENCODE_MODEL_HEAVY=opencode/claude-opus-4-5
# Note: Anthropic API key is managed by OpenCode. Run `opencode auth` to configure.
```

### GitLab (cloud or self-hosted)
- Scopes: `api`, `read_repository`, `write_repository`; auto-detects hosts with
`gitlab`, otherwise set `ADW_PLATFORM_HINT=gitlab`. See
[GitLab Configuration Tutorial](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/backends/gitlab-configuration.md).
```env
GITLAB_REPO_URL=https://gitlab.com/<group>/<project>
GITLAB_TOKEN=glpat-***
# Self-hosted
#GITLAB_REPO_URL=https://code.company.com:8443/team/project
#ADW_PLATFORM_HINT=gitlab
```

## Fork Workflow Setup
Keep branches/PRs on your fork and route issues upstream when needed. See the
fork+upstream settings in
[Backend Configuration](backend_configuration.md#github-single-repository-installs-default)
for required variables and the [Routing Feature](https://github.com/Gorkowski/Agent/blob/main/docs/Features/fork-upstream-routing.md)
for behavior.
```env
GITHUB_REPO_URL=https://github.com/<you>/Agent
GITHUB_UPSTREAM_URL=https://github.com/<org>/Agent
ADW_TARGET_REPO=upstream
#GITHUB_UPSTREAM_PAT=ghp_upstream
```

## Validation
Run after editing `.env`; re-run when credentials change.
```bash
adw setup validate
adw health
```
```mermaid
graph LR
    A[setup validate] --> B[Env Vars]
    B --> C[Anthropic]
    C --> D[Platform API]
    D --> E[Repo Access]
    E --> F[Git Config]
    F --> G[System Check]
    G --> H{All Pass?}
    H --|Yes| I[✓ Ready]
    H --|No| J[Show Fixes]
```
Sample output:
```bash
$ adw setup validate
✓ Environment file loaded (.env)
✓ Anthropic API connectivity
✓ GitHub connectivity: ok (fork)
✓ Git config present (user.name, user.email)
```
If proxies block requests, set `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`.

## Scaffolding Documentation
Use `adw setup docs scaffold` to create `adw-docs/` from packaged language stubs and
write `.adw-docs-manifest.yaml` at the repository root for downstream commands.

Supported languages:
- python
- cpp
- typescript
- minimal

Typical workflow:
```bash
adw setup docs scaffold --language python
cat .adw-docs-manifest.yaml
adw setup docs apply --language python
```

`--force` overwrites an existing `adw-docs/` directory (no backup is taken):
```bash
adw setup docs scaffold --language cpp --force
```

Success output (truncated):
```bash
$ adw setup docs scaffold --language python
✓ Scaffolded adw-docs with 'python' templates.
ℹ Created files:
  adw-docs/README.md
  adw-docs/testing_guide.md
  adw-docs/code_style.md
  .adw-docs-manifest.yaml
ℹ Next steps:
  1. Review .adw-docs-manifest.yaml and add project values as needed.
  2. Commit adw-docs/ and the manifest to your repository.
```

Error output when docs already exist without `--force`:
```bash
$ adw setup docs scaffold --language python
[ERROR] adw-docs already exists at /path/to/repo/adw-docs.
         Re-run with --force to overwrite existing docs.
```

Manifest schema example (created_by includes `--force` when provided):
```yaml
version: "1.0"
language: python
created_at: 2025-12-15T10:30:00Z
created_by: "adw setup docs scaffold --language python"
values: {}
```
`.adw-docs-manifest.yaml` is preferred; `.yml` is accepted when `.yaml` is absent.
The command errors if both exist. The manifest stores placeholder values for future
apply operations.

Language stub content comes from [E7-F2 Language Stub Templates](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/features/completed/E7-F2-language-stub-templates.md).
Apply and placeholder replacement shipped with
[E7-F3 Docs Apply](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/features/completed/E7-F3-docs-apply-command.md), and project detection
shipped with [E7-F4 Project Detection](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/features/completed/E7-F4-project-detection.md)
(completed 2025-12-17 via #1050-#1053).
See the [E7-F1 Docs Scaffold Command plan](https://github.com/Gorkowski/Agent/blob/main/adw-docs/dev-plans/features/completed/E7-F1-docs-scaffold-command.md) for context.

## Next Steps
- Run: `adw workflow complete <issue-number>` (use `--target` or `ADW_TARGET_REPO`).
- Explore advanced options in [Backend Configuration](backend_configuration.md).
- More GitLab examples: [GitLab Configuration Tutorial](https://github.com/Gorkowski/Agent/blob/main/docs/Examples/backends/gitlab-configuration.md).
- Troubleshoot setup failures in [troubleshooting_setup.md](troubleshooting_setup.md);
  start with `adw health` when errors appear.
