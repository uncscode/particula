# ADW Setup Agent - Usage Guide

## Overview

The `setup-adw` agent helps set up ADW in repositories where it's installed as a package. This agent is for **users deploying ADW** to their own projects, not for ADW maintainers.

The setup process includes:
1. Scaffolding `adw-docs/` with language-specific documentation stubs
2. Initializing the template manifest with ~15 keywords
3. Applying templates to create `.opencode/` configuration
4. Managing template keyword tokens and docs placeholders
5. Validating the configuration

**Note**: Environment configuration (`adw setup env`) is a user-only interactive wizard and should not be run by this agent.

## When to Use

| Scenario | Use This Agent |
|----------|----------------|
| First-time ADW setup in a repo | Yes |
| Scaffolding adw-docs/ documentation | Yes |
| Applying templates to .opencode/ | Yes |
| Managing template keyword tokens | Yes |
| Managing docs placeholder values | Yes |
| Validating ADW configuration | Yes |
| Updating after ADW package upgrade | Yes |
| Syncing platform labels | Yes |
| Configuring .env credentials | No - user-only wizard |
| Extracting docs to templates (maintainer) | No - use `adw-setup-maintainer` |

## Prerequisites

Before using this agent:

1. **Install ADW**:
   ```bash
   pip install adw
   # or
   uv pip install adw
   ```

2. **Initialize Git** (if not done):
   ```bash
   git init
   ```

3. **Configure environment** (user runs manually):
   ```bash
   adw setup env
   ```

## Permissions

| Tool | Enabled | Rationale |
|------|---------|-----------|
| `read` | Yes | Read configs, detect project type |
| `edit` | Yes | Update configuration files |
| `write` | Yes | Create .opencode/ and adw-docs/ files |
| `list/glob/grep` | Yes | File discovery |
| `adw` | Yes | Run setup commands |
| `workflow_builder_read` | Yes | Verify workflows (`list/get/validate`) |
| `workflow_builder_mutate` | Yes | Create/update setup workflows when needed |
| `workflow_builder` | Yes (compatibility) | Transitional compatibility surface |
| `git_diff` | Yes | Check status and inspect repo state |
| `git_stage` | Yes | Stage setup-generated changes |
| `git_commit` | Yes | Create setup commits when needed |
| `task` | Yes | Invoke adw-commit subagent |
| `bash` | Yes | Run adw CLI commands directly |
| `platform_operations` | No | Not creating PRs |
| `run_pytest` | No | Not testing |
| `run_linters` | No | Not linting |

## Key Commands

### Documentation Scaffolding

```bash
# Scaffold for Python project
adw setup docs scaffold --language python

# Scaffold for C++ project
adw setup docs scaffold --language cpp

# Scaffold for TypeScript project
adw setup docs scaffold --language typescript

# Minimal scaffolding
adw setup docs scaffold --language minimal

# Force overwrite existing adw-docs/
adw setup docs scaffold --language python --force
```

### Documentation Token Management

```bash
# List configured placeholder values
adw setup docs token list

# Set or update a placeholder value
adw setup docs token set PROJECT_NAME "My Project"

# Remove a placeholder value
adw setup docs token remove OLD_KEY
```

### Apply Documentation Templates

```bash
# Preview substitutions
adw setup docs apply --dry-run

# Check for unresolved placeholders
adw setup docs apply --check

# Apply substitutions
adw setup docs apply
```

### Template Setup

```bash
# Initialize manifest (prompts for ~15 keywords)
adw setup template init
adw setup template init --yes  # Accept defaults

# Preview what would be applied
adw setup template apply --dry-run

# Check for placeholders only
adw setup template apply --check

# Apply templates
adw setup template apply
adw setup template apply --yes  # Skip prompts
```

### Template Token Management

```bash
# List all defined keywords
adw setup template token list

# Add or update a keyword
adw setup template token add PACKAGE_NAME --default mypackage --description "Python package name"

# Remove a keyword
adw setup template token remove CUSTOM_KEYWORD
```

### Validation

```bash
# Fast preflight (no network)
adw setup check

# Full validation (with connectivity)
adw setup validate
adw setup validate --format json  # CI-friendly

# Health check
adw health
```

### Platform Labels

```bash
# Preview what would be synced
adw setup labels --dry-run

# Sync labels
adw setup labels
```

### Pull OpenCode Configuration

```bash
# Pull from default source (Agent repo)
adw setup pull-opencode

# Pull specific version
adw setup pull-opencode --ref v2.3.0

# Preview what would be pulled
adw setup pull-opencode --dry-run
```

## Setup Workflow

### Step 1: Scaffold Documentation

```bash
adw setup docs scaffold --language python
```

Creates language-specific documentation stubs:
- `code_style.md` - Coding conventions
- `testing_guide.md` - Testing practices
- `architecture_reference.md` - Architecture overview
- `docstring_guide.md` - Documentation format

### Step 2: Configure Documentation Placeholders

```bash
# List current values
adw setup docs token list

# Set project-specific values
adw setup docs token set PROJECT_NAME "My Project"
adw setup docs token set PACKAGE_NAME "mypackage"
```

### Step 3: Initialize Template Manifest

```bash
adw setup template init
```

Select `template` mode (default for deployments).

Configure ~15 keywords:
```
PACKAGE_NAME [adw]: mypackage
PROJECT_NAME [Agent Developer Workflow]: My Project
SOURCE_DIR [adw/]: src/mypackage/
REPO_URL []: https://github.com/myuser/myproject
TEST_COMMAND [pytest]: pytest src/
...
```

### Step 4: Apply Templates

```bash
# Preview first
adw setup template apply --dry-run

# Then apply
adw setup template apply
```

### Step 5: Final Validation

```bash
adw health
adw status
adw setup validate
```

## Usage Examples

### Example 1: Fresh Python Project Setup

```
User: "Help me set up ADW in this Python project"

Agent: "I'll analyze your repository first...

Repository Analysis:
- Language: Python (pyproject.toml found)
- Package: mypackage
- Test Framework: pytest
- No existing .opencode/ directory
- No existing adw-docs/ directory

Setup Plan:
1. Scaffold adw-docs/ documentation
2. Initialize template manifest  
3. Apply templates
4. Validate configuration

Ready? [Y/n]"

User: "Y"

Agent: [Runs through each step]

"ADW SETUP COMPLETE

Files Created:
- adw-docs/*.md (documentation guides)
- .opencode/.adw-template-manifest.yaml
- .opencode/agent/*.md (22 agents)
- .opencode/workflow/*.json (8 workflows)

Validation:
- adw health: PASSED
- Workflows: complete, patch, plan, build, test, review, document, ship

Next Steps:
1. Review configuration
2. Try: adw workflow complete <issue-number>
3. Commit: git add .opencode adw-docs"
```

### Example 2: Update After ADW Upgrade

```
User: "I upgraded ADW, what do I need to update?"

Agent: "I'll check for template changes...

Running: adw setup template apply --dry-run

New templates available:
- .opencode/agent/new-agent.md
- .opencode/workflow/documentation-update.json

Existing files (won't be overwritten):
- 45 files unchanged

Apply new templates? [Y/n]"

User: "Y"

Agent: "Running: adw setup template apply --yes

Added 2 new files
All placeholders resolved

Run 'adw health' to verify configuration."
```

### Example 3: Update Template Keywords

```
User: "I need to change the package name in templates"

Agent: "I'll update the template keyword...

Running: adw setup template token add PACKAGE_NAME --default newpackage

Keyword updated. Now applying to templates...

Running: adw setup template apply --yes

Templates updated with new package name."
```

## Manifest Mode

For deployments, always use `template` mode:

```yaml
# .opencode/.adw-template-manifest.yaml
mode: template  # Not 'live'
values:
  PACKAGE_NAME: mypackage
  PROJECT_NAME: My Project
  ...
```

If you accidentally select `live` mode:
```bash
# Reinitialize and select 'template'
adw setup template init
```

## Troubleshooting

### ADW CLI Not Found

```
Error: adw: command not found

Solution:
  pip install adw
  # or
  uv pip install adw
```

### Template Apply Failed

```
Error: Template manifest not found

Solution:
  adw setup template init
```

### Wrong Manifest Mode

```
Error: Manifest mode is 'live', expected 'template'

Solution:
  # Reinitialize in template mode
  adw setup template init
  # Select 'template' when prompted
```

### Placeholders Remain

```
Warning: Unknown placeholders found: {{CUSTOM_VAR}}

Solution:
  # Add to manifest with token command
  adw setup template token add CUSTOM_VAR --default "value"
  
  # Or manually replace in .opencode/ files
```

## Integration with Other Agents

| Agent | Integration |
|-------|-------------|
| `adw-setup-maintainer` | Complementary - maintainer creates templates you deploy |
| `adw-commit` | Invoked to commit setup files |
| `adw-build` | Uses configuration created by this agent |
| `tester` | Uses test paths configured here |

## See Also

- [ADW Setup Guide](../setup_guide.md) - Complete setup documentation
- [Backend Configuration](../backend_configuration.md) - Platform setup details
- [ADW Setup Maintainer](adw-setup-maintainer.md) - Template maintenance agent
