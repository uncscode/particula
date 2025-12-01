# Documentation Agent System - Usage Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-30

## Overview

The Documentation Agent System is a comprehensive suite of agents that automate documentation updates after code implementation is complete. It consists of one primary orchestrator agent and nine specialized subagents, each responsible for a specific documentation domain.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    documentation (Primary Agent)                     │
│                         mode: primary                                │
│                                                                      │
│  Responsibilities:                                                   │
│  - Read adw_spec (implementation plan + issue)                       │
│  - Analyze git diff to understand changes                           │
│  - Create todo list determining which subagents needed              │
│  - Orchestrate subagent invocations                                 │
│  - Coordinate final validation and commit                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Subagent Invocations                        │
    └─────────────────────────────────────────────────────────────────┘
                                  │
    ┌─────────────┬───────────────┼───────────────┬─────────────┐
    │             │               │               │             │
    ▼             ▼               ▼               ▼             ▼
┌─────────┐ ┌─────────┐   ┌─────────────┐ ┌───────────┐ ┌─────────────┐
│docstring│ │  docs   │   │docs-feature │ │  examples │ │architecture │
│         │ │         │   │             │ │           │ │             │
│ *.py    │ │README.md│   │feature/*.md │ │Examples/* │ │architecture/│
│docstrings│ │Agent/*.md│  │             │ │.py, .ipynb│ │decisions/*  │
└─────────┘ └─────────┘   └─────────────┘ └───────────┘ └─────────────┘
    │             │               │               │             │
    ▼             ▼               ▼               ▼             ▼
┌─────────┐ ┌─────────┐   ┌─────────────┐ ┌───────────┐ ┌─────────────┐
│docs-    │ │ theory  │   │  features   │ │docs-      │ │ git-commit  │
│maint.   │ │         │   │             │ │validator  │ │             │
│         │ │Theory/* │   │Features/*   │ │(validates)│ │(commits all)│
│maint/*.md│ │         │   │             │ │           │ │             │
└─────────┘ └─────────┘   └─────────────┘ └───────────┘ └─────────────┘
```

## Agent Summary

### Primary Agent

| Agent | Mode | Purpose |
|-------|------|---------|
| **documentation** | `primary` | Orchestrates all documentation updates |

### Documentation Subagents

| Subagent | Mode | Scope | Purpose |
|----------|------|-------|---------|
| **docstring** | `subagent` | `*.py` files | Update Python docstrings |
| **docs** | `subagent` | `docs/Agent/*.md`, `README.md`, `docs/*.md` | General documentation |
| **docs-feature** | `subagent` | `docs/Agent/feature/*.md` | Feature documentation |
| **docs-maintenance** | `subagent` | `docs/Agent/maintenance/*.md` | Maintenance docs, migrations, release notes |
| **examples** | `subagent` | `docs/Examples/*.md`, `.py`, `.ipynb` | Practical examples and tutorials |
| **architecture** | `subagent` | `docs/Agent/architecture/*.md` | ADRs, architecture outline |
| **theory** | `subagent` | `docs/Theory/*.md` | Conceptual documentation |
| **features** | `subagent` | `docs/Features/*.md` | High-level feature docs |
| **docs-validator** | `subagent` | All docs (read-only) | Validate links and formatting |

### Existing Subagents Used

| Subagent | Purpose |
|----------|---------|
| **git-commit** | Commit documentation changes |
| **linter** | Validate code quality (called by docstring) |

## File Ownership Map

```
Path                                    Responsible Agent
────────────────────────────────────────────────────────────────
*.py (source code)                      docstring
README.md                               docs
AGENTS.md                               docs
docs/index.md                           docs
docs/*.md (root level)                  docs
docs/Agent/*.md (main guides)           docs
docs/Agent/agents/*.md                  docs
docs/Agent/feature/*.md                 docs-feature
docs/Agent/maintenance/*.md             docs-maintenance
docs/Agent/architecture/*.md            architecture
docs/Agent/architecture/decisions/*.md  architecture
docs/Examples/*.md                      examples
docs/Examples/**/*.py                   examples
docs/Examples/**/*.ipynb                examples
docs/Theory/*.md                        theory
docs/Features/*.md                      features
All markdown links                      docs-validator (validates)
```

## Invocation

### Via ADW Workflow

```bash
# Run as part of complete workflow
uv run adw complete <issue-number>

# Run document phase specifically
uv run adw workflow run document <issue-number> --adw-id <adw-id>
```

### Direct Agent Invocation

```bash
# Primary agent (orchestrates all)
opencode /documentation <issue-number> --adw-id <adw-id>

# Individual subagents (usually invoked by primary)
opencode /docstring "Arguments: adw_id=<id>"
opencode /docs "Arguments: adw_id=<id>"
opencode /examples "Arguments: adw_id=<id>"
# etc.
```

## Decision Logic

The primary agent determines which subagents to invoke based on:

### Change Analysis

```
IF .py files changed:
    → docstring subagent

ALWAYS:
    → docs subagent

IF issue_class == /feature:
    → docs-feature subagent
    → examples subagent (if user-facing)
    → features subagent (if major)

IF issue_class == /bug OR deprecation:
    → docs-maintenance subagent

IF new module/component added:
    → architecture subagent

IF new design pattern introduced:
    → theory subagent

ALWAYS (after all documentation updates):
    → docs-validator subagent
    → git-commit subagent
```

### Trigger Conditions

| Condition | Subagents Invoked |
|-----------|-------------------|
| Any Python file changed | docstring |
| Any code change | docs |
| New feature (`/feature` issue) | docs-feature, examples, features |
| Bug fix (`/bug` issue) | docs-maintenance |
| New module added | architecture |
| New component/class | architecture (outline) |
| Architectural decision | architecture (ADR) |
| New design pattern | theory |
| Deprecation | docs-maintenance |
| Migration | docs-maintenance |
| ALWAYS | docs-validator, git-commit |

## Output Signals

### Primary Agent (documentation)

| Signal | Meaning |
|--------|---------|
| `DOCUMENTATION_COMPLETE` | All documentation updated successfully |
| `DOCUMENTATION_FAILED` | Critical failure, workflow stopped |

### Subagent Signals

| Subagent | Success Signal | Failure Signal |
|----------|---------------|----------------|
| docstring | `DOCSTRING_UPDATE_COMPLETE` | `DOCSTRING_UPDATE_FAILED` |
| docs | `DOCS_UPDATE_COMPLETE` | `DOCS_UPDATE_FAILED` |
| docs-feature | `DOCS_FEATURE_UPDATE_COMPLETE` | `DOCS_FEATURE_UPDATE_FAILED` |
| docs-maintenance | `DOCS_MAINTENANCE_UPDATE_COMPLETE` | `DOCS_MAINTENANCE_UPDATE_FAILED` |
| examples | `EXAMPLES_UPDATE_COMPLETE` | `EXAMPLES_UPDATE_FAILED` |
| architecture | `ARCHITECTURE_UPDATE_COMPLETE` | `ARCHITECTURE_UPDATE_FAILED` |
| theory | `THEORY_UPDATE_COMPLETE` | `THEORY_UPDATE_FAILED` |
| features | `FEATURES_UPDATE_COMPLETE` | `FEATURES_UPDATE_FAILED` |
| docs-validator | `DOCS_VALIDATION_COMPLETE` | `DOCS_VALIDATION_FAILED` |
| git-commit | `GIT_COMMIT_SUCCESS` | `GIT_COMMIT_FAILED` |

## Example Workflows

### Scenario 1: New Feature Implementation

**Context:** Issue #456 adds a new authentication module

**Subagents Invoked:**
1. **docstring** → Update docstrings in `adw/auth/operations.py`
2. **docs** → Update README.md with auth CLI commands
3. **docs-feature** → Create `docs/Agent/feature/authentication-system.md`
4. **examples** → Create `docs/Examples/authentication-tutorial.ipynb`
5. **architecture** → Update outline with auth module, potentially create ADR
6. **docs-validator** → Validate all links
7. **git-commit** → Commit "docs: add authentication documentation"

### Scenario 2: Bug Fix

**Context:** Issue #789 fixes IndexError in parser

**Subagents Invoked:**
1. **docstring** → Update docstrings if function signatures changed
2. **docs** → Update README if behavior documented there
3. **docs-maintenance** → Add to release notes
4. **docs-validator** → Validate links
5. **git-commit** → Commit "docs: update parser documentation"

### Scenario 3: Architectural Change

**Context:** Issue #234 introduces workflow engine

**Subagents Invoked:**
1. **docstring** → Update docstrings in workflow module
2. **docs** → Update README, testing_guide
3. **docs-feature** → Create feature docs for workflow engine
4. **examples** → Create workflow examples and notebooks
5. **architecture** → Create ADR, update outline
6. **theory** → Create conceptual doc for declarative workflows
7. **features** → Create high-level workflow engine doc
8. **docs-validator** → Validate all links
9. **git-commit** → Commit all changes

## Configuration

### Environment Variables

No additional environment variables required. Uses standard ADW configuration.

### Tool Access

Each agent has specific tool permissions defined in its `tools:` section:

```yaml
# Primary agent
tools:
  adw_spec: true
  read: true
  todowrite: true
  todoread: true
  bash: true
  task: true        # For invoking subagents
  glob: true
  grep: true

# Subagents (example: docstring)
tools:
  adw_spec: true
  read: true
  edit: true        # Can modify files
  write: true       # Can create files
  glob: true
  bash: true
  task: true        # For invoking linter
  todowrite: true
  todoread: true
```

## Best Practices

### For Primary Agent

1. **Analyze before invoking**: Read implementation plan and git diff thoroughly
2. **Create comprehensive todos**: One todo per subagent to invoke
3. **Handle partial failures**: Log errors, continue with other subagents
4. **Always validate**: Invoke docs-validator before commit
5. **Always commit**: Use git-commit subagent for consistent commits

### For Subagents

1. **Read required guides**: Follow documentation standards
2. **Validate links**: Check markdown links before completing
3. **Use todos**: Track progress through tasks
4. **Report clearly**: Output success/failure signals consistently
5. **Stay in scope**: Only modify files within designated scope

### For Documentation Quality

1. **Follow Google-style**: Docstrings use Google format
2. **Use kebab-case**: File names in kebab-case
3. **100-char lines**: Line length limit for prose
4. **Include examples**: All docs should have examples
5. **Link extensively**: Cross-reference related docs
6. **Validate before commit**: Always run docs-validator

## Troubleshooting

### Issue: Subagent Not Invoked

**Cause:** Change analysis didn't detect need for subagent

**Solution:** Primary agent should be more comprehensive in analysis or explicitly check issue class

### Issue: Broken Links After Update

**Cause:** Subagent created files with incorrect relative paths

**Solution:** docs-validator will catch this; fix links based on validation report

### Issue: Commit Failed

**Cause:** Pre-commit hooks failing, no changes to commit, or git error

**Solution:** Check git-commit output for specific error; retry if pre-commit hooks modified files

### Issue: Documentation Not Matching Code

**Cause:** Implementation changed after documentation written

**Solution:** Re-run documentation phase with updated context

## Integration with ADW Workflow

### In complete.json Workflow

The documentation agent runs as the `document` phase:

```json
{
  "name": "document",
  "type": "agent",
  "agent": "documentation",
  "description": "Update documentation to reflect implementation",
  "requires": ["review"]
}
```

### Phase Order

```
plan → build → test → review → document → ship
```

The documentation phase:
- Runs after review is complete
- Has access to full implementation context
- Updates all documentation before shipping
- Must complete before PR can be created

## See Also

- [documentation_guide.md](../documentation_guide.md) - Documentation standards
- [docstring_guide.md](../docstring_guide.md) - Docstring format
- [architecture_reference.md](../architecture_reference.md) - Architecture overview
- [execute-plan.md](./execute-plan.md) - Build phase agent
- [git-commit.md](./git-commit.md) - Commit subagent
- [linter.md](./linter.md) - Linter subagent
