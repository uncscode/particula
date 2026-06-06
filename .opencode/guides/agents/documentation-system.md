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
│docstring│ │  docs   │   │plan-update- │ │  examples │ │architecture │
│         │ │         │   │   full      │ │           │ │             │
│ *.py    │ │README.md│   │sections/*.md│ │Examples/* │ │architecture/│
│docstrings│ │Agent/*.md│  │             │ │.py, .ipynb│ │decisions/*  │
└─────────┘ └─────────┘   └─────────────┘ └───────────┘ └─────────────┘
    │             │               │               │             │
    ▼             ▼               ▼               ▼             ▼
┌─────────┐ ┌─────────┐   ┌─────────────┐ ┌───────────┐ ┌─────────────┐
│plan-    │ │ theory  │   │  features   │ │docs-      │ │ git-commit  │
│update-  │ │         │   │             │ │validator  │ │             │
│short    │ │Theory/* │   │Features/*   │ │(validates)│ │(commits all)│
│(status) │ │         │   │             │ │           │ │             │
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
| **docs** | `subagent` | `adw-docs/*.md`, `README.md`, `docs/*.md` | General documentation |
| **plan-update-full** | `subagent` | `plans/sections/**/*.md` | Plan section content updates |
| **plan-update-short** | `subagent` | Plan metadata via `adw_plans` | Phase status and plan lifecycle |
| **examples** | `subagent` | `docs/Examples/*.md`, `.py`, `.ipynb` | Practical examples and tutorials |
| **architecture** | `subagent` | `adw-docs/architecture/*.md` | ADRs, architecture outline |
| **theory** | `subagent` | `docs/Theory/*.md` | Conceptual documentation |
| **features** | `subagent` | `docs/Features/*.md` | High-level feature docs |
| **docs-validator** | `subagent` | All docs (read-only) | Validate links, formatting, and agent references via narrow wrappers |

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
adw-docs/*.md (main guides)           docs
adw-docs/agents/*.md                  docs
plans/sections/**/*.md                    plan-update-full
Plan phase status (via adw_plans)         plan-update-short
adw-docs/architecture/*.md            architecture
adw-docs/architecture/decisions/*.md  architecture
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
uv run adw workflow complete <issue-number>

# Run document phase specifically
uv run adw workflow document <issue-number> --adw-id <adw-id>
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
    → plan-update-full subagent
    → examples subagent (if user-facing)
    → features subagent (if major)

IF issue_class == /bug OR deprecation:
    → plan-update-full subagent

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
| New feature (`/feature` issue) | plan-update-full, examples, features |
| Bug fix (`/bug` issue) | plan-update-full |
| New module added | architecture |
| New component/class | architecture (outline) |
| Architectural decision | architecture (ADR) |
| New design pattern | theory |
| Deprecation | plan-update-full |
| Migration | plan-update-full |
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
| plan-update-full | `PLAN_UPDATE_FULL_COMPLETE` | `PLAN_UPDATE_FULL_FAILED` |
| plan-update-short | `PLAN_UPDATE_SHORT_COMPLETE` | `PLAN_UPDATE_SHORT_FAILED` |
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
3. **plan-update-full** → Update plan sections for authentication feature
4. **examples** → Create `docs/Examples/authentication-tutorial.ipynb`
5. **architecture** → Update outline with auth module, potentially create ADR
6. **docs-validator** → Validate all links
7. **git-commit** → Commit "docs: add authentication documentation"

### Scenario 2: Bug Fix

**Context:** Issue #789 fixes IndexError in parser

**Subagents Invoked:**
1. **docstring** → Update docstrings if function signatures changed
2. **docs** → Update README if behavior documented there
3. **plan-update-full** → Update plan sections for parser fix
4. **docs-validator** → Validate links
5. **git-commit** → Commit "docs: update parser documentation"

### Scenario 3: Architectural Change

**Context:** Issue #234 introduces workflow engine

**Subagents Invoked:**
1. **docstring** → Update docstrings in workflow module
2. **docs** → Update README, testing_guide
3. **plan-update-full** → Update plan sections for workflow engine
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

Each agent has explicit permissions defined in its `permission:` map.
Active guidance uses a deny-by-default baseline plus explicit allowlisting.

```yaml
# Primary agent
permission:
  "*": deny
  adw_spec_read: allow
  adw_spec_write: allow
  adw_spec_messages: allow
  read: allow
  todowrite: allow
  task: allow         # For invoking subagents
  list: allow
  grep: allow

# Subagents (example: docstring)
permission:
  "*": deny
  adw_spec_read: allow
  adw_spec_messages: allow
  read: allow
  edit: allow         # Covers write/edit/apply_patch built-ins
  list: allow
  grep: allow
  task: allow         # For invoking linter
  todowrite: allow
```

Historical `tools:` examples may appear only in migration/deprecated context
and should never be presented as the active pattern.

## Best Practices

### For Primary Agent

1. **Analyze before invoking**: Read implementation plan and git diff thoroughly
2. **Create comprehensive todos**: One todo per subagent to invoke
3. **Handle partial failures**: Log errors, continue with other subagents
4. **Always validate**: Invoke docs-validator before commit
5. **Always commit**: Use git-commit subagent for consistent commits

Docs-validator should use validation-safe wrappers for repo checks. In
particular, prefer `run_validate_agent_references` over raw shell/script
execution for agent-reference validation.

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
