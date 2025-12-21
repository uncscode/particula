<!--
Summary: Task-driven conditional documentation guide for {{PROJECT_NAME}}.
Usage:
- Copy from this template into {{DOCS_DIR}}/conditional_docs.md when scaffolding documentation.
- Retain the sections in this order so the validation tests can locate each heading.
- Update placeholder values with repository-specific metadata before shipping the stub.
Placeholders:
- {{PROJECT_NAME}}
- {{DOCS_DIR}}
- {{LAST_UPDATED}}
- {{LINE_LENGTH}}
- {{REPO_URL}}
- {{SOURCE_DIR}}
-->

# Conditional Documentation Guide

**Project:** {{PROJECT_NAME}}
**Last Updated:** {{LAST_UPDATED}}

This guide helps {{PROJECT_NAME}} contributors decide which documentation to read based on a focused task. Only open the referenced guide if one of the listed conditions applies; avoid blanket reading so you stay efficient.

## Instructions

- Review the work item and the scope in the associated issue or plan.
- Check each documentation entry below, evaluating the conditions for relevance.
- Read a given guide only when at least one condition matches your current task.
- Record new references in this file whenever new docs are scaffolded so the pattern stays complete.

## Placeholder Table

| Placeholder | Description |
| --- | --- |
| {{PROJECT_NAME}} | Human-friendly project name injected from the manifest. |
| {{DOCS_DIR}} | Target directory for generated documentation (usually `adw-docs`). |
| {{SOURCE_DIR}} | Primary source directory that contains the {{PROJECT_NAME}} codebase. |
| {{LAST_UPDATED}} | Date this template was scaffolded so readers know how recent the advice is. |
| {{LINE_LENGTH}} | Target line-length guideline for prose and code examples referenced here. |
| {{REPO_URL}} | Repository URL surfaced for quick navigation from stubbed docs. |

## Conditional Documentation

### Core Repository Files

- `README.md`
  - Conditions:
    - When you first understand the repository structure, entry points, or installation instructions.
    - When you configure packaging, CI, or onboarding automation in {{PROJECT_NAME}}.
- `code_culture.md`
  - Conditions:
    - When you want to align with the development philosophy (smooth is safe, safe is fast) before writing code.
    - When you are learning the 100-line rule or the preferred issue flow for {{PROJECT_NAME}}.

### Development Documentation

- `code_style.md`
  - Conditions:
    - When you are writing new production code and need naming, formatting, or typing guidance.
    - When you are reviewing code for stylistic consistency.
- `testing_guide.md`
  - Conditions:
    - When adding tests, troubleshooting failures, or interpreting fixture expectations.
    - When you are responsible for increasing coverage for a targeted module.
- `linting_guide.md`
  - Conditions:
    - When configuring linting tooling (ruff, mypy) or responding to lint findings.
    - When adjusting formatting commands or lint scripts in automation.
- `docstring_guide.md`
  - Conditions:
    - When writing or updating public docstrings in {{SOURCE_DIR}}.
    - When documenting non-obvious behavior that requires examples.
- `review_guide.md`
  - Conditions:
    - When conducting or responding to code reviews for {{PROJECT_NAME}}.
    - When you need to interpret the review checklist or reviewer expectations.
- `commit_conventions.md`
  - Conditions:
    - When crafting git commits or updating commit templates for the repo.
    - When you want to verify that branch naming and issue linking follow policy.
- `pr_conventions.md`
  - Conditions:
    - When opening a pull request or updating PR titles and descriptions.
    - When you need to understand the labels and reviewers that should be involved.

### Architecture References

- `architecture/architecture_guide.md`
  - Conditions:
    - When designing new modules, data flow, or integration patterns.
    - When you need to match error-handling, testing, or orchestration conventions defined by {{PROJECT_NAME}} architecture.
- `architecture/architecture_outline.md`
  - Conditions:
    - When you need a high-level map of component responsibilities and dependencies.
    - When figuring out where to place a new feature or service in the system.
- `architecture/decisions/README.md`
  - Conditions:
    - When creating a new Architecture Decision Record (ADR) or reviewing past decisions.
    - When you care about historical context for a technology or design choice.

### Development Plans

- `dev-plans/README.md`
  - Conditions:
    - When you author or review epics, features, or maintenance plans.
    - When you link work items to roadmap documentation or completed plan archives.

## Usage Tips

1. **Match the document to your task**: Only open the documentation entry whose conditions match the work you are currently doing.
2. **Read sweep-friendly sections**: Start with README or dev-plan summaries before digging into detailed architecture or testing chapters.
3. **Link to {{REPO_URL}}**: Bookmark document paths that are referenced frequently so you can jump back quickly when the repo structure changes.
4. **Keep this file current**: Add new entries if future scaffolds include additional docs so the conditional list never lags behind reality.
