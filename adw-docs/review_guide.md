<!--
Purpose: Document the code-review expectations that ADW agents and maintainers can scaffold per project.
Usage:
1. Copy this file into <DOCS_DIR>/review_guide.md during scaffold or repo setup.
2. Replace {{PROJECT_NAME}}, {{VERSION}}, {{LAST_UPDATED}}, and other placeholders with project-specific values.
3. Keep the sections updated whenever review expectations or commands change.
Placeholders: PROJECT_NAME, VERSION, LAST_UPDATED, DOCS_DIR, SOURCE_DIR, TEST_DIR, REPO_NAME, TEST_COMMAND, LINT_COMMAND, FORMAT_COMMAND, TYPE_CHECK_COMMAND, TEST_FILE_PATTERN
-->

# Code Review Guide

**Project:** {{PROJECT_NAME}}
**Version:** {{VERSION}}
**Last Updated:** {{LAST_UPDATED}}

## Overview
This guide captures the code-review standards and workflows that keep the {{PROJECT_NAME}} repository consistent, predictable, and review-ready. Follow the checklist, validation commands, and severity definitions before requesting a review.

## Repository Structure
```bash
{{REPO_NAME}}/
├── {{SOURCE_DIR}}/           # Primary application code
├── {{TEST_DIR}}/             # Tests matching {{TEST_FILE_PATTERN}}
├── {{DOCS_DIR}}/             # Documentation and guides
└── pyproject.toml            # Project configuration
```

## Validation Commands
Before opening a review, run the commands below and fix any issues they surface:
- **Tests**: `{{TEST_COMMAND}}` – ensures the suite covering {{TEST_FILE_PATTERN}} files passes.
- **Linting**: `{{LINT_COMMAND}}` – enforces formatting, style, and lint rules.
- **Formatting**: `{{FORMAT_COMMAND}}` – updates code format so diffs stay clean and consistent.
- **Type checking**: `{{TYPE_CHECK_COMMAND}}` – catches typing regressions (Python only).

## Review Checklist
- [ ] Tests pass when executing `{{TEST_COMMAND}}` (include unit, integration, or regression suites as needed).
- [ ] Lint and formatting checks succeed: `{{LINT_COMMAND}}`, `{{FORMAT_COMMAND}}`.
- [ ] Type checking passes (if applicable) via `{{TYPE_CHECK_COMMAND}}`.
- [ ] Documentation or guides are updated when public APIs or behaviors change.
- [ ] No hardcoded secrets, credentials, or sensitive data are introduced.
- [ ] Error handling follows existing conventions (clear messaging, logging, rollbacks).
- [ ] Code matches existing patterns and fits the repository’s architecture.

## Issue Severity
- **Blocker**: Critical defects that break the build, corrupt data, or leave the application unusable. Must be resolved before merging.
- **Major**: Functionality or security issues that degrade user experience or violate expectations. Address before merge unless explicitly approved.
- **Minor**: Style quirks, minor refactors, or documentation tweaks that can be resolved in follow-up work.

## See Also
- [{{DOCS_DIR}}/testing_guide.md]({{DOCS_DIR}}/testing_guide.md) – Testing framework guidance and execution tiers.
- [{{DOCS_DIR}}/linting_guide.md]({{DOCS_DIR}}/linting_guide.md) – Lint rules and formatter usage.
- [{{DOCS_DIR}}/code_style.md]({{DOCS_DIR}}/code_style.md) – Naming, imports, and style conventions.
- [{{DOCS_DIR}}/code_culture.md]({{DOCS_DIR}}/code_culture.md) – Development philosophy, 100-line rule, and smooth-is-fast principles.
