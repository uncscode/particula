<!--
Summary: Architecture outline starter that gives a concise system overview, component map, and quick reference for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/architecture/architecture_outline.md when scaffolding documentation.
- Fill in component descriptions, stack choices, and extension points.
- Keep links relative to the architecture folder; update when files move.
Placeholders:
- {{PROJECT_NAME}}
- {{VERSION}}
- {{LAST_UPDATED}}
- {{SOURCE_DIR}}
- {{LANGUAGE}}
- {{TEST_FRAMEWORK}}
- {{LINTER_TOOLS}}
- {{DOCS_DIR}}
-->

# Architecture Outline

**Project:** {{PROJECT_NAME}}
**Version:** {{VERSION}}
**Last Updated:** {{LAST_UPDATED}}

## System Overview

Summarize the mission of {{PROJECT_NAME}}, key users, and the primary workflows it supports.

## Core Components

| Component | Purpose | Location |
| --- | --- | --- |
| Core | Domain types and validation | `{{SOURCE_DIR}}/core/` |
| Services | Business logic and orchestration | `{{SOURCE_DIR}}/services/` |
| API | Adapters and entrypoints (CLI/HTTP/workers) | `{{SOURCE_DIR}}/api/` |
| Utils | Cross-cutting helpers | `{{SOURCE_DIR}}/utils/` |

## Module Structure

```
{{SOURCE_DIR}}/
├── core/
├── services/
├── api/
└── utils/
```

Add or prune directories to reflect the actual repository layout.

## Technology Stack

| Category | Technology | Purpose |
| --- | --- | --- |
| Language | {{LANGUAGE}} | Primary implementation language |
| Testing | {{TEST_FRAMEWORK}} | Test framework |
| Linting/Formatting | {{LINTER_TOOLS}} | Code quality tools |
| Documentation | {{DOCS_DIR}} | Default docs root |

## Quick Reference

### Design Principles
- Separation of concerns
- Explicit dependencies
- Fail fast, with actionable errors
- Design for testability

### Extension Points
- Add new services under `{{SOURCE_DIR}}/services/`
- Add shared helpers under `{{SOURCE_DIR}}/utils/`
- Register new adapters in `{{SOURCE_DIR}}/api/`
- Document architecture changes in `architecture/decisions/`

## Related Documentation

- [Architecture Guide](architecture_guide.md)
- [Architecture Decisions](decisions/README.md)
- [Architecture Reference](../architecture_reference.md)
