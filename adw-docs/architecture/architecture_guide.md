<!--
Summary: Architecture guide starter that documents principles, patterns, anti-patterns, and architecture practices for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/architecture/architecture_guide.md when scaffolding docs.
- Replace placeholders with project specifics and extend each section as the system evolves.
- Keep the guide aligned with the outline and ADR index; update links when files move.
Placeholders:
- {{PROJECT_NAME}}
- {{VERSION}}
- {{LAST_UPDATED}}
- {{SOURCE_DIR}}
- {{TEST_DIR}}
- {{LANGUAGE}}
- {{TEST_FRAMEWORK}}
- {{LINTER_TOOLS}}
- {{DOCS_DIR}}
-->

# Architecture Guide

**Project:** {{PROJECT_NAME}}
**Version:** {{VERSION}}
**Last Updated:** {{LAST_UPDATED}}

## Overview

This guide captures the architectural principles, patterns, and practices that keep {{PROJECT_NAME}} consistent and maintainable. Extend each section with project-specific details as the system matures.

## 1. Architectural Principles

### Core Principles

1. **Separation of Concerns:** Each module owns a single responsibility.
2. **Explicit Dependencies:** Prefer dependency injection and clear interfaces.
3. **Fail Fast:** Validate inputs early and surface actionable errors.
4. **Testability:** Design for deterministic, isolated tests from the start.

### Design Guidelines

- Prefer composition over inheritance for extensibility.
- Keep interfaces small and focused; document contracts.
- Favor pure functions where possible to reduce side effects.
- Encapsulate external I/O behind adapters to simplify testing.

## 2. System Architecture

### Component Organization

```
{{SOURCE_DIR}}/
├── core/            # Shared types and domain models
├── services/        # Business logic and orchestration
├── api/             # External interfaces (CLI, HTTP, workers)
├── utils/           # Cross-cutting helpers
└── tests/           # See {{TEST_DIR}} for layout
```

### Module Boundaries

| Module | Responsibility | Dependencies |
| --- | --- | --- |
| core | Domain types and validation | None |
| services | Business rules and coordination | core |
| api | Inbound/adaptor layers | services, core |
| utils | Shared utilities | core |

## 3. Design Patterns

Recommended patterns to apply within {{PROJECT_NAME}}:

- **Repository Pattern:** Isolate data access behind interfaces for swapping backends.
- **Strategy Pattern:** Make algorithms interchangeable via injected strategies.
- **Factory Pattern:** Centralize complex object creation for consistency.
- **Observer/Publisher:** Emit events for cross-cutting concerns without tight coupling.

## 4. Anti-Patterns to Avoid

- **God Objects:** Components that accumulate unrelated responsibilities.
- **Circular Dependencies:** Modules that import each other directly.
- **Magic Numbers/Strings:** Unlabeled literals that obscure intent.
- **Silent Failures:** Catching exceptions without logging or handling.

## 5. Data Flow

Describe how data moves through {{PROJECT_NAME}}:

1. Inputs arrive via API/adapters in `{{SOURCE_DIR}}/api/`.
2. Services in `{{SOURCE_DIR}}/services/` coordinate validation and business rules.
3. Domain models in `{{SOURCE_DIR}}/core/` enforce invariants.
4. Outputs return through adapters or persisted via repositories.

## 6. Error Handling

- Use precise exception types with contextual messages.
- Log failures with correlation identifiers where applicable.
- Prefer retry with backoff at integration boundaries.
- Document expected failure modes alongside APIs.

## 7. Testing Architecture

### Test Categories

- **Unit:** Isolated functions and classes.
- **Integration:** Cross-module behaviors and adapters.
- **End-to-End:** Representative workflows across components.

### Test Organization

```
{{TEST_DIR}}/
├── unit/
├── integration/
└── fixtures/
```

- Keep test doubles close to subjects under test.
- Align coverage with documented patterns and module boundaries.

## Related Documentation

- [Architecture Outline](architecture_outline.md)
- [Architecture Decisions](decisions/README.md)
- [Architecture Reference](../architecture_reference.md)
- [Code Style](../code_style.md)
- [Testing Guide](../testing_guide.md)
