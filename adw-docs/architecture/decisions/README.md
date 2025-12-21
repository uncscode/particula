<!--
Summary: ADR index stub explaining purpose, statuses, and how to author decision records for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/architecture/decisions/README.md when scaffolding documentation.
- Update the index table as ADRs are created and keep statuses in sync with records.
- Link new ADRs back into the architecture guide or outline when decisions affect those docs.
Placeholders:
- {{PROJECT_NAME}}
- {{DOCS_DIR}}
- {{DATE}}
-->

# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for {{PROJECT_NAME}}.

## What is an ADR?

An ADR documents a significant architectural decision, its context, alternatives, and consequences so the reasoning stays discoverable over time.

## ADR Index

| ID | Title | Status | Date |
| --- | --- | --- | --- |
| ADR-001 | [Example Decision](ADR-001-example.md) | Proposed | {{DATE}} |

Add a new row for each ADR and keep the table sorted by ID.

## Creating a New ADR

1. Copy `template.md` to `ADR-XXX-short-title.md` (replace `XXX` with the next number).
2. Fill in all sections and replace placeholders.
3. Set **Status** to `Proposed`, `Accepted`, `Superseded`, or `Deprecated`.
4. Link the ADR from the architecture guide/outline where relevant.
5. Update the index above with the new record.

## ADR Statuses

- **Proposed:** Under review and discussion.
- **Accepted:** Approved and implemented.
- **Superseded:** Replaced by a newer decision (link both ways).
- **Deprecated:** No longer recommended but retained for history.

## References

- [ADR Template](template.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
