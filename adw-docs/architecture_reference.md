<!--
Summary: Architecture reference landing page that points readers to architecture docs, ADRs, and reviews.
Usage:
- Copy into {{DOCS_DIR}}/architecture_reference.md when scaffolding documentation.
- Update the guidance or links if the architecture docs move.
- Keep sections concise (≈100 lines) and focused on navigation.
Placeholders:
- {{PROJECT_NAME}}
- {{VERSION}}
- {{LAST_UPDATED}}
- {{DOCS_DIR}}
-->

# Architecture Reference

**Project:** {{PROJECT_NAME}}
**Version:** {{VERSION}}
**Last Updated:** {{LAST_UPDATED}}

This document is the first stop for anyone looking for architecture resources in {{PROJECT_NAME}}.
It connects contributors with guidance on how to explore the system, make decisions, and keep architectural records healthy.

## Architecture at a Glance

{{DOCS_DIR}}/
├── architecture/
│   ├── architecture_guide.md         # Detailed principles and patterns
│   ├── architecture_outline.md       # High-level system overview
│   └── decisions/                   # ADRs and governance
├── dev-plans/README.md               # Feature/maintenance plan guidance
├── review_guide.md                  # Code review criteria
└── code_style.md                    # Coding conventions

## Navigation

### For New Contributors
1. **[Architecture Outline](architecture/architecture_outline.md)** – Understand the major components and how the repo is organized.
2. **[Architecture Guide](architecture/architecture_guide.md)** – Read the patterns and principles that keep {{PROJECT_NAME}} consistent.
3. **[Architecture Decisions](architecture/decisions/README.md)** – See why past choices were made and what trade-offs were accepted.
4. **[Review Guide](review_guide.md)** – Learn the expectations for authoring and reviewing architectural work.

### For Feature Implementers
1. **[Architecture Guide](architecture/architecture_guide.md)** – Check that your approach matches established patterns before coding.
2. **[Architecture Outline](architecture/architecture_outline.md)** – Locate the correct modules for your feature and understand dependencies.
3. **[Code Style](code_style.md)** – Follow the style rules that keep {{PROJECT_NAME}} readable and consistent.
4. **[Dev Plans](dev-plans/README.md)** – Confirm the feature belongs to an existing epic or maintenance plan and matches its scope.

### For Making Architectural Decisions
1. **[Architecture Guide](architecture/architecture_guide.md)** – Align with the documented principles.
2. **[Architecture Decisions](architecture/decisions/README.md)** – Record the motivation, context, and alternatives for your decision.
3. **[Review Guide](review_guide.md)** – Use the architecture review checklist to ensure the decision is well-scrutinized.
4. **Architecture ADR template** – Author the record from [architecture/decisions/template.md](architecture/decisions/template.md) so formatting and metadata stay consistent.

### For Code Reviews
1. **[Review Guide](review_guide.md)** – Apply the architecture-specific review criteria.
2. **[Architecture Guide](architecture/architecture_guide.md)** – Verify the change adheres to documented patterns.
3. **[Code Style](code_style.md)** – Look for style regressions and formatting issues.
4. **[Architecture Decisions](architecture/decisions/README.md)** – Understand existing ADRs so you can assess the impact of the new change.

## When to Create ADRs

Create an Architecture Decision Record whenever you are making a substantial change to how {{PROJECT_NAME}} is structured or behaves.
Begin with the consistent metadata in [architecture/decisions/template.md](architecture/decisions/template.md) so the context, options, and outcome stay traceable.
The decision should reference the architecture guide, outline, and any relevant review notes.
If the change will be reviewed formally, reference the architecture review checklist from [architecture/decisions/README.md](architecture/decisions/README.md), mention the ADR in that discussion, and request an architecture review via the `/architecture_review` command so reviewers know to evaluate the proposal.

## Maintaining Architecture Documentation

Keep the architecture guide, outline, and ADR index in sync with the living system:

- **Update the guide** when you introduce new architectural patterns, frameworks, or major components.
- **Refresh the outline** when modules are renamed, relocated, or retired so the high-level view stays accurate.
- **Review ADRs** when decisions are superseded and mark them accordingly.
- **Validate links** any time files move; run the scaffold link checks to ensure no relative paths break.
- **Coordinate with dev-plans** when features span multiple epics so implementers know where to look for architecture context.

Treat this file as both a directory and a reminder: architecture documentation thrives when it is easy to find references, understand the rationale, and keep everything connected.
