<!--
Summary: Dev plan scaffolding overview for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/dev-plans/README.md when scaffolding documentation.
- Keep the quick links, structure, and testing expectations aligned with {{REPO_URL}}/tree/{{DEFAULT_BRANCH}}/adw-docs/dev-plans.
Placeholders:
- {{PROJECT_NAME}}
- {{DOCS_DIR}}
- {{REPO_URL}}
- {{DEFAULT_BRANCH}}
- {{LINE_LENGTH}}
- {{COVERAGE_THRESHOLD}}
-->

# Dev plans for {{PROJECT_NAME}}

This dev-plans/ scaffold mirrors the published guidance so downstream repos can drop in ready-to-use templates and keep their planning culture aligned with the primary docs set.

## Quick Links

| Type | Template | Destination |
|------|----------|-------------|
| **Epics** | `template-epic.md` | `epics/` and `epics/.gitkeep` |
| **Features** | `template-feature.md` | `features/` and `features/.gitkeep` |
| **Maintenance** | `template-maintenance.md` | `maintenance/` and `maintenance/.gitkeep` |

## Folder Structure

```
adw-docs/dev-plans/
├── README.md                      # You are here (overview & conventions)
├── template-epic.md               # Epic planning template
├── template-feature.md            # Feature planning template
├── template-maintenance.md        # Maintenance planning template
├── epics/                         # New plans inherit E* IDs
│   └── .gitkeep
├── features/                      # Lettered F* documents
│   └── .gitkeep
└── maintenance/                   # Long-lived health work
    └── .gitkeep
```

## ID Naming Convention

- **Epics** (`E1`, `E2`, ...): files live in `epics/` and include phases like `E1-P1`.
- **Features** (`E1-F1`, `F2`, ...): place linked or standalone plans in `features/` with matching IDs.
- **Maintenance** (`E1-M1`, `M2`, ...): maintain evergreen checks in `maintenance/` with hierarchical IDs.

## Decision Guide

1. **Multi-release coordination** → Start an **epic** in `epics/`.
2. **User-facing enhancement** → Author a **feature** plan in `features/`.
3. **Recurring health work** → Draft a **maintenance** plan in `maintenance/`.
4. **Unsure** → Begin with a feature or maintenance entry and link it to an epic later.

## Working With Templates

1. Copy the template that matches your plan type and place it under `{{DOCS_DIR}}/dev-plans/`.
2. Assign the next available ID from the matching `index.md` (epics, features, or maintenance) and update the metadata block accordingly.
3. Complete every placeholder, including phase IDs (e.g., `E1-P2`) and dates, while keeping prose within {{LINE_LENGTH}} characters per line.
4. Run all tests locally before marking the plan ready, and always add “Update dev-docs” as the final phase.
5. Update the relevant `index.md` entry after adding or finishing a plan.

## Critical Testing Requirements

- **No Coverage Modifications**: Never lower the existing coverage threshold ({{COVERAGE_THRESHOLD}}%) to make tests pass.
- **Self-Contained Tests**: Each phase must ship with its own `*_test.py` suites that exercise the new behavior.
- **Test-First Completion**: Write tests before the implementation and guard completion until they pass.
- **80%+ Coverage**: Every phase must pass at least {{COVERAGE_THRESHOLD}}% coverage on the lines it touches.

## Scaffolding Notes

Drop completed plans into the `epics/`, `features/`, or `maintenance/` directories; the `.gitkeep` files keep the directories present for downstream tooling. Keep this README aligned with the published version at {{REPO_URL}}/tree/{{DEFAULT_BRANCH}}/adw-docs/dev-plans to keep contributors aware of the quick links and testing expectations.
