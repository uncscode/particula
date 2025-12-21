<!--
Summary: Epic planning template for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/dev-plans/epics/{{EPIC_ID}}-name.md when starting a new epic.
- Keep this content aligned with {{REPO_URL}}/tree/{{DEFAULT_BRANCH}}/adw-docs/dev-plans/template-epic.md.
Placeholders:
- {{PROJECT_NAME}}
- {{DOCS_DIR}}
- {{REPO_URL}}
- {{DEFAULT_BRANCH}}
- {{LINE_LENGTH}}
- {{COVERAGE_THRESHOLD}}
- {{EPIC_ID}}
- {{EPIC_NAME}}
- {{EPIC_STATUS}}
- {{PRIORITY}}
- {{OWNERS}}
- {{START_DATE}}
- {{TARGET_DATE}}
- {{LAST_UPDATED}}
- {{SIZE}}
-->

# Epic {{EPIC_ID}}: {{EPIC_NAME}}

**Status**: {{EPIC_STATUS}}
**Priority**: {{PRIORITY}}
**Owners**: {{OWNERS}}
**Start Date**: {{START_DATE}}
**Target Date**: {{TARGET_DATE}}
**Last Updated**: {{LAST_UPDATED}}
**Size**: {{SIZE}}

## Vision
Describe how {{EPIC_NAME}} advances {{PROJECT_NAME}}. Keep the narrative focused and within {{LINE_LENGTH}} characters per line.

## Scope
List what is in scope for this epic and call out anything that is intentionally out of scope.

## Dependencies
Mention other plans, teams, or external events that this epic depends on.

## Phase Checklist
- [ ] Phase 1 (`{{EPIC_ID}}-P1`) — Describe the first deliverable.
- [ ] Phase 2 (`{{EPIC_ID}}-P2`) — Outline the follow-up work.
- [ ] Phase 3 (`{{EPIC_ID}}-P3`) — Document additional phases as needed; add `E*-P*` IDs inline.

## Critical Testing Requirements
- **No Coverage Modifications**: Keep coverage thresholds as shipped ({{COVERAGE_THRESHOLD}}%).
- **Self-Contained Tests**: Attach `*_test.py` files that prove the epic is complete.
- **Test-First Completion**: Write and pass tests before declaring the epic ready for review.
- **80%+ Coverage**: Every phase must ship tests that maintain at least {{COVERAGE_THRESHOLD}}% coverage.

## Testing Strategy
- Include the specific `{{TEST_DIR}}/**/*_test.py` suites that verify this epic.
- Note any fixtures or harnesses needed for the tests.
- Highlight which teams or agents own each test to keep coverage ownership clear.

## Additional Notes
- Always update the relevant `index.md` in {{DOCS_DIR}}/dev-plans/epics/ after creating or modifying this document.
- Merge and ship this file from {{DEFAULT_BRANCH}} once all checkboxes and tests are complete.
