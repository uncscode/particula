<!--
Summary: Maintenance planning template for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/dev-plans/maintenance/{{MAINTENANCE_ID}}-name.md when documenting recurring work.
- Keep this content aligned with {{REPO_URL}}/tree/{{DEFAULT_BRANCH}}/adw-docs/dev-plans/template-maintenance.md.
Placeholders:
- {{PROJECT_NAME}}
- {{DOCS_DIR}}
- {{REPO_URL}}
- {{DEFAULT_BRANCH}}
- {{LINE_LENGTH}}
- {{COVERAGE_THRESHOLD}}
- {{MAINTENANCE_ID}}
- {{MAINTENANCE_AREA}}
- {{MAINTENANCE_STATUS}}
- {{PRIORITY}}
- {{OWNERS}}
- {{TARGET_DATE}}
- {{LAST_UPDATED}}
- {{SIZE}}
-->

# Maintenance {{MAINTENANCE_ID}}: {{MAINTENANCE_AREA}}

**Status**: {{MAINTENANCE_STATUS}}
**Priority**: {{PRIORITY}}
**Owners**: {{OWNERS}}
**Target Date**: {{TARGET_DATE}}
**Last Updated**: {{LAST_UPDATED}}
**Size**: {{SIZE}}

## Vision
Explain why this maintenance capability matters for {{PROJECT_NAME}} and what the expected outcome is.

## Scope
Clarify the systems, services, or docs this maintenance plan touches.

## Dependencies
List any projects, observability signals, or calendars that influence the timeline.

## Phase Checklist
- [ ] Phase 1 (`{{MAINTENANCE_ID}}-P1`) — Describe what happens during the first execution.
- [ ] Phase 2 (`{{MAINTENANCE_ID}}-P2`) — Outline follow-up checks or automation tuning.
- [ ] Phase 3 (`{{MAINTENANCE_ID}}-P3`) — Add steps for documenting lessons learned or updating dashboards.

## Critical Testing Requirements
- **No Coverage Modifications**: Never drop the threshold below {{COVERAGE_THRESHOLD}}% to ship maintenance fixes.
- **Self-Contained Tests**: Include `*_test.py` files that prove each maintenance change behaves as expected.
- **Test-First Completion**: Write tests or signals before marking maintenance updates as complete.
- **80%+ Coverage**: Every runnable artifact must reach {{COVERAGE_THRESHOLD}}% coverage before shipping.

## Testing Strategy
- Reference the `{{TEST_DIR}}/**/*_test.py` suites (unit, integration, regression) that validate this maintenance work.
- Describe how test data is refreshed and whether the plan runs on a schedule or manually.

## Recurrence Notes
1. Keep phase descriptions under {{LINE_LENGTH}} characters per line for readability.
2. Update the `maintenance/index.md` entry whenever the cadence or scope changes.
3. Merge and release from {{DEFAULT_BRANCH}} with links back to this template.
