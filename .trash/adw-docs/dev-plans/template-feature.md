<!--
Summary: Feature planning template for {{PROJECT_NAME}}.
Usage:
- Copy into {{DOCS_DIR}}/dev-plans/features/{{FEATURE_ID}}-name.md after selecting the next feature ID.
- Keep this template synced with {{REPO_URL}}/tree/{{DEFAULT_BRANCH}}/adw-docs/dev-plans/template-feature.md.
Placeholders:
- {{PROJECT_NAME}}
- {{DOCS_DIR}}
- {{REPO_URL}}
- {{DEFAULT_BRANCH}}
- {{LINE_LENGTH}}
- {{COVERAGE_THRESHOLD}}
- {{FEATURE_ID}}
- {{FEATURE_NAME}}
- {{FEATURE_STATUS}}
- {{PRIORITY}}
- {{OWNERS}}
- {{START_DATE}}
- {{TARGET_DATE}}
- {{LAST_UPDATED}}
- {{SIZE}}
-->

# Feature {{FEATURE_ID}}: {{FEATURE_NAME}}

**Status**: {{FEATURE_STATUS}}
**Priority**: {{PRIORITY}}
**Owners**: {{OWNERS}}
**Start Date**: {{START_DATE}}
**Target Date**: {{TARGET_DATE}}
**Last Updated**: {{LAST_UPDATED}}
**Size**: {{SIZE}}

## Vision
Explain why {{FEATURE_NAME}} matters for {{PROJECT_NAME}} and how it fits the roadmap.

## Scope
Detail what this feature will include and explicitly call out what is excluded.

## Dependencies
List other epics, features, or maintenance plans that must land first.

## Phase Checklist
- [ ] Phase 1 (`{{FEATURE_ID}}-P1`) — Summarize the first deliverable.
- [ ] Phase 2 (`{{FEATURE_ID}}-P2`) — Define follow-up work.
- [ ] Phase 3 (`{{FEATURE_ID}}-P3`) — Add or remove phases as needed to cover all planned work.

## Critical Testing Requirements
- **No Coverage Modifications**: Keep the threshold at {{COVERAGE_THRESHOLD}}% or higher.
- **Self-Contained Tests**: Ship `*_test.py` suites that prove the feature works end-to-end.
- **Test-First Completion**: Tests must exist and pass before finishing each phase.
- **80%+ Coverage**: Maintain at least {{COVERAGE_THRESHOLD}}% coverage for touched code.

## Testing Strategy
- Point reviewers to the exact `{{TEST_DIR}}/**/*_test.py` files that verify this feature.
- Document any new fixtures, mocks, or data pipelines used for coverage.
- Confirm tests remain fast (≤1 second each) to support rapid verification.

## Shipping Checklist
1. Update `{{DOCS_DIR}}/dev-plans/features/index.md` with the new entry.
2. Link the feature to issues following the `[{{FEATURE_ID}}-P*]` convention.
3. After all tests pass, merge from {{DEFAULT_BRANCH}} with a PR that references the plan.
