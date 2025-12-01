# Feature: {{FEATURE_NAME}}

**Status:** {{STATUS}}                    # Todo, In Progress, Done, Blocked
**Priority:** {{PRIORITY}}                # P0 (Critical), P1 (High), P2 (Medium), P3 (Low)
**Assignees:** {{ASSIGNEES}}              # @username1, @username2
**Labels:** {{LABELS}}                    # feature, enhancement, new-capability, etc.
**Milestone:** {{MILESTONE}}              # v2.1.0, Q1-2025, etc.
**Size:** {{SIZE}}                        # XS (~10 LOC), S (~50 LOC), M (~100 LOC), L (~500 LOC), XL (~1000 LOC)

**Start Date:** {{START_DATE}}            # YYYY-MM-DD
**Target Date:** {{TARGET_DATE}}          # YYYY-MM-DD
**Created:** {{CREATED_DATE}}             # YYYY-MM-DD
**Updated:** {{UPDATED_DATE}}             # YYYY-MM-DD

**Related Issues:** {{ISSUE_LINKS}}       # #123, #456
**Related PRs:** {{PR_LINKS}}             # #789, #101
**Related ADRs:** {{ADR_LINKS}}           # ADR-001, ADR-002

---

## Overview

{{FEATURE_OVERVIEW_DESCRIPTION}}

### Problem Statement

{{PROBLEM_STATEMENT}}

### Value Proposition

{{VALUE_PROPOSITION}}

## Phases

> **Philosophy:** Each phase should be one GitHub issue with a focused scope (~100 lines of code or less, excluding tests/docs). Small, focused changes make reviews smooth and safe. Smooth is safe, and safe is fast.
>
> **Note:** All phases are documented in this single file. For single-phase work, use only Phase 1. For multi-phase work, list all phases here with their implementation details in separate sections below.

- [ ] **Phase 1:** {{PHASE_1_NAME}} - {{PHASE_1_DESCRIPTION}}
  - GitHub Issue: #{{ISSUE_NUMBER_1}} (if created)
  - Status: {{PHASE_1_STATUS}}
  - Size: {{PHASE_1_SIZE}} (~{{PHASE_1_LOC}} lines of code)
  - Dependency: None
  - Estimated Effort: {{PHASE_1_EFFORT}}

<!-- For multi-phase features, add additional phases below. For single-phase, remove Phase 2 and Phase 3. -->
- [ ] **Phase 2:** {{PHASE_2_NAME}} - {{PHASE_2_DESCRIPTION}}
  - GitHub Issue: #{{ISSUE_NUMBER_2}} (if created)
  - Status: {{PHASE_2_STATUS}}
  - Size: {{PHASE_2_SIZE}} (~{{PHASE_2_LOC}} lines of code)
  - Dependency: Requires Phase 1 completion (#{{ISSUE_NUMBER_1}})
  - Estimated Effort: {{PHASE_2_EFFORT}}

- [ ] **Phase 3:** {{PHASE_3_NAME}} - {{PHASE_3_DESCRIPTION}}
  - GitHub Issue: #{{ISSUE_NUMBER_3}} (if created)
  - Status: {{PHASE_3_STATUS}}
  - Size: {{PHASE_3_SIZE}} (~{{PHASE_3_LOC}} lines of code)
  - Dependency: Requires Phase 2 completion (#{{ISSUE_NUMBER_2}})
  - Estimated Effort: {{PHASE_3_EFFORT}}

## User Stories

### Story 1: {{USER_STORY_1_TITLE}}
**As a** {{USER_TYPE_1}}
**I want** {{USER_WANT_1}}
**So that** {{USER_BENEFIT_1}}

**Acceptance Criteria:**
- [ ] {{ACCEPTANCE_1_1}}
- [ ] {{ACCEPTANCE_1_2}}
- [ ] {{ACCEPTANCE_1_3}}

### Story 2: {{USER_STORY_2_TITLE}}
**As a** {{USER_TYPE_2}}
**I want** {{USER_WANT_2}}
**So that** {{USER_BENEFIT_2}}

**Acceptance Criteria:**
- [ ] {{ACCEPTANCE_2_1}}
- [ ] {{ACCEPTANCE_2_2}}
- [ ] {{ACCEPTANCE_2_3}}

## Technical Approach

### Architecture Changes

{{ARCHITECTURE_CHANGES_DESCRIPTION}}

**Affected Components:**
- {{COMPONENT_1}} - {{COMPONENT_1_CHANGES}}
- {{COMPONENT_2}} - {{COMPONENT_2_CHANGES}}
- {{COMPONENT_3}} - {{COMPONENT_3_CHANGES}}

### Design Patterns

{{DESIGN_PATTERNS_TO_USE}}

### Data Model Changes

{{DATA_MODEL_CHANGES}}

```{{PRIMARY_LANGUAGE}}
{{DATA_MODEL_EXAMPLE}}
```

### API Changes

{{API_CHANGES_DESCRIPTION}}

**New Endpoints:**
```
{{NEW_API_ENDPOINT_1}}
{{NEW_API_ENDPOINT_2}}
```

**Modified Endpoints:**
```
{{MODIFIED_API_ENDPOINT_1}}
{{MODIFIED_API_ENDPOINT_2}}
```

## Implementation Tasks

> For single-phase features, list all tasks here. For multi-phase features, provide high-level tasks and link to phase-specific files for detailed tasks.

### Backend Tasks
- [ ] {{BACKEND_TASK_1}}
- [ ] {{BACKEND_TASK_2}}
- [ ] {{BACKEND_TASK_3}}

**Estimated Effort:** {{BACKEND_EFFORT_ESTIMATE}}

### Frontend Tasks
- [ ] {{FRONTEND_TASK_1}}
- [ ] {{FRONTEND_TASK_2}}
- [ ] {{FRONTEND_TASK_3}}

**Estimated Effort:** {{FRONTEND_EFFORT_ESTIMATE}}

### Database Tasks
- [ ] {{DATABASE_TASK_1}}
- [ ] {{DATABASE_TASK_2}}
- [ ] {{DATABASE_TASK_3}}

**Estimated Effort:** {{DATABASE_EFFORT_ESTIMATE}}

### Infrastructure Tasks
- [ ] {{INFRA_TASK_1}}
- [ ] {{INFRA_TASK_2}}

**Estimated Effort:** {{INFRA_EFFORT_ESTIMATE}}

## Dependencies

### Upstream Dependencies
- {{DEPENDENCY_1}} - {{DEPENDENCY_1_DESCRIPTION}}
- {{DEPENDENCY_2}} - {{DEPENDENCY_2_DESCRIPTION}}

### Downstream Dependencies
- {{DEPENDENT_1}} - {{DEPENDENT_1_DESCRIPTION}}
- {{DEPENDENT_2}} - {{DEPENDENT_2_DESCRIPTION}}

### External Dependencies
- {{EXTERNAL_DEPENDENCY_1}} - {{EXTERNAL_DEPENDENCY_1_DESCRIPTION}}
- {{EXTERNAL_DEPENDENCY_2}} - {{EXTERNAL_DEPENDENCY_2_DESCRIPTION}}

## Blockers

- [ ] {{BLOCKER_1}} - Status: {{BLOCKER_1_STATUS}}
- [ ] {{BLOCKER_2}} - Status: {{BLOCKER_2_STATUS}}

## Testing Strategy

### Unit Tests
{{UNIT_TEST_STRATEGY}}

**Test Cases:**
- [ ] {{UNIT_TEST_1}}
- [ ] {{UNIT_TEST_2}}
- [ ] {{UNIT_TEST_3}}

### Integration Tests
{{INTEGRATION_TEST_STRATEGY}}

**Test Cases:**
- [ ] {{INTEGRATION_TEST_1}}
- [ ] {{INTEGRATION_TEST_2}}
- [ ] {{INTEGRATION_TEST_3}}

### End-to-End Tests
{{E2E_TEST_STRATEGY}}

**Test Scenarios:**
- [ ] {{E2E_TEST_1}}
- [ ] {{E2E_TEST_2}}
- [ ] {{E2E_TEST_3}}

### Performance Tests
{{PERFORMANCE_TEST_STRATEGY}}

**Metrics to Track:**
- {{PERFORMANCE_METRIC_1}}: Target {{PERFORMANCE_TARGET_1}}
- {{PERFORMANCE_METRIC_2}}: Target {{PERFORMANCE_TARGET_2}}

## Documentation

- [ ] Update architecture guide
- [ ] Update API documentation
- [ ] Update user guide
- [ ] Create migration guide (if applicable)
- [ ] Update {{ADDITIONAL_DOC_1}}
- [ ] Update {{ADDITIONAL_DOC_2}}

## Security Considerations

{{SECURITY_CONSIDERATIONS}}

**Security Checklist:**
- [ ] {{SECURITY_CHECK_1}}
- [ ] {{SECURITY_CHECK_2}}
- [ ] {{SECURITY_CHECK_3}}

## Performance Considerations

{{PERFORMANCE_CONSIDERATIONS}}

**Performance Targets:**
- {{PERF_TARGET_1}}
- {{PERF_TARGET_2}}

## Rollout Strategy

### Deployment Plan
{{DEPLOYMENT_PLAN}}

### Feature Flags
- {{FEATURE_FLAG_1}} - {{FEATURE_FLAG_1_PURPOSE}}
- {{FEATURE_FLAG_2}} - {{FEATURE_FLAG_2_PURPOSE}}

### Rollback Plan
{{ROLLBACK_PLAN}}

## Success Criteria

- [ ] {{SUCCESS_CRITERION_1}}
- [ ] {{SUCCESS_CRITERION_2}}
- [ ] {{SUCCESS_CRITERION_3}}
- [ ] All tests passing
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Performance targets met
- [ ] Security review passed

## Metrics to Track

- {{METRIC_1}}: Baseline {{BASELINE_1}}, Target {{TARGET_1}}
- {{METRIC_2}}: Baseline {{BASELINE_2}}, Target {{TARGET_2}}
- {{METRIC_3}}: Baseline {{BASELINE_3}}, Target {{TARGET_3}}

## Timeline

| Phase/Milestone | Start Date | Target Date | Actual Date | Status |
|----------------|------------|-------------|-------------|--------|
| {{MILESTONE_1}} | {{START_1}} | {{TARGET_1}} | {{ACTUAL_1}} | {{STATUS_1}} |
| {{MILESTONE_2}} | {{START_2}} | {{TARGET_2}} | {{ACTUAL_2}} | {{STATUS_2}} |
| {{MILESTONE_3}} | {{START_3}} | {{TARGET_3}} | {{ACTUAL_3}} | {{STATUS_3}} |

## Open Questions

- [ ] {{QUESTION_1}}
- [ ] {{QUESTION_2}}
- [ ] {{QUESTION_3}}

## Notes

{{ADDITIONAL_NOTES}}

## Change Log

| Date | Change | Author |
|------|--------|--------|
| {{LOG_DATE_1}} | {{LOG_CHANGE_1}} | {{LOG_AUTHOR_1}} |
| {{LOG_DATE_2}} | {{LOG_CHANGE_2}} | {{LOG_AUTHOR_2}} |
| {{LOG_DATE_3}} | {{LOG_CHANGE_3}} | {{LOG_AUTHOR_3}} |
