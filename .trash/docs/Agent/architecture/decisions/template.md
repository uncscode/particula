# ADR-{{NUMBER}}: {{TITLE}}

**Status:** {{STATUS}}
**Date:** {{DATE}}
**Decision Makers:** {{DECISION_MAKERS}}
**Technical Story:** {{ISSUE_LINK}}

## Context

{{CONTEXT_DESCRIPTION}}

### Problem Statement

{{PROBLEM_STATEMENT}}

### Forces

{{FORCES_DESCRIPTION}}

**Driving Forces:**
- {{DRIVING_FORCE_1}}
- {{DRIVING_FORCE_2}}
- {{DRIVING_FORCE_3}}

**Restraining Forces:**
- {{RESTRAINING_FORCE_1}}
- {{RESTRAINING_FORCE_2}}
- {{RESTRAINING_FORCE_3}}

## Decision

{{DECISION_DESCRIPTION}}

### Chosen Option

**Option {{CHOSEN_OPTION_NUMBER}}: {{CHOSEN_OPTION_NAME}}**

{{CHOSEN_OPTION_DESCRIPTION}}

## Alternatives Considered

### Option 1: {{OPTION_1_NAME}}

**Description:** {{OPTION_1_DESCRIPTION}}

**Pros:**
- {{OPTION_1_PRO_1}}
- {{OPTION_1_PRO_2}}
- {{OPTION_1_PRO_3}}

**Cons:**
- {{OPTION_1_CON_1}}
- {{OPTION_1_CON_2}}
- {{OPTION_1_CON_3}}

**Reason for Rejection:** {{OPTION_1_REJECTION_REASON}}

---

### Option 2: {{OPTION_2_NAME}}

**Description:** {{OPTION_2_DESCRIPTION}}

**Pros:**
- {{OPTION_2_PRO_1}}
- {{OPTION_2_PRO_2}}
- {{OPTION_2_PRO_3}}

**Cons:**
- {{OPTION_2_CON_1}}
- {{OPTION_2_CON_2}}
- {{OPTION_2_CON_3}}

**Reason for Rejection:** {{OPTION_2_REJECTION_REASON}}

---

### Option 3: {{OPTION_3_NAME}}

**Description:** {{OPTION_3_DESCRIPTION}}

**Pros:**
- {{OPTION_3_PRO_1}}
- {{OPTION_3_PRO_2}}
- {{OPTION_3_PRO_3}}

**Cons:**
- {{OPTION_3_CON_1}}
- {{OPTION_3_CON_2}}
- {{OPTION_3_CON_3}}

**Reason for Rejection:** {{OPTION_3_REJECTION_REASON}}

---

## Rationale

{{DECISION_RATIONALE}}

### Why This Approach?

{{WHY_THIS_APPROACH}}

### Trade-offs Accepted

1. **{{TRADEOFF_1}}**: {{TRADEOFF_1_DESCRIPTION}}
2. **{{TRADEOFF_2}}**: {{TRADEOFF_2_DESCRIPTION}}
3. **{{TRADEOFF_3}}**: {{TRADEOFF_3_DESCRIPTION}}

## Consequences

### Positive

- {{POSITIVE_CONSEQUENCE_1}}
- {{POSITIVE_CONSEQUENCE_2}}
- {{POSITIVE_CONSEQUENCE_3}}

### Negative

- {{NEGATIVE_CONSEQUENCE_1}}
- {{NEGATIVE_CONSEQUENCE_2}}
- {{NEGATIVE_CONSEQUENCE_3}}

### Neutral

- {{NEUTRAL_CONSEQUENCE_1}}
- {{NEUTRAL_CONSEQUENCE_2}}

## Implementation

### Required Changes

1. **{{CHANGE_AREA_1}}**
   - {{CHANGE_1_DESCRIPTION}}
   - Affected files: {{CHANGE_1_FILES}}
   - Estimated effort: {{CHANGE_1_EFFORT}}

2. **{{CHANGE_AREA_2}}**
   - {{CHANGE_2_DESCRIPTION}}
   - Affected files: {{CHANGE_2_FILES}}
   - Estimated effort: {{CHANGE_2_EFFORT}}

3. **{{CHANGE_AREA_3}}**
   - {{CHANGE_3_DESCRIPTION}}
   - Affected files: {{CHANGE_3_FILES}}
   - Estimated effort: {{CHANGE_3_EFFORT}}

### Migration Plan

{{MIGRATION_PLAN}}

**Steps:**
1. {{MIGRATION_STEP_1}}
2. {{MIGRATION_STEP_2}}
3. {{MIGRATION_STEP_3}}
4. {{MIGRATION_STEP_4}}

### Testing Strategy

{{TESTING_STRATEGY}}

### Rollback Plan

{{ROLLBACK_PLAN}}

## Validation

### Success Criteria

- [ ] {{SUCCESS_CRITERION_1}}
- [ ] {{SUCCESS_CRITERION_2}}
- [ ] {{SUCCESS_CRITERION_3}}

### Metrics

- **{{METRIC_1}}**: {{METRIC_1_TARGET}}
- **{{METRIC_2}}**: {{METRIC_2_TARGET}}
- **{{METRIC_3}}**: {{METRIC_3_TARGET}}

## References

### Related ADRs

- [ADR-{{RELATED_ADR_1}}]({{RELATED_ADR_1_NUMBER}}-{{RELATED_ADR_1_TITLE}}.md): {{RELATED_ADR_1_SUMMARY}}
- [ADR-{{RELATED_ADR_2}}]({{RELATED_ADR_2_NUMBER}}-{{RELATED_ADR_2_TITLE}}.md): {{RELATED_ADR_2_SUMMARY}}

### External References

- {{EXTERNAL_REFERENCE_1}}
- {{EXTERNAL_REFERENCE_2}}
- {{EXTERNAL_REFERENCE_3}}

### Documentation Updates

Files requiring updates after implementation:
- [ ] `{{DOCUMENTATION_FILE_1}}`
- [ ] `{{DOCUMENTATION_FILE_2}}`
- [ ] `{{DOCUMENTATION_FILE_3}}`

## Notes

{{ADDITIONAL_NOTES}}

---

## Status Values

- **Proposed**: Decision under consideration
- **Accepted**: Decision approved and ready for implementation
- **Superseded**: Replaced by another decision (link to new ADR)
- **Deprecated**: No longer recommended but still in use
- **Rejected**: Decision was not accepted

## ADR Template Usage

When creating a new ADR:

1. Copy this template to `decisions/NNN-title.md` where NNN is the next sequential number (e.g., 001, 002, 003)
2. Replace all `{{PLACEHOLDERS}}` with actual content
3. Remove sections that don't apply (but keep major structure)
4. Update the Architecture Guide to reference the new ADR
5. Add the ADR to the decisions index/README if one exists
6. Link related ADRs bidirectionally
