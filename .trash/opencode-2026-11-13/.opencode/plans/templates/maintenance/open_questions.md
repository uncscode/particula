<!-- TEMPLATE: Replace this entire file with unresolved questions -->

Track unresolved decisions, scope ambiguities, or intent-alignment questions
that need clarification before or during implementation.

Review agents append pivotal questions here during plan review passes.
The question resolver records evidence-backed answers and adds recommended
multiple-choice responses to decisions requiring human authority. The questions
surfacer posts only the remaining unchecked choices to the planning PR.

**Resolved example:**
- [x] Should deprecated wrappers be removed immediately or marked first?
  - Resolved 2026-04-01: Mark them first and remove them in the next maintenance cycle.
  - Rationale: The repository migration policy preserves compatibility surfaces temporarily.
  - Evidence:
    - `<repo-relative-file>:<line>` - Compatibility wrappers remain active during migration.
  - Resolved by: plan-question-resolver

**Unresolved example:**
- [ ] What downtime window is acceptable for schema migrations?
  - Open: Technical evidence cannot determine the operator's release window.
  - Recommendation: **A - Require a zero-downtime migration**
  - Suggested answer: Choose **A** because it avoids coordinating an outage across users.
  - Options:
    - [ ] A. Require a zero-downtime migration (Recommended)
    - [ ] B. Allow a scheduled maintenance window of up to 15 minutes
  - Evidence considered:
    - No decisive repository precedent found.
