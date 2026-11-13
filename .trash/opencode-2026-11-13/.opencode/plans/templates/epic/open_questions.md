<!-- TEMPLATE: Replace this entire file with unresolved questions -->

Track unresolved decisions, scope ambiguities, or intent-alignment questions
that need clarification before or during implementation.

Review agents append pivotal questions here during plan review passes.
The question resolver records evidence-backed answers and adds recommended
multiple-choice responses to decisions requiring human authority. The questions
surfacer posts only the remaining unchecked choices to the planning PR.

**Resolved example:**
- [x] Is the migration path backward-compatible with existing plan artifacts?
  - Resolved 2026-04-01: Preserve the existing artifact schema during migration.
  - Rationale: Persisted plan artifacts are an explicit compatibility boundary.
  - Evidence:
    - `<repo-relative-file>:<line>` - Existing artifacts use the versioned schema.
  - Resolved by: plan-question-resolver

**Unresolved example:**
- [ ] Should child features share one tracking branch or use separate branches?
  - Open: Either topology is supported, and the choice determines review policy.
  - Recommendation: **A - Use separate branches per child feature**
  - Suggested answer: Choose **A** because it keeps feature reviews independently reversible.
  - Options:
    - [ ] A. Use separate branches per child feature (Recommended)
    - [ ] B. Use one shared epic tracking branch for all child features
  - Evidence considered:
    - `<repo-relative-file>:<line>` - Accumulation supports independent slice branches.
