<!-- TEMPLATE: Replace this entire file with unresolved questions -->

Track unresolved decisions, research items, or ambiguities that need
clarification before or during implementation. The question resolver checks
evidence-backed answers and adds recommended choices to human decisions.

**Resolved example:**
- [x] How should the handoff comment handle repositories with no PR template?
  - Resolved 2026-03-29: Post a standalone comment; do not depend on PR templates.
  - Rationale: Existing platform routing already supports standalone comments.
  - Evidence:
    - `<repo-relative-file>:<line>` - Comment routing does not require a PR template.
  - Resolved by: plan-question-resolver

**Unresolved example:**
- [ ] Should protected-branch PRs be created as drafts?
  - Open: Repository behavior cannot establish the desired release policy.
  - Recommendation: **A - Create protected-branch PRs as drafts**
  - Suggested answer: Choose **A** because it preserves a maintainer review gate.
  - Options:
    - [ ] A. Create protected-branch PRs as drafts (Recommended)
    - [ ] B. Create ready-for-review PRs and rely on branch protection
  - Evidence considered:
    - `<repo-relative-file>:<line>` - The platform API supports either draft state.
