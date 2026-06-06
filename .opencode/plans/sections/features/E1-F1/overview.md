<!-- TEMPLATE: Replace this entire file with your feature's overview -->

Summarize the problem this feature solves and why it matters.

**Required elements:**
- **Problem Statement:** What specific pain point does this address?
- **Value Proposition:** What does the user/system gain?
- **User Stories:** 1-3 "As a... I want... so that..." statements

**Example (E16-F6):**
- **Problem Statement:** Once E16 removes per-slice PRs in accumulate mode,
  final completion needs a dedicated handoff step that prepares the cumulative
  implementation summary and creates the one allowed PR/MR.
- **Value Proposition:** `ship-auto-final` isolates final-handoff concerns from
  normal slice shipping and makes the manual merge boundary explicit.
- **User Stories:**
  - As an operator, I want one final PR/MR at manifest completion so I can review
    the accumulated feature branch as a single unit.
