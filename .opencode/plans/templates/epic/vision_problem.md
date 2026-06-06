<!-- TEMPLATE: Replace this entire file with the epic's vision -->

Describe the problem this epic solves at the program level. This is the
north-star narrative that motivates all child features and maintenance tracks.

**Required elements:**
- What problems exist today (numbered list)
- The vision: what the world looks like after this epic ships
- Why now: what makes this urgent or important

**Example (E17):**
Development plans currently live as monolithic markdown files that combine
structured metadata with long-form prose. This creates several problems:

1. **No programmatic querying** -- finding all P1 active features requires
   grep across dozens of files and fragile parsing.
2. **Duplicated index maintenance** -- index files must be hand-updated every
   time a plan changes status.
3. **No validation** -- metadata like status transitions, phase IDs, and
   dependency cycles are not validated.
4. **No runtime analytics** -- there is no structured place to record actual
   cost, duration, or token usage per workflow run.

### The Vision
Replace the markdown-only system with a two-layer structured model:
- **Layer 1 (canonical source):** One JSON file per plan with structured
  metadata, validated by JSON Schema and Pydantic models.
- **Layer 2 (generated views):** Markdown indexes and rendered plan pages,
  all generated from Layer 1 and never hand-edited.
