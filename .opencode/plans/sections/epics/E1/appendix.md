<!-- TEMPLATE: Replace this entire file with supporting material -->

Link supporting documents, ADRs, workflow definitions, dashboards, diagrams,
or alternative approaches that were considered and rejected.

**Example (E17):**

### Alternative Approaches Considered
- **Dolt:** Version-controlled SQL database. Rejected -- introduces a second
  VCS model inside Git, GitHub/GitLab cannot review Dolt data natively.
- **SQLite as primary store:** Rejected -- binary files create poor Git diffs,
  merge conflicts are unresolvable.
- **YAML instead of JSON:** Rejected -- JSON has stricter parsing and better
  validation tooling.
- **Monolithic single JSON file:** Rejected -- merge conflicts scale with
  number of plans, diffs become unreadable.

### Key Design Decisions
1. JSON for structured metadata, markdown for narrative prose
2. One JSON file per plan, never moved on completion
3. Lifecycle model separates execution status from active/completed/closed grouping
4. Runtime analytics stored in separate JSONL, not in plan JSON
