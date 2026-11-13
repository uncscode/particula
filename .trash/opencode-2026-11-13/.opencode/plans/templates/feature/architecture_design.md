<!-- TEMPLATE: Replace this entire file with your architecture details -->

Describe the high-level design, data/API changes, and security considerations.

**Required subsections:**

### High-Level Design
How does this feature fit into the existing system? Include a flow diagram
or ASCII art showing the data/control flow.

### Data / API / Workflow Changes
- **Data Model:** What models or schemas change?
- **API Surface:** New or modified endpoints, CLI commands, or tool interfaces?
- **Workflow Hooks:** How does this integrate with existing workflows?

### Security & Compliance
Any security reviews, permissions changes, or compliance requirements?

**Example (E16-F6):**

### High-Level Design
```
Scheduler returns AllComplete for accumulate-mode manifest
  -> P1: ship-auto-final workflow (summary handoff only)
    -> shipper-auto-final agent
      -> gather cumulative diff stat
      -> compose implementation summary
      -> persist summary fields for downstream finalization
  -> P2: runtime dispatcher finalization
    -> open_final_pr() / router.create_pr(head=source_branch, base=target_branch)
    -> check idempotency, post safety comment, record "final_pr_opened"
```

### Data / API / Workflow Changes
- **Data Model:** Reads `manifest.source_branch`, `manifest.target_branch`,
  manifest checkpoints, and slice completion states
- **API Surface:** New `ship-auto-final.json` workflow and `shipper-auto-final`
  agent; reuses existing `router.create_pr()` and `router.create_issue_comment()`

### Security & Compliance
The final PR/MR must always require manual merge. No auto-merge or auto-conflict
resolution may be implied by comments or workflow steps.
