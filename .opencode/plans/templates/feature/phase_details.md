<!-- TEMPLATE: Replace this entire file with your phase checklist -->

List every phase as a checklist item. Each phase should be one reviewable PR
(~100 LOC of production code). Pack related work into a single phase rather
than splitting too finely -- only split when a feature exceeds 15-20 phases.

The final phase should always be "Update development documentation".

**Format per phase:**
- [ ] **{PLAN_ID}-P{n}:** {Phase title}
  - Issue: TBD | Size: {XS|S|M} | Status: Not Started
  - Goal: {One sentence describing the deliverable}
  - Files: {Key files to create or modify}
  - Tests: {What tests to add}

**Example (E16-F6):**
- [x] **E16-F6-P1:** Create `ship-auto-final` workflow and agent with tests
  - Issue: #2120 | Size: S | Status: Shipped
  - Goal: Ship workflow JSON definition and agent markdown with schema tests
  - Files: `.opencode/workflow/ship-auto-final.json`, `.opencode/agent/shipper-auto-final.md`
  - Tests: Workflow schema validation, agent instruction parsing

- [x] **E16-F6-P2:** Wire final PR/MR from the tracking feature branch with tests
  - Issue: #2122 | Size: S | Status: Shipped
  - Goal: Runtime creates one PR from source_branch to target_branch with idempotency
  - Files: `adw/automode/scheduler.py`, `adw/platforms/router.py`
  - Tests: PR creation, duplicate prevention, branch targeting

- [x] **E16-F6-P3:** Add final handoff comments and non-automation guardrails
  - Issue: #2124 | Size: S | Status: Shipped
  - Goal: Post deterministic handoff comment with manual-review requirements
  - Tests: Comment content, duplicate comment prevention

- [x] **E16-F6-P4:** Update development documentation
  - Issue: #2126 | Size: XS | Status: Shipped
  - Goal: Update runbook, agent docs, and workflow reference
