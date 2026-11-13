<!-- TEMPLATE: Replace this entire file with your feature's scope -->

One paragraph describing what this feature delivers, followed by explicit
in-scope and out-of-scope lists.

**Required elements:**
- Brief summary of the deliverable
- **In scope:** Concrete items being built or changed
- **Out of scope:** Items explicitly excluded to prevent scope creep

**Example (E16-F6):**
Add `ship-auto-final` workflow/agent for deterministic summary handoff, then
wire runtime final PR/MR creation as a separate phase while keeping manual
handoff guardrails explicit.

**In scope:** `ship-auto-final.json` workflow, `shipper-auto-final` agent,
summary persistence for downstream PR creation, final PR/MR wiring,
idempotency checks, safety comments/guardrails.

**Out of scope:** auto-merge, auto-approve, conflict resolution automation.
