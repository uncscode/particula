<!-- TEMPLATE: Replace this entire file with your dependency mapping -->

List upstream and downstream dependencies so phase ordering is clear.

**Required elements:**
- **Upstream:** What must exist before this feature can start?
- **Downstream:** What depends on this feature being shipped?
- **Phase ordering notes:** Any sequencing constraints between phases

**Example (E16-F6):**
- **Upstream:**
  - E16-F5 (Branch-First Auto-Mode Dispatcher) -- manifest completion detection
  - E16-F2 (Ship-Auto Branch Integration) -- branch accumulation infrastructure
- **Downstream:**
  - Auto-mode runbook documentation depends on final handoff behavior
  - Future auto-merge features would extend the guardrail system from P3
- **Phase ordering:** P1 (agent) must ship before P2 (PR wiring) which must
  ship before P3 (guardrail comments)
