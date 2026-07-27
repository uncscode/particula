# Dependencies

## Upstream

- **E7-F1 (required):** Supplies typed backend/device/process requests,
  capability matrix, execution context/result, and CPU reference adapter. E7-F6
  extends these contracts rather than defining a competing selection layer.
- Existing CPU `RunnableABC` behavior, optional-Warp import guards, explicit
  conversion helpers, and deliberate direct-kernel exports are shipped inputs.
- Issue #1451 and Epic G roadmap lines 1472-1542 remain scope authority.

## Downstream

- **E7-F2 and E7-F3:** GPU condensation and Brownian coagulation adapters rely on
  frozen capability errors and fallback behavior.
- **E7-F4:** Resident sessions rely on the rule that active device state cannot
  silently fall back and that restore boundaries are explicit.
- **E7-F5:** Scheduler propagates errors and consumes requested/selected backend
  metadata without inventing policy.
- **E7-F7/E7-F8/E7-F9:** Transport, RNG, and closeout regressions inherit the
  public API and stability contract.

## Phase Ordering

P1 precedes P2 so all validation paths use typed errors. P2 precedes P3 because
fallback consumes availability outcomes. P4 freezes only the tested P1-P3
surface. P5 verifies the combined contract. P6 documents the accepted behavior
last. Every implementation phase includes its unit tests; there is no separate
unit-testing phase.
