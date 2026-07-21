# Dependencies

## Upstream

- **E6-F5 / T5: Fixed particle-slot activation and diagnostics** is mandatory.
  E6-F7 consumes its active/free truth table, ascending slot order, request
  shape, exact counts, and atomic CPU activation without weakening semantics.
- **E6-F6 / T6: Slot-exhaustion resampling and volume scaling** is mandatory.
  E6-F7 passes complete finalized demand to its resampling-first planner and
  optional scaling fallback. Both policies off with insufficient capacity must
  fail before gas or particle mutation.
- Existing `GasData`, `ParticleData`, dynamics strategy/runnable conventions,
  and nucleation theory provide container and API foundations.

## Downstream

- **E6-F8 / T8** requires E6-F5, E6-F6, and this CPU scientific/source oracle
  before direct fixed-shape Warp nucleation parity.
- **E6-F9 / T9** consumes CPU/GPU process contracts in integrated direct tests.
- Epic G may later schedule the process but cannot expand E6-F7 into backend
  selection, resident loops, or hidden transfers.

## Phase Ordering

P1 freezes equations, units, domains, and strategy behavior before P2 source
finalization. P2 precedes P3 because exhaustion consumes finalized complete
demand. P4 exposes only the stable P1-P3 contract. P5 adds the runnable. P6
verifies the complete transaction and creates the E6-F8 oracle. P7 is the final
documentation phase. Every production phase includes co-located tests.

E6-F5 and E6-F6 remain explicit dependencies even on a coordinated branch.
E6-F7 may not ship temporary private slot/exhaustion behavior.
