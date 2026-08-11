# Testing Strategy

Every phase ships its implementation or evidence artifact with co-located tests.
Coverage thresholds are never lowered; changed executable modules maintain at
least 80% coverage. Test files use `*_test.py`. Warp CPU is the routine installed-
Warp baseline; CUDA rows are optional and skip cleanly when unavailable.

## Per-Phase Approach

- **P1 (completed, issue #1528):** `diagnostics_test.py` and
  `gpu_resources_test.py` cover the concrete six-operation protocol, independent
  NumPy float64 reducer oracles, identity retention, canonical ordering, empty
  shapes, invalid schema/device/capacity/alias preflight, and no-write failure
  boundaries. Coverage includes total particle-plus-gas mass, particle-number
  concentration, direct latent-energy ledger copy, and source/sink-signed
  conservation residuals; public descriptors and observation results were not
  introduced.
- **P2 (completed, issue #1529):** Focused `checkpoint_test.py` coverage freezes
  schema-v3 continuation metadata with valid empty current-word payloads,
  asserts bidirectional coagulation/wall-loss resource-to-continuation failure
  handling, verifies canonical-primary payload immutability, and confirms
  schema-v2 noncommunication restart. Broader malformed-payload matrices and
  uninterrupted-versus-restart equivalence remain P2 follow-up evidence.
- **P3 (completed, issue #1530):** `full_loop_test.py` runs two real
  resident-scheduler dispatches for independently constructed closed GAS and
  PARTICLES maps. It covers the twelve-node ordinary trace, vapor-pressure and
  saturation-refresh windows, current NumPy float64 thermodynamic observations,
  one CPU-to-resident upload per container, stable resident identities/schemas,
  and closed GAS particle-plus-gas inventory at `rtol=1e-12`, `atol=1e-30`.
  A controlled late wall-loss writer failure verifies token closure, session
  faulting, and later lifecycle rejection. The regression also covers the
  corrected ordinary nucleation dispatch; no public ordering change is claimed.
- **P4:** Independent multi-box versus decomposed one-box parity; unrelated-box
  addition, disablement, and reordering metamorphic tests; a 4-box,
  16-particle-slot, 2-species fixed-capacity particle-resolved fixture.
- **P5:** CPU extensive-amount oracles for advection/mixing/dilution/expansion,
  open/closed ledgers, conservation, checkpoint/restart, and RNG continuation.
- **P6:** Execute the published example with warnings as errors; assert public
  imports, checkpoint-only transfer counts, clean no-Warp guidance, and outputs.
- **P7:** Run focused regressions, full fast suite, export checks, optional CUDA
  rows, coverage, and `mkdocs build --strict`; publish reproducible commands.

## Required Validation Matrix

| Concern | Required evidence | Backend / acceptance |
|---------|-------------------|----------------------|
| Selection adapters | Condensation and Brownian workflows vs CPU | Warp CPU, recorded `rtol`/`atol` |
| Resident loop | One setup upload; no bulk transfer/sync until checkpoint | Transfer/sync spies, exact counts |
| Ordering | Current environment, vapor pressure, saturation, gas | Exact call/order and state assertions |
| Identity/capacity | Containers, arrays, sidecars stable across steps | Identity, shape, dtype, capacity assertions |
| Conservation | Particle-plus-gas species and transport ledgers | Independent float64 oracle; explicit tolerances |
| Independent boxes | Multi-box equals decomposed one-box runs | Deterministic parity; stochastic stream/stat rules |
| Communication | Advection, dilution, mixing, expansion | CPU extensive-amount oracle; open/closed accounting |
| RNG/restart | Added/disabled/reordered boxes; checkpoint continuation | Stable logical streams; exact same-backend continuation |
| Errors/fallback | Missing device and unsupported physics | Clear error or explicit transition; never silent |
| CUDA | Same bounded rows where hardware exists | Optional pass; clean skip when unavailable |
| Documentation | Complete example and support contract | Executable docs regression; strict build |

Stochastic tests do not require exact CPU/CUDA trajectories. Use stable stream
contracts for same-backend restart and aggregate/statistical bounds otherwise.
Conservation should retain the repository's tight concentration-weighted policy
where applicable (`rtol=1e-12`, `atol=1e-30`) unless an owning process contract
records a different justified tolerance.
