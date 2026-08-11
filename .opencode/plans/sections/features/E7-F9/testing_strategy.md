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
- **P4 (completed, issue #1531):** `multi_box_loop_test.py` exercises the real
  resident scheduler with a 4-box, 16-slot, 2-species fixed-capacity fixture.
  It compares zero-duration multi-box state with decomposed one-box sessions and
  checks tight closed per-box/species inventory (`rtol=1e-12`, `atol=1e-30`).
  Logical-ID permutation, unrelated valid-box addition, selected and empty
  wall-loss rows, and all-free no-work rows prove isolation without relying on
  physical lane order. Positive-duration cases verify separate pinned
  coagulation/wall-loss streams and same-backend logical-box continuity. Neutral
  wall loss has Warp-CPU 100-seed aggregate binomial evidence; CUDA has a
  12-seed finite, bounded smoke row that skips cleanly when unavailable.
- **P5 (completed, issue #1532):** `transport_loop_test.py` uses an independent
  NumPy float64 extensive-amount oracle for two-step directed expansion and
  reciprocal mixing with dilution, sparse closed-map conservation, and
  empty/disabled write-free barriers. `restart_loop_test.py` verifies
  exact-device closed-transport restart, fresh restored identities, preserved
  published coagulation/wall-loss stream words without initialization, continued
  transport equivalence, and nonexact-device rejection. The direct
  `communication_test.py` row reconciles open-boundary total change with
  source-minus-sink ledgers at `rtol=1e-12`, `atol=1e-30`. These are test-only
  changes; no production or public-contract behavior changed.
- **P6 (completed, issue #1533):**
  `particula/tests/gpu_resident_multi_timestep_docs_test.py` covers deterministic
  forced-disabled and missing-Warp guidance; broken enabled imports; resolver,
  setup, and dispatch failures without fallback; and the real Warp-CPU example
  when available. Enabled coverage asserts one source upload per CPU container,
  two source and one restarted dispatch, `(3, 1)` caller diagnostics, preserved
  ordinary-step identities, manual exact-device restart into fresh identities,
  and cached source finalization. A warnings-as-errors subprocess remains
  optional-Warp and does not require CUDA.
- **P7 / issue #1534 (2026-08-11, blocked):** Focused regressions (289), exports
  (15), resident fast suite (891), full-package coverage (93%), execution-scope
  coverage (95%; recorded P1--P6 aggregate 86%), and optional CUDA rows (1 and
  5) passed. The equivalent strict wrapper passed, but the exact required
  `mkdocs build --strict` command remains required before closeout.

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
