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
- **P7 / issue #1534 (2026-08-11, shipped):** Required artifacts and validation
  evidence are recorded. Focused assertions (289), export boundary (15),
  resident fast suite (891), full-package coverage (6,254 tests, 93%),
  changed-executable-module coverage (891 tests, 95%; `diagnostics.py` 79%,
  `gpu_resources.py` 87%, `checkpoint.py` 87%, `resident_scheduler.py` 86%),
  strict MkDocs, and optional CUDA rows all passed. Warp CPU was available.

  Changed-module coverage targets the P1--P6 executable modules
  `particula/execution/diagnostics.py`,
  `particula/execution/gpu_resources.py`,
  `particula/execution/checkpoint.py`, and
  `particula/execution/resident_scheduler.py`; its aggregate >=80% gate passed.
  P7 Markdown-only changes are not coverage targets. CUDA remains optional
  pass-or-clean-skip evidence.

  Required rerun commands (run sequentially and retain literal output):

  ```bash
  pytest particula/execution/tests/diagnostics_test.py particula/execution/tests/gpu_resources_test.py particula/execution/tests/checkpoint_test.py particula/execution/tests/rng_invariance_test.py particula/execution/tests/full_loop_test.py particula/execution/tests/multi_box_loop_test.py particula/execution/tests/transport_loop_test.py particula/execution/tests/restart_loop_test.py particula/execution/tests/condensation_integration_test.py particula/execution/tests/coagulation_integration_test.py particula/execution/tests/errors_test.py particula/execution/tests/fallback_test.py particula/execution/tests/fallback_integration_test.py particula/tests/gpu_resident_multi_timestep_docs_test.py -q
  pytest particula/execution/tests/exports_test.py particula/tests/execution_exports_test.py -q
  pytest particula/execution/tests/ -q
  pytest --cov=particula --cov-report=term-missing
  pytest particula/execution/tests/ -q --cov=particula.execution.diagnostics,particula.execution.gpu_resources,particula.execution.checkpoint,particula.execution.resident_scheduler --cov-report=term-missing --cov-fail-under=80
  mkdocs build --strict
  pytest particula/execution/tests/multi_box_loop_test.py -q -m "warp and cuda"
  pytest particula/execution/tests/condensation_integration_test.py particula/execution/tests/coagulation_integration_test.py -q -m "warp and cuda"
  ```

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
