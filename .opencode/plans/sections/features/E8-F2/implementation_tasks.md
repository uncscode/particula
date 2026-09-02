# Implementation Tasks

## Backend

- [x] Add exact frozen prepared-plan carriers and `prepare_*` validation in the
  focused concrete-only `particula/execution/resident_enqueue.py` module
  (#1552). The READY-only carrier retains metadata by identity and performs no
  executor construction or lifecycle mutation.
- [x] Extract fixed state-update copy launches from validation/readback in
  `particula/execution/state_updates.py`; bind exact P1 input/destination
  identities during preparation and retain standalone execution compatibility
  (#1553).
- [x] Add P1-bound thermodynamic common binding and current-cursor consumer
  preparation in `particula/execution/thermodynamic_updates.py`; enqueue only
  bound vapor-pressure/saturation operations while the coordinator retains
  freshness and cursor ownership (#1553).
- [x] Extract read-only `ResidentDiagnosticsExecutor` validation into a
  functional seam while retaining the executor compatibility wrapper (#1552).
- [x] Bind validated diagnostics registrations, arrays, and outputs to a
  P1-prepared diagnostics carrier with enqueue-only ordered dispatch while
  retaining the standalone executor path (#1553).
- [x] Extract read-only resident communication-request validation and reuse it
  through shared scheduler complete-loop metadata validation (#1552).
- [x] Add P1-bound closed-map GAS/PARTICLES communication setup and a frozen
  prepared binding in `particula/execution/resident_communication.py`; retain
  exact pinned resources, duration, and optional final-volume identity (#1554).
- [x] Extract explicit-input native prepared GAS, PARTICLES, and resident-volume
  launches in `particula/gpu/kernels/communication.py`; compose fixed
  communication-before-volume enqueue without host work and retain legacy
  compatibility adapters (#1554).
- [x] Refactor `condensation_step_gpu` into private
  `_PreparedCondensationCall` setup and enqueue paths in
  `particula/gpu/kernels/condensation.py`, preserving validation, fallback
  allocation, identity, return, and fixed-four-substep physics contracts (#1555).
- [x] Retain the prepared condensation call through the concrete private
  `_PreparedWarpCondensationBinding` in
  `particula/execution/adapters/condensation.py`; enqueue uses retained
  references only and adds no scheduler/resource/checkpoint integration (#1555).
- [x] Refactor resident Brownian coagulation, dilution, and selected/all-box wall
  loss into validated setup plus private enqueue in
  `particula/gpu/kernels/{coagulation,dilution,wall_loss}.py` and their resident
  adapters (#1556). Private frozen calls/bindings retain validated references;
  public wrappers, exports, process behavior, and persistent-RNG semantics are
  unchanged.
- [x] Complete the nucleation and nested exhaustion prepared device-only enqueue
  sequence in `particula/gpu/kernels/{nucleation,exhaustion}.py` (#1557).
  Preparation preserves direct validation precedence, pins launch arrays,
  freezes scalar controls, and allocates all private status/workspace storage.
  Enqueue scans live pinned payloads and issues only a fixed device-gated
  P1--P5/P4 sequence; observers alone perform bounded status interpretation.
- [x] Bind `ResidentNucleationAdapter` to its exact resident identities and the
  prepared nucleation enqueue delegate without scheduler composition, resource
  changes, token entry, payload inspection, transfer, synchronization, or cache
  retention (#1557).
- [x] Publish the concrete-only prepared-enqueue developer contract and source
  docstring reconciliation, then record focused documentation assertions (22
  passed in 0.09s) and `mkdocs build --strict` (exit 0 in 14.67s) (#1559).
- [ ] Compose all prepared operations in canonical schedule order without
  per-node dictionary lookup, executor construction, allocation, validation,
  import resolution, host readback, or synchronization during enqueue.
- [ ] Preserve lifecycle token and writer-failure semantics, E8-F1 signature
  checks before enqueue, direct API validation ordering, and all existing
  concrete-only export boundaries.

## Tooling and Tests

- [x] Add P1 setup contract tests for exact bindings, lifecycle/signature and
  schedule gating, second-check drift, and no mutation or forbidden host work
  on preparation rejection (#1552).
- [x] Add P2 per-module validate-once/enqueue-only tests for state,
  thermodynamic, and diagnostics seams. They cover identity/pinning rejection,
  writer order, empty no-ops, and traps for setup-only validation, allocation,
  readback, synchronization, registry, schedule, and freshness work (#1553).
- [x] Add P3 communication/volume regressions for closed GAS/PARTICLES bindings,
  identity drift rejection, absent/equal/changed final volumes, no-op barriers,
  gated GAS overdraw, dispatch order, and forbidden enqueue-time operations
  (#1554).
- [x] Extend focused condensation kernel and adapter tests for prepared setup,
  enqueue-only restrictions, direct-wrapper compatibility, retained identities,
  and four-substep behavior (#1555).
- [x] Extend focused coagulation, dilution, wall-loss, and resident-adapter tests
  for prepared setup/enqueue delegation, frozen references, selected lanes,
  no-op behavior, output/RNG identity, and forbidden enqueue-time setup work
   (#1556).
- [x] Add completion evidence for the device-only nucleation/exhaustion enqueue:
  prepared equivalence, post-prepare dynamic payload gating, fixed-policy
  non-substitution, forbidden allocation/readback/synchronization operations,
  and P1--P5/P4 conservation/error precedence (#1557). Existing adapter identity
  and public-resolver compatibility-fallback coverage remains valid (#1557).
- [ ] Add a CUDA-gated capture smoke test using the repository's established
  pass-or-clean-skip helper; Warp CPU must explicitly remain uncaptured.
- [ ] Run focused tests without coverage, then the untargeted
  `.opencode/tools/run_pytest.py` full-package coverage gate.
