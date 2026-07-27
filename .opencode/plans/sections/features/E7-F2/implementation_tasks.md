# Implementation Tasks

## Capability and Configuration

- [x] Add direct-module-only condensation process/capability declarations in
  `particula/execution.py` for CPU and declarative Warp profiles.
- [x] Define immutable configuration vocabulary and exact requirements mapping
  for execution mode, latent heat, ideal/kappa/nonrepresentable activity, and
  static/composition-weighted/nonrepresentable surface semantics.
- [x] Declare all 36 CPU combinations and the eight supported Warp-profile
  combinations; staggered and nonrepresentable Warp semantics fail closed via
  the existing matrix error.
- [x] Preserve CPU-only import safety: P1 has no adapter module, Warp import,
  runtime/device availability logic, fallback, export, or GPU API behavior.

## Adapter Implementation

- [ ] Define immutable configuration/state views in
  `particula/execution/adapters/condensation.py`, including
  `CondensationExecutionConfig`, CPU `Aerosol` state, and resident Warp
  particle/gas/environment/sidecar references.
- [ ] Implement CPU delegation to `MassCondensation.execute()` without changing
  time-step, substep, exception, returned-object, or mutation behavior.
- [ ] Implement a GPU-scoped adapter that calls `condensation_step_gpu` with
  caller-owned resident state and sidecars; perform no conversion or sync.
- [ ] Normalize the heterogeneous backend outputs into E7-F1's
  `ExecutionResult` while preserving actual object identity and mutation facts.
- [ ] Register the adapter through the typed context only after deterministic
  validation; never retry on CPU after a Warp error.
- [ ] Preserve direct kernel APIs and narrow export boundaries.

## Validation and Testing

- [ ] Add per-phase tests in
  `particula/execution/tests/condensation_adapter_test.py` for every helper and
  rejection branch; add cross-backend workflow fixtures only in
  `particula/execution/tests/condensation_integration_test.py`.
- [ ] Lock selection-level validation order and prove invalid requests do not
  invoke an adapter, allocate adapter resources, transfer, or mutate state.
- [ ] Assert same-device fixed-shape metadata and caller-owned sidecar identity.
- [ ] Cover fixed four-substep semantics, coupled gas inventory,
  vapor-pressure refresh, total transfer, latent heat, and energy accounting.
- [ ] Compare selected CPU and Warp CPU workflows with explicit per-case
  tolerances and particle-plus-gas conservation checks.
- [ ] Add optional CUDA parametrization that skips cleanly when unavailable.
- [ ] Run focused tests with `-Werror`, changed-module coverage >=80%, Ruff,
  mypy, existing kernel/export regressions, and strict documentation build.
