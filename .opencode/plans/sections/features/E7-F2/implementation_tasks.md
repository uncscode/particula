# Implementation Tasks

## Capability and Configuration

- [x] Add direct selection-boundary condensation process/capability declarations
  in `particula/execution/__init__.py` for CPU and declarative Warp profiles.
- [x] Define immutable configuration vocabulary and exact requirements mapping
  for execution mode, latent heat, ideal/kappa/nonrepresentable activity, and
  static/composition-weighted/nonrepresentable surface semantics.
- [x] Declare all 36 CPU combinations and the eight supported Warp-profile
  combinations; staggered and nonrepresentable Warp semantics fail closed via
  the existing matrix error.
- [x] Preserve CPU-only import safety: P1 has no adapter module, Warp import,
  runtime/device availability logic, fallback, export, or GPU API behavior.

## Adapter Implementation

- [x] Define immutable configuration/state views in
  `particula/execution/adapters/condensation.py`, including
  `CondensationExecutionConfig`, CPU `Aerosol` state, and resident Warp
  particle/gas/environment/sidecar references. Migrate `particula.execution`
  to a package while preserving its exact ten-name public selection `__all__`.
- [x] Implement CPU delegation to `MassCondensation.execute()` without changing
  time-step, substep, exception, returned-object, or mutation behavior.
- [x] Implement a GPU-scoped adapter that calls `condensation_step_gpu` with
  caller-owned resident state and sidecars, forwarding Warp thermal sidecars by
  identity while retaining CPU-isothermal dispatch; perform no conversion or
  sync.
- [x] Normalize the heterogeneous backend outputs into E7-F1's
  `ExecutionResult` while preserving actual object identity and mutation facts.
- [x] Support typed context registration and resolution of the concrete adapter
  objects after deterministic preflight; never retry on CPU after a Warp error.
- [x] Preserve direct kernel APIs and narrow export boundaries.

## Validation and Testing

- [x] Add P2 carrier tests in
  `particula/execution/tests/condensation_adapter_test.py` for every helper and
  rejection branch, including import/export, CPU/Warp metadata, validation
  ordering, ownership, and non-mutation. Add cross-backend workflow fixtures only in
  `particula/execution/tests/condensation_integration_test.py`.
- [x] Lock P3 adapter preflight and prove invalid requests do not invoke a
  backend call, resolve the lazy Warp helper, transfer, synchronize, or mutate
  state.
- [x] Assert P2 same-device fixed-shape metadata, writable-output ownership,
  caller-owned sidecar identity, and no execution/transfer/synchronization.
- [x] Cover fixed four-substep semantics, coupled gas inventory,
  vapor-pressure refresh, total transfer, latent heat, and energy accounting
  through the local independent Warp P2 oracle and resident integration cases.
- [x] Cover selected native CPU and Warp CPU workflows with explicit per-case
  tolerances and particle-plus-gas conservation checks, without asserting
  cross-backend numerical equality.
- [x] Add optional CUDA parametrization for uptake, two-box, and latent-heat
  cases that skips cleanly when unavailable.
- [x] Cover CPU/Warp selected dispatch, exact native arguments and call counts,
  identity-preserving normalization, exception propagation, lazy Warp import,
  and the public export boundary.
- [x] Cover selected-Warp latent-heat sidecar identity, direct native thermal
  validation propagation, omitted/zero heat behavior, energy accounting, and
  unsupported profile rejection before lazy resolution or writes.
