# Success Criteria

- [ ] E7-F1 and E7-F6 dependency contracts are consumed without duplicating or
  weakening their selection, fallback, error, or export policy.
- [ ] At least one supported condensation workflow executes through the same
  user-facing selection boundary on CPU and Warp CPU.
- [ ] Isothermal and latent-heat capabilities have explicit configuration,
  state, return, mutation, ownership, and failure semantics.
- [ ] GPU staggered condensation and unsupported BAT/activity mappings fail with
  deterministic capability errors before adapter-driven mutation.
- [ ] Warp execution delegates to `condensation_step_gpu` without physics
  rewrites, conversion, restore, implicit synchronization, or silent fallback.
- [ ] Returned/result metadata accurately records identity and in-place changes
  to particles, gas, transfer, vapor-pressure, and energy outputs.
- [ ] CPU/Warp CPU parity passes for the recorded fixture matrix and explicit
  tolerances; particle-plus-gas conservation passes its declared bounds.
- [ ] Validation-order and failure-boundary tests prove atomic preflight and do
  not overpromise rollback after launched work.
- [ ] Optional CUDA rows skip cleanly without suitable hardware and pass when
  available; CUDA is not required in routine CI.
- [ ] Existing direct CPU and GPU APIs, narrow exports, and optional-Warp CPU
  imports remain compatible.
- [ ] Every changed module has co-located tests and >=80% coverage; focused
  tests, Ruff, mypy, docs regressions, and strict MkDocs build pass.
- [ ] Documentation identifies E7-F4/E7-F5 handoff requirements while keeping
  resident sessions, full scheduling, graph capture, performance, and autodiff
  outside E7-F2.

## P1 Delivered Criteria (Issue #1470)

- [x] Every valid semantic configuration maps to one exact, immutable
  four-axis requirement set without composing partial declarations.
- [x] The catalogue contains 36 CPU and 8 declarative Warp-profile entries;
  Warp fails closed for staggered and nonrepresentable activity/surface modes.
- [x] Metadata queries are dependency-neutral and read-only, preserve the
  existing unsupported-declaration error, and cannot select adapters or parse
  native devices.
- [x] Focused execution tests cover mapping, rejection order, purity, and
  optional-import isolation while leaving exports and GPU APIs unchanged.

## P2 Delivered Criteria (Issue #1471)

- [x] `particula.execution` is a package with unchanged legacy selection
  imports and exact ten-name `__all__`; concrete carriers remain absent from
  selection and top-level exports.
- [x] Concrete CPU and lazy Warp carriers are frozen, identity-retaining,
  metadata-only, and non-executing; CPU-only import/construction remains
  Warp/GPU-free.
- [x] Warp construction validates ordered primary metadata and only writable
  output ownership, retaining opaque sidecars without inspecting them.
- [x] Focused tests prove validation ordering, alias/overlap rejection,
  non-mutation, and no transfer, synchronization, conversion, or execution.

## P3 Delivered Criteria (Issue #1472)

- [x] Concrete-only selected isothermal CPU and Warp adapters validate their
  exact P3/P2 carriers and controls before one native backend call.
- [x] CPU preserves `MassCondensation.execute()` control, return, exception,
  and identity semantics; Warp preserves the native kernel tuple by identity.
- [x] Warp resolves its native kernel lazily only after preflight and performs
  no transfer, restoration, synchronization, fallback, or recovery.
- [x] Focused dispatch and export-boundary tests cover CPU/Warp call contracts,
   identity normalization, lazy import, and narrow public exports.

## P4 Delivered Criteria (Issue #1473)

- [x] Selected Warp dispatch forwards `latent_heat`, `energy_transfer`, and
  deferred `thermal_work` by identity in its one native call; CPU remains
  isothermal and the native tuple shape is unchanged.
- [x] Capability-profile rejection for staggered and nonrepresentable Warp
  semantics occurs before lazy resolver lookup, native dispatch, and writes.
- [x] Direct-kernel thermal validation and exceptions propagate unchanged through
  the adapter; no adapter thermal solver, recovery, rollback, transfer, or sync
  was added.
- [x] Focused tests cover sidecar identity, no-/zero-heat behavior, finalized
  transfer energy accounting, deferred-work behavior, native validation seams,
  and unsupported-profile preflight.
