# Condensation Stiffness Study and Recommendation Record

This note is the canonical decision record for GPU condensation integration.
The shipped production path uses four fixed equal substeps; P2/P3 candidate
comparisons remain historical evidence only. The current production GPU path is
`float64` bounded and couples P2-finalized particle transfer to gas
concentration; this remains a narrow direct-kernel contract rather than a
general GPU production claim.

## Current Runtime Scope

- Production path: explicit fixed-four GPU condensation. Every successful
  `condensation_step_gpu(...)` call performs exactly four `time_step / 4.0`
  substeps.
- Particle and gas update: each raw proposal is P2-finalized against inventory,
  then applied to particle mass and coupled to gas concentration by the matching
  particle-concentration-weighted transfer.
- Per-substep order: optionally refresh composition-weighted surface tension,
  overwrite `gas.vapor_pressure`, refresh environment properties, produce a raw
  transfer proposal, P2-finalize and apply it, accumulate the finalized
  transfer, and couple gas concentration. Later proposals read coupled gas;
  vapor-pressure refresh does not read gas concentration.
- Transfer buffers: the resolved total transfer is cleared once after preflight
  and accumulates P2-finalized transfer over all four substeps. A supplied total
  buffer is returned by identity; the separate work buffer retains only the
  final raw proposal.
- Failure boundary: invalid aggregate P2 state or sidecars fail before launches
  or mutation. A later non-finite fresh proposal fails before P2 mutation in
  that cycle but does not roll back earlier completed substeps.
- Baseline backend for this phase: `np.float64` inputs and Warp CPU execution.
- Accepted study inputs: scalar `temperature` and `pressure`, direct Warp
  arrays with shape `(n_boxes,)`, and the tested hybrid mode where one direct
  input stays scalar while the other uses a Warp `(n_boxes,)` array.
- Study cases are deterministic fixed-shape builders that should be recreated
  from clean inputs after failed validation or intentionally unstable runs.
- Recommendation dependency boundary: the conclusions below assume the shipped
  E2-F2 environment-shape contract for scalar or direct Warp `(n_boxes,)`
  temperature and pressure inputs, and they stay inside the E2-F6 `float64`
  evidence envelope.

## Shared Case Catalog

The executable test baseline defines three named regimes:

- `nanometer`
- `accumulation_mode`
- `droplet_like`

Each `CondensationStiffnessCase` declares:

- `name`
- `n_boxes`
- `n_particles`
- `n_species`
- `time_step`
- scalar baseline `temperature` and `pressure`
- deterministic scaling for particle mass, gas concentration, and vapor
  pressure
- optional zero-mass or zero-concentration particle edges

The important contract is fixed shape, not random sampling. Every case rebuilds
deterministic particle, gas, vapor-pressure, and environment inputs with
explicit `(n_boxes, n_particles, n_species)` and `(n_boxes,)` dimensions.

## Shared Metrics and Classification

The baseline tests use the following helper concepts:

- `_particle_mass_is_nonnegative`: all post-step particle masses are `>= 0`.
- `_particle_values_are_finite`: particle/gas/support arrays remain finite.
- `_fractional_mass_change_per_bin`: fractional change is computed only for
  positive initial masses, so zero-initial-mass bins need a separate stability
  check.
- `_zero_mass_entries_remain_stable`: zero-initial-mass bins remain at zero so
  the zero-mass edge stays deterministic; helper inputs must have matching
  shapes.
- `_validate_stiffness_case_metadata`: declared case metadata must match array
  shape and `np.float64` dtype expectations.
- `_classify_particle_only_condensation_stiffness`: historical test helper name
  retained for the recorded study; it is not the current gas-coupled runtime
  contract.

### Boundary Semantics

- Fractional-mass-change thresholds are inclusive: exact equality remains
  `stable`.
- Zero-initial-mass growth remains `unstable` even though fractional change is
  reported only for positive initial masses.
- Declared shape mismatches fail validation.
- Declared dtype mismatches fail validation.
- Zero-mass entries are treated as stable only when they remain unchanged.
- Accepted explicit `environment=...` inputs with `(n_boxes,)` Warp arrays match
  accepted direct `(n_boxes,)` Warp-array inputs and should not mutate
  caller-owned temperature or pressure arrays.

## Measured Recorded Timestep Grid

The current recorded sweep is implemented in
`particula/gpu/kernels/tests/_condensation_test_support.py` and exposed through
the discoverable wrapper
`particula/gpu/kernels/tests/condensation_stiffness_test.py`. For each named
case, the collected tests:

- execute the fixed timestep grid from `_RECORDED_TIMESTEP_GRID_BY_CASE`
- reuse one caller-owned preallocated `mass_transfer` buffer per case/device
- rebuild particle, gas, and vapor-pressure inputs before every trial
- historical baseline: record the then-current particle-only behavior; these
  rows do not describe the later P2-finalized gas-coupled runtime
- keep scalar `temperature` / `pressure` inputs for single-box cases and direct
  Warp `(n_boxes,)` arrays for the multi-box `droplet_like` case

| Case | Environment input mode | Timestep | Threshold | Classification | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `nanometer` | scalar `temperature` / `pressure` | `0.00005` | `1.0` | `stable` | Historical particle-only baseline; caller-owned buffer reused and overwritten. |
| `nanometer` | scalar `temperature` / `pressure` | `0.05` | `1.0` | `stable` | Historical particle-only baseline. |
| `nanometer` | scalar `temperature` / `pressure` | `50.0` | `1.0` | `stable` | Historical particle-only baseline. |
| `accumulation_mode` | scalar `temperature` / `pressure` | `0.004` | `1.0` | `stable` | Historical particle-only baseline; caller-owned buffer reused and overwritten. |
| `accumulation_mode` | scalar `temperature` / `pressure` | `0.4` | `1.0` | `stable` | Historical particle-only baseline. |
| `accumulation_mode` | scalar `temperature` / `pressure` | `40.0` | `1.0` | `stable` | Historical particle-only baseline. |
| `droplet_like` | direct Warp `(n_boxes,)` arrays | `0.04` | `1.0` | `stable` | Historical particle-only baseline; multi-box direct arrays supported. |
| `droplet_like` | direct Warp `(n_boxes,)` arrays | `4.0` | `1.0` | `stable` | Historical particle-only baseline. |
| `droplet_like` | direct Warp `(n_boxes,)` arrays | `400.0` | `1.0` | `stable` | Historical particle-only baseline. |

Across the current recorded grid, the executable tests observe the same
historical particle-only maximum fractional-mass-change magnitude (`1.0`) for every row.
The baseline therefore applies one inclusive threshold (`1.0`) across the full
grid and records every row as `stable` under that shared rule. Separate unit
tests still cover the `unstable` branch for larger fractional changes,
zero-mass growth, and non-finite values. This is recorded-grid evidence for the
historical fixed-shape particle-only path, not evidence for the current
gas-coupled conservation contract or a general stable-timestep limit.

## Candidate Evaluation Evidence

P2/P3 recorded two deterministic prototype candidates implemented in
`particula/gpu/kernels/tests/_condensation_test_support.py` and collected
through `particula/gpu/kernels/tests/condensation_stiffness_test.py`. They
remain test-local evidence only; the public `condensation_step_gpu(...)`
runtime and package export surface are unchanged. The later production path
does ship gas coupling, but these candidates remain historical evidence only.

| Candidate | Family | Buffer reuse | Determinism | Finite/non-negative masses | CPU-reference agreement | Graph capture | Autodiff note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_count_substeps_4` | Historical P2/P3 fixed-count explicit sub-stepping evidence; its fixed-four behavior is now shipped. | Pass: one caller-owned `mass_transfer` array plus fixed-shape `work`/`accumulator` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 5e-2` at the baseline timestep and `max relative error <= 5e-2` across the recorded grids for `nanometer`, `accumulation_mode`, and `droplet_like`. These bounds are case-specific evidence, not a general accuracy tolerance or stable-timestep limit. | Not evaluated as production graph-capture evidence. | Not evaluated as production autodiff evidence. |
| `asymptotic_relaxation` | Asymptotic first-order bounded relaxation | Pass: one caller-owned `mass_transfer` array plus one fixed-shape `work` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 3.5e-1` at the baseline timestep and `max relative error <= 3.5e-1` across the recorded grid. This looser bound is recorded as prototype evidence only and is not suitable for a production recommendation by itself. | Pass: fixed-shape algebra with no adaptive search or variable-length loops. | `exp(...)` relaxation remains differentiable away from the same clamp boundary, so it is a plausible autodiff target but not yet production-qualified. |

### Historical Phase Boundary Decision

- This decision preceded the shipped P2-finalized gas-coupled production path.
  The candidate coverage remains deterministic evaluation evidence, not a
  current runtime description.
- The asymptotic candidate remains evidence-only because the tolerance required
  to track the current CPU/explicit reference is materially looser than the
  fixed-count candidate.
- Because the candidate evidence was credible in test-local helpers, no private
  production helper was added to `particula/gpu/kernels/condensation.py`.

## Final Recommendation

### Shipped fixed-four implementation

`fixed_count_substeps_4` is the shipped production integration behavior of
`condensation_step_gpu(...)`, not a future prototype. Each valid call uses four
equal substeps, updates particle mass and gas concentration using the same
P2-finalized transfer, and keeps production calculations device resident.

Why this is the recommended path:

- The historical P2/P3 comparison recorded `rtol <= 5e-2` at the baseline
  timestep and maximum relative error `<= 5e-2` only for the named
  `nanometer`, `accumulation_mode`, and `droplet_like` grids.
- Successful calls preserve supplied total-buffer identity, clear that total
  once after preflight, and accumulate P2-finalized transfer. A
  separate work buffer retains the final raw proposal.
- `CondensationScratchBuffers` remains a concrete-module-only sidecar. Its
  supplied active-device, stable-shape `wp.float64` fields may be omitted
  independently, in which case the step allocates fallback buffers.
- This shipped behavior does not establish graph capture/replay or autodiff
  readiness.

### Alternatives considered but not selected

- **Historical single-step explicit update:** retained only as a P2/P3
  comparison baseline; it is not the shipped production path.
- **`asymptotic_relaxation`:** retain as evidence-only. It remains interesting
  for differentiability because the algebra is smooth away from clamp
  boundaries, but the measured CPU-reference agreement is materially looser
  (`rtol <= 3.5e-1` / `max relative error <= 3.5e-1`) than the fixed-count
  candidate.
- **Adaptive or dynamic-loop schemes:** deferred; no adaptive-stepping support
  is documented by this record.

### Current gas-coupling boundary

The shipped direct kernel couples each finalized transfer to gas concentration;
later mass-transfer proposals observe that updated gas state. This is not
temperature feedback, a strategy or `Runnable` path, adaptive stepping, graph
capture/replay, autodiff support, or general CPU-strategy parity. A failed
aggregate P2 preflight is atomic, while a later raw-proposal failure retains
completed earlier substeps rather than rolling back the whole call.

### Downstream gates and dependency boundaries

E4-F4 P2 ships a low-level, per-substep latent-heat rate correction with
CPU-oracle/Warp parity. Issue #1272 also ships its caller-owned, signed
`energy_transfer` diagnostic: after successful preflight, its active-device
`wp.float64` `(n_boxes, n_species)` output is overwritten with P2-finalized
transfer times latent heat. It does not add temperature-state
feedback, E4-F6
independent-device plus graph/autodiff evidence, or E4-F7 strategy/runnable
and final support work.

- **E2-F2 environment-shape dependency:** the recommendation assumes the shipped
  contract for scalar inputs and explicit direct Warp `(n_boxes,)` environment
  arrays. Broader conclusions should not be inferred for different environment
  ownership or shape models.
- **E2-F6 precision dependency:** the recommendation is supported only inside
  the current `float64` / `wp.float64` evidence envelope. It does not approve a
  lower-precision or mixed-precision production migration.

## What This Record Does Not Publish

This record does **not** publish:

- adaptive or exhaustive timestep search results
- generalized stable timestep limits
- general CPU-strategy parity or accuracy claims beyond the documented direct
  kernel cases
- P3/P4 temperature-feedback claims; E4-F6 independent-device,
  graph-capture/replay, or autodiff
  claims. Issue #1272's caller-owned signed `energy_transfer` diagnostic is
  shipped bookkeeping, not temperature feedback.
- strategy/runnable-level latent-heat support

Later phases can build on this measured baseline and recommendation without
redefining case shapes, metric names, or threshold meaning.
