# Condensation Stiffness Study and Recommendation Record

This note is the canonical decision record for GPU condensation integration.
The shipped production path uses four fixed equal substeps; P2/P3 candidate
comparisons remain historical evidence only. The current production GPU path
remains particle-only and `float64` bounded; this page does not claim that
gas-coupled production condensation has shipped.

## Current Runtime Scope

- Production path: explicit fixed-four GPU condensation. Every successful
  `condensation_step_gpu(...)` call performs exactly four `time_step / 4.0`
  substeps.
- Particle update: particle masses are clamped to remain non-negative.
- Gas update: the current Warp path is particle-only and does not yet update gas
  concentrations during production condensation.
- Per-substep order: optionally refresh composition-weighted surface tension,
  overwrite `gas.vapor_pressure`, refresh environment properties, produce a raw
  transfer proposal, then apply and accumulate its mass-clamped transfer.
- Transfer buffers: the resolved total transfer is cleared once after preflight
  and accumulates applied clamped transfer over all four substeps. A supplied
  total buffer is returned by identity; the separate work buffer retains only
  the final raw proposal.
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
- `_classify_particle_only_condensation_stiffness`: returns a stable/unstable
  result for the current particle-only GPU path and marks the
  `particle_only_update` caveat explicitly.

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
- verify that the current Warp path updates particle masses only and leaves gas
  concentration unchanged
- keep scalar `temperature` / `pressure` inputs for single-box cases and direct
  Warp `(n_boxes,)` arrays for the multi-box `droplet_like` case

| Case | Environment input mode | Timestep | Threshold | Classification | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `nanometer` | scalar `temperature` / `pressure` | `0.00005` | `1.0` | `stable` | Caller-owned buffer reused and overwritten; executed Warp gas state unchanged. |
| `nanometer` | scalar `temperature` / `pressure` | `0.05` | `1.0` | `stable` | Same fixed-shape particle-only path; executed Warp gas state unchanged. |
| `nanometer` | scalar `temperature` / `pressure` | `50.0` | `1.0` | `stable` | Same caller-owned buffer contract; executed Warp gas state unchanged. |
| `accumulation_mode` | scalar `temperature` / `pressure` | `0.004` | `1.0` | `stable` | Caller-owned buffer reused and overwritten; executed Warp gas state unchanged. |
| `accumulation_mode` | scalar `temperature` / `pressure` | `0.4` | `1.0` | `stable` | Same fixed-shape particle-only path; executed Warp gas state unchanged. |
| `accumulation_mode` | scalar `temperature` / `pressure` | `40.0` | `1.0` | `stable` | Same caller-owned buffer contract; executed Warp gas state unchanged. |
| `droplet_like` | direct Warp `(n_boxes,)` arrays | `0.04` | `1.0` | `stable` | Multi-box direct-array environment inputs stay supported; executed Warp gas state unchanged. |
| `droplet_like` | direct Warp `(n_boxes,)` arrays | `4.0` | `1.0` | `stable` | Same fixed-shape particle-only path; executed Warp gas state unchanged. |
| `droplet_like` | direct Warp `(n_boxes,)` arrays | `400.0` | `1.0` | `stable` | Same caller-owned buffer contract; executed Warp gas state unchanged. |

Across the current recorded grid, the executable tests observe the same
particle-only maximum fractional-mass-change magnitude (`1.0`) for every row.
The baseline therefore applies one inclusive threshold (`1.0`) across the full
grid and records every row as `stable` under that shared rule. Separate unit
tests still cover the `unstable` branch for larger fractional changes,
zero-mass growth, and non-finite values. This is recorded-grid evidence for the
current fixed-shape particle-only path, not a gas-coupled conservation result
and not a general stable-timestep limit for other cases.

## Candidate Evaluation Evidence

P2/P3 recorded two deterministic prototype candidates implemented in
`particula/gpu/kernels/tests/_condensation_test_support.py` and collected
through `particula/gpu/kernels/tests/condensation_stiffness_test.py`. They
remain test-local evidence only; the public `condensation_step_gpu(...)`
runtime and package export surface are unchanged, no production gas-coupled
hook shipped, and no new private production helper was added.

| Candidate | Family | Buffer reuse | Determinism | Finite/non-negative masses | CPU-reference agreement | Graph capture | Autodiff note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_count_substeps_4` | Historical P2/P3 fixed-count explicit sub-stepping evidence; its fixed-four behavior is now shipped. | Pass: one caller-owned `mass_transfer` array plus fixed-shape `work`/`accumulator` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 5e-2` at the baseline timestep and `max relative error <= 5e-2` across the recorded grids for `nanometer`, `accumulation_mode`, and `droplet_like`. These bounds are case-specific evidence, not a general accuracy tolerance or stable-timestep limit. | Not evaluated as production graph-capture evidence. | Not evaluated as production autodiff evidence. |
| `asymptotic_relaxation` | Asymptotic first-order bounded relaxation | Pass: one caller-owned `mass_transfer` array plus one fixed-shape `work` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 3.5e-1` at the baseline timestep and `max relative error <= 3.5e-1` across the recorded grid. This looser bound is recorded as prototype evidence only and is not suitable for a production recommendation by itself. | Pass: fixed-shape algebra with no adaptive search or variable-length loops. | `exp(...)` relaxation remains differentiable away from the same clamp boundary, so it is a plausible autodiff target but not yet production-qualified. |

### Phase Boundary Decision

- Gas coupling is still deferred. No production gas-state update hook shipped in
  this issue.
- `condensation_step_gpu(...)` remains particle-only in production, so the new
  candidate coverage should be read strictly as deterministic evaluation
  evidence rather than a runtime capability change.
- The exact split boundary remains the same: any production gas-coupled path
  must land with same-issue particle-plus-gas conservation regression coverage
  in `particula/integration_tests/condensation_particle_resolved_test.py`.
- The asymptotic candidate remains evidence-only because the tolerance required
  to track the current CPU/explicit reference is materially looser than the
  fixed-count candidate.
- Because the candidate evidence was credible in test-local helpers, no private
  production helper was added to `particula/gpu/kernels/condensation.py`.

## Final Recommendation

### Shipped fixed-four implementation

`fixed_count_substeps_4` is the shipped production integration behavior of
`condensation_step_gpu(...)`, not a future prototype. Each valid call uses four
equal substeps, updates particle masses in place, leaves
`gas.concentration` unchanged, and keeps production calculations device
resident.

Why this is the recommended path:

- The historical P2/P3 comparison recorded `rtol <= 5e-2` at the baseline
  timestep and maximum relative error `<= 5e-2` only for the named
  `nanometer`, `accumulation_mode`, and `droplet_like` grids.
- Successful calls preserve supplied total-buffer identity, clear that total
  once after preflight, and accumulate applied mass-clamped transfer. A
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

### Gas-coupled follow-up gate

The recommendation remains bounded to particle-only production condensation.
Any future production gas-coupled GPU path must land with the production hook
plus same-issue particle-plus-gas conservation regression coverage in
`particula/integration_tests/condensation_particle_resolved_test.py`.
Until that gate lands, roadmap and implementation guidance must not claim that
GPU condensation updates gas concentrations in production.

### Downstream gates and dependency boundaries

E4-F4 P2 ships a low-level, per-substep latent-heat rate correction with
CPU-oracle/Warp parity. It does not imply the deferred P3/P4 temperature
feedback or energy bookkeeping, E4-F5 gas coupling and particle-plus-gas
conservation, E4-F6 independent-device plus graph/autodiff evidence, or E4-F7
strategy/runnable and final support work.

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
- gas-coupled conservation claims that the current production path does not yet
  satisfy
- P3/P4 temperature-feedback or energy-bookkeeping claims; E4-F5 gas coupling
  or conservation claims; or E4-F6 independent-device, graph-capture/replay,
  or autodiff claims
- strategy/runnable-level latent-heat support

Later phases can build on this measured baseline and recommendation without
redefining case shapes, metric names, threshold meaning, or the current
particle-only production boundary.
