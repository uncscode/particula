# Condensation Stiffness Study and Recommendation Record

This note is the canonical decision record for future GPU condensation
integration work. It separates the current shipped runtime boundary, the
measured P2/P3 evidence, and the final P4 recommendation derived from that
evidence. The current production GPU path remains particle-only and `float64`
bounded; this page does not claim that gas-coupled production condensation has
shipped.

## Current Runtime Scope

- Production path: explicit fixed-step GPU condensation.
- Particle update: particle masses are clamped to remain non-negative.
- Gas update: the current Warp path is particle-only and does not yet update gas
  concentrations during production condensation.
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

This phase adds two deterministic prototype candidates implemented in
`particula/gpu/kernels/tests/_condensation_test_support.py` and collected
through `particula/gpu/kernels/tests/condensation_stiffness_test.py`. They
remain test-local evidence only; the public `condensation_step_gpu(...)`
runtime and package export surface are unchanged, no production gas-coupled
hook shipped, and no new private production helper was added.

| Candidate | Family | Buffer reuse | Determinism | Finite/non-negative masses | CPU-reference agreement | Graph capture | Autodiff note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_count_substeps_4` | Fixed-count explicit sub-stepping | Pass: one caller-owned `mass_transfer` array plus fixed-shape `work`/`accumulator` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 5e-2` at the baseline timestep and `max relative error <= 5e-2` across the recorded grid. These recorded bounds are evidence, not a shipped production tolerance. | Pass: fixed loop count (`4`) and fixed-shape scratch keep the prototype graph-capture-friendly. | Clamp boundaries are still non-smooth, but there are no data-dependent loop counts. |
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

### Recommended implementation foundation

Later GPU condensation implementation phases should build on
`fixed_count_substeps_4` as the preferred integration foundation.

Why this is the recommended path:

- It has the strongest recorded agreement with the current CPU/explicit
  reference, with documented `rtol <= 5e-2` at the baseline timestep and
  `max relative error <= 5e-2` across the recorded grid.
- It preserves deterministic fixed-shape execution with caller-owned buffer
  reuse and fixed scratch layouts across repeated runs.
- Its fixed loop count (`4`) is graph-capture-friendly in a way adaptive or
  data-dependent loop counts are not.
- It fits Warp autodiff expectations better than dynamic-loop schemes because
  the repeated work is statically bounded even though clamp boundaries remain
  non-smooth.
- Its evidence quality is materially stronger than the asymptotic alternative,
  so it provides the clearest foundation for later production work.

### Alternatives considered but not selected

- **Current single-step explicit update:** keep as the shipped baseline only.
  It is still useful as the production reference path, but it is not a strong
  forward-looking foundation for broader stiffness handling across the recorded
  particle-size range.
- **`asymptotic_relaxation`:** retain as evidence-only. It remains interesting
  for differentiability because the algebra is smooth away from clamp
  boundaries, but the measured CPU-reference agreement is materially looser
  (`rtol <= 3.5e-1` / `max relative error <= 3.5e-1`) than the fixed-count
  candidate.
- **Adaptive or dynamic-loop schemes:** defer. They conflict with the fixed
  iteration count and stable allocation layout preferred for Warp graph capture,
  and they complicate autodiff replay because backward passes do not reliably
  mirror data-dependent loop structure.

### Gas-coupled follow-up gate

The recommendation remains bounded to particle-only production condensation.
Any future production gas-coupled GPU path must land with the production hook
plus same-issue particle-plus-gas conservation regression coverage in
`particula/integration_tests/condensation_particle_resolved_test.py`.
Until that gate lands, roadmap and implementation guidance must not claim that
GPU condensation updates gas concentrations in production.

### Dependency boundaries that still limit the recommendation

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

Later phases can build on this measured baseline and recommendation without
redefining case shapes, metric names, threshold meaning, or the current
particle-only production boundary.
