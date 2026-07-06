# Condensation Stiffness Study Baseline

This note defines the reusable baseline for future GPU condensation stiffness
work. It names deterministic study cases, shared metric language, explicit
pass/fail rules, and the currently recorded timestep grid for the particle-only
GPU path. It still does not publish adaptive search results, integrator
recommendations, or gas-coupled conservation claims.

## Current Runtime Scope

- Production path: explicit fixed-step GPU condensation.
- Particle update: particle masses are clamped to remain non-negative.
- Gas update: the current Warp path is particle-only and does not yet update gas
  concentrations during production condensation.
- Baseline backend for this phase: `np.float64` inputs and Warp CPU execution.
- Accepted environment inputs: scalar `temperature` and `pressure`, or direct
  Warp arrays with shape `(n_boxes,)`.
- Study cases are deterministic fixed-shape builders that should be recreated
  from clean inputs after failed validation or intentionally unstable runs.

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

The current recorded sweep mirrors
`particula/gpu/kernels/tests/condensation_test.py` exactly. For each named
case, the tests:

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

This phase adds two deterministic prototype candidates in
`particula/gpu/kernels/tests/condensation_test.py`. They remain test-local
evidence only; the public `condensation_step_gpu(...)` runtime and package
export surface are unchanged.

| Candidate | Family | Buffer reuse | Determinism | Finite/non-negative masses | CPU-reference agreement | Graph capture | Autodiff note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_count_substeps_4` | Fixed-count explicit sub-stepping | Pass: one caller-owned `mass_transfer` array plus fixed-shape `work`/`accumulator` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 5e-2` at the baseline timestep and `max relative error <= 5e-2` across the recorded grid. | Pass: fixed loop count (`4`) and fixed-shape scratch keep the prototype graph-capture-friendly. | Clamp boundaries are still non-smooth, but there are no data-dependent loop counts. |
| `asymptotic_relaxation` | Asymptotic first-order bounded relaxation | Pass: one caller-owned `mass_transfer` array plus one fixed-shape `work` scratch reused across runs. | Pass: repeated runs for named stiffness cases produce identical arrays. | Pass: candidate tests require finite outputs and `>= 0` particle masses. | Pass within documented `rtol <= 3.5e-1` at the baseline timestep and `max relative error <= 3.5e-1` across the recorded grid. This looser bound keeps the candidate in prototype/evidence scope only. | Pass: fixed-shape algebra with no adaptive search or variable-length loops. | `exp(...)` relaxation remains differentiable away from the same clamp boundary, so it is a plausible autodiff target but not yet production-qualified. |

### Phase Boundary Decision

- Gas coupling is still deferred. No production gas-state update hook shipped in
  this issue.
- The exact split boundary remains the same: any production gas-coupled path
  must land with same-issue particle-plus-gas conservation regression coverage
  in `particula/integration_tests/condensation_particle_resolved_test.py`.
- The asymptotic candidate remains evidence-only because the tolerance required
  to track the current CPU/explicit reference is materially looser than the
  fixed-count candidate.
- Because the candidate evidence was credible in test-local helpers, no private
  production helper was added to `particula/gpu/kernels/condensation.py`.

## What This Phase Does Not Publish

This baseline does **not** publish:

- adaptive or exhaustive timestep search results
- generalized stable timestep limits
- final candidate integrator recommendations
- gas-coupled conservation claims that the current production path does not yet
  satisfy

Later phases can measure explicit Warp behavior against this shared baseline
without redefining case shapes, metric names, or threshold meaning.
