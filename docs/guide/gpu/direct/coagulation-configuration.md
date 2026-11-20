# GPU coagulation configuration and sidecar ownership

Native resolved coagulation combines Brownian motion, charged Brownian motion,
differential sedimentation and turbulent shear in one candidate/acceptance stream.
All 15 nonempty subsets are executable, including Brownian + sedimentation +
shear. This development capability removes the former 256 and 16,384 active-slot
limits. It does not change the supported population measure or collision physics.

```python
from particula.gpu.kernels import coagulation_step_gpu
from particula.gpu.kernels.coagulation import (
    CoagulationMechanismConfig,
    CoagulationWorkConfig,
    coagulation_preflight_gpu,
)

mechanisms = CoagulationMechanismConfig(
    mechanisms=("brownian", "sedimentation_sp2016", "turbulent_shear_st1956")
)
work = CoagulationWorkConfig(max_trials_per_box=65536)
inputs = dict(
    temperature=283.15, pressure=101325.0,
    mechanism_config=mechanisms, work_config=work,
    turbulent_dissipation=1e-4, fluid_density=1.2,
    max_collisions=particles.masses.shape[1] // 2,
)
report = coagulation_preflight_gpu(particles, time_step=0.1, **inputs)
# Inspect report["admitted"] before calling the step at the same state.
if report["admitted"].all():
    particles, pairs, counts = coagulation_step_gpu(
        particles, time_step=0.1, **inputs
    )
```

The configuration and preflight helpers are concrete-module APIs, not package
exports. The return tuple remains `(particles, collision_pairs, n_collisions)`.
Supplied outputs are returned by identity. Temperature and pressure accept
scalars, supported floating Warp vectors `(n_boxes,)`, or a same-device
`WarpEnvironmentData` with the direct arguments set to `None`. Physical box
volume belongs to `ParticleData`; an explicit volume must agree with it.

## Physics and represented counts

| Identifier | Bit | Physical model |
| --- | --- | --- |
| `brownian` | 1 | Neutral Brownian coagulation |
| `charged_hard_sphere` | 2 | Complete charged Brownian rate with selected fit |
| `sedimentation_sp2016` | 4 | Slip-corrected Stokes differential settling, efficiency 1 |
| `turbulent_shear_st1956` | 8 | Saffman–Turner turbulent shear |

`charged_model` accepts `hard_sphere`, `dyachkov2007`, `gatti2008`,
`gopalakrishnan2012`, or `chahl2019`. Nondefault fits require the charged
mechanism. These are complete Brownian rates: adding ordinary Brownian sums
both complete rates. Do not add a charged term merely to obtain an allowed
combination. Mechanisms normalize to the table's order; duplicates and unknown
identifiers reject before particle access.

Only `particle_resolved` is supported. Positive concentration is represented
count, and must be equal within each physical box, including non-unit counts.
Unequal weights, discrete populations and continuous PDFs are unsupported.
No fallback, physical-box subdivision, resampling or weight change is implicit.
Species masses and signed charge are conserved with represented-count weighting.

Shear requires positive finite `turbulent_dissipation` [m²/s³] and
`fluid_density` [kg/m³], supplied as floating scalars or same-device fp64 vectors.
Settling uses `π(r_i+r_j)²|v_i-v_j|` [m³/s]. Direct settling uses each particle's
effective material density; prepared strategies retain their native mean-density
convention. Compare CPU and GPU observables only under matching assumptions.

## Resources and rejection

Parallel tiles evaluate every active unordered pair and reduce the maximum of
the summed rate. Property and active-index storage scale linearly with slot
capacity; reduction scratch uses at most 4,096 lanes per box. Total pair work is
still quadratic. Tile sizes are not scientific population limits. Device memory,
signed slot indexing and representable allocation sizes remain real limits.

For active count `A`, equal represented count `w`, volume `V` [m³], majorant `M`
[m³/s] and timestep `dt` [s], expected trial demand is
`lambda = M * A * (A - 1) / 2 * w * dt / V`.
Admission requires `ceil(lambda)` to fit the trial budget, collision capacity
and available disjoint pairs. This deliberately conservative bound ensures the
whole scheduled pass fits even if every proposal is accepted.

`CoagulationWorkConfig` defaults to 65,536 trials per box and 256 MiB for collision
outputs. These are configurable resource budgets. A request exceeding the byte
budget raises an actionable allocation error instead of shrinking the buffer.
An overloaded step raises `CoagulationBudgetError` with the box, required and
available trials, and a budget-admissible timestep. Infinite demand rejects.
The fixed-step API rejects the whole call on excess demand.
A smaller admitted timestep still requires a separate convergence study.

Preflight returns active counts, unordered pair evaluations, expected/required
trials, available trials, budget-admissible timesteps, admission flags and the
bound workspace/collision bytes. It allocates temporary workspace and downloads
compact diagnostics, but leaves caller particles, outputs and RNG unchanged.
Its state-dependent result must be recomputed after the population changes.

## Persistent execution

Supplied RNG streams advance across successful calls. `initialize_rng=True`
requests a reset only after admission succeeds. Invalid and over-budget calls
preserve caller particles, outputs and RNG, even when reset was requested.
Omitted RNG uses convenience call-local storage. Allocation, transfer and
synchronization boundaries remain explicit; compact preflight readback does not
copy the simulation to CPU for execution.

Pass `coagulation_work_config=work` to a runnable's `prepare()` to freeze resource
budgets. Prepared combinations support unique subsets of the same mechanisms.
Scratch is allocated during preparation and reused during resident execution.
Per-step device admission blocks dependent writers on failure; observation
faults the owner. Earlier successful processes or steps are not rolled back.
Checkpoints retain budget configuration and sampler compatibility; incompatible
older sampler checkpoints reject rather than silently changing continuation.

The [worked resource-admission example](../../../studies/validation/coagulation/native-sampling/scalable_coagulation.py)
uses explicit Warp CPU and can be run on requested native CUDA. The
[validation record](../../../contribute/validation/gpu/coagulation.md) distinguishes
historical qualification from new checks. Native CUDA capture and performance
claims require actual CUDA evidence; Warp CPU is not a substitute.

### Executed work

Pass a separate `work_counts` Warp int32 array of shape `(n_boxes, 2)` to
`coagulation_step_gpu` to observe scheduled and attempted candidate trials.
The returned `n_collisions` contains accepted, applied collisions. These are
last-step counts, distinct from preflight expectations. Admission rejection
preserves caller-owned counters, RNG and collision outputs. Counters do not
consume random draws. Download them only at an explicit observation boundary.

## Dispersion and dielectric capture

The `van_der_waals` mechanism uses bit 16 and takes `hamaker_constant` [J]
and `relative_permittivity` on `CoagulationMechanismConfig`. It supports
standalone use and combinations with settling, shear, or both. Each pair's
current radii, masses and signed charges determine the rate. Preparation
uploads a dimensionless dielectric geometry table. The exact tiled maximum
and the sampler use the same fp64 pair calculation. An out-of-domain pair
rejects the step before collision outputs, particle storage or RNG change.
[Validation and ranges](../../../studies/validation/coagulation/dispersion/index.md)


## Adaptive direct intervals

Call `adaptive_coagulation_step_gpu(particles, temperature, pressure, time_step,
subcycling=policy, work_config=work, rng_states=rng)` from
`particula.gpu.kernels.coagulation`. Each call allocates interval scratch and
returns an `AdaptiveCoagulationResult`. Its `collision_pairs` and `n_collisions`
are device arrays for the complete interval. Per-box exclusive end offsets in
`diagnostics["pair_end_offsets"]` divide the ordered log into passes. Caller
supplied single-pass outputs retain the last pass. Reuse a persistent RNG
array across calls. If a policy limit rejects after tentative merges, the call
restores particle, RNG and supplied output arrays before raising.

The direct memory reservation includes sampler scratch, history and copies of
mutable caller storage. Original caller allocations belong to the caller.
Prepared execution retains its buffers across steps and supports CUDA graph
replay. The standalone resident adapter continues to run one fixed pass.
[Prepared lifecycle and retry](../prepared.md#adaptive-coagulation-intervals)
