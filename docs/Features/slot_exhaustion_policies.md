# Fixed-Capacity Slot Exhaustion Primitives

## Complete direct-process context

Fixed-shape slot sidecars remain caller-owned in the illustrative [direct-process example](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_complete_process_sequence.py). See [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and the [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py). This documentation does not add resizing, hidden transfer, CPU fallback, or a production scheduler.

This guide documents shipped fixed-capacity CPU primitives and direct Warp
primitives. The concrete CPU-only GPU diagnostics
`particula.dynamics.nucleation.particle_source.commit_particle_source`
transaction is a shipped consumer that composes slot discovery, policy
resolution, and activation, but is not package-exported or a public API.
Fixed-slot management owns the authoritative discovery, free-index classification, and
activation boundary. The shipped public, CPU-only one-box
`particula.dynamics.Nucleation` runnable consumes these primitives; scheduler
integration and broader policy orchestration remain deferred.

For the supported one-box CPU consumer and its public construction boundary,
see the [CPU Nucleation Strategy System](nucleation_strategy_system.md).

## Native population boundary

Native resampling and representative-volume scaling operate on explicitly
tagged `particle_resolved` counts at a positive physical volume. PMF/PDF
storage is not interchangeable with those counts. A selected scale multiplies
existing counts, provisional source counts and volume together, preserving
pre-existing number/mass density and gas concentration. This is distinct from
physical volume evolution, which retains particle counts and dilutes gas.
The native source consumer supplies explicit ordered species mapping; see
[native source integration](native_scientific_processes.md#native-source-integration-v030).
Direct and prepared GPU exhaustion calls reject untagged, discrete, and PDF
metadata before payload access. Native source staging requires an explicit
species binding and carries raw event counts through scaling; final demand
density is computed from those counts and the current volume. The import
snippets below require the v0.3.0 development checkout, with Warp installed for GPU
imports.

## CPU planning contract

The CPU resolver exposes planning records only:

```python
from particula.particles.exhaustion import (
    ExhaustionControls,
    ExhaustionInputs,
    resolve_exhaustion,
)
```

`ExhaustionControls()` defaults to `resampling=True` and
`representative_volume_scaling=False`. `resolve_exhaustion()` validates all
input sidecars before returning an immutable plan and never writes particles.
It does not discover slots, activate slots, construct a source, or apply a
plan.

| Requested capacity | Controls and releasable capacity | Planning record |
| --- | --- | --- |
| `requested_count <= free_count` | Any control values | Activation-prefix record from the exact free-index prefix |
| Exhausted | Resampling enabled and releasable count is sufficient | Deferred resampling record |
| Exhausted | Resampling is unavailable or insufficient; scaling enabled | Deferred scaling record |
| Exhausted | Neither policy can represent the request | `ValueError` before a plan is returned |

Thus, resampling takes precedence over enabled scaling only for exhausted
capacity. These are CPU resolver selection rules, not runtime orchestration.
They do not silently truncate requested demand. Deferred records do not release
or activate slots; their release indices are empty and their scale factor is
`nan`.

`ExhaustionInputs` uses NumPy `int32` counts shaped `(B,)`.
`free_indices` is NumPy `int32` shaped `(B, N)`: each row has an ascending
free-index prefix followed by `-1` unused entries. The public plan records are
intentionally limited to the resolver contract.

## CPU resampling and scaling boundaries

`plan_resampling`, `apply_resampling`, and
`apply_representative_volume_scaling` are concrete helpers in
`particula.particles.exhaustion`; they are not package exports or a high-level
policy API. `plan_resampling` is read-only and returns a private detached
implementation plan, so applications must not expose or depend on
`_ResamplingPlan` or `_ResamplingBoxPlan`. `apply_resampling` validates every
box and the target state before its first assignment, then commits the complete
fixed-capacity remap.

CPU CPU activation consumes writable `ParticleData` NumPy `float64` arrays with these
schemas: `masses (B, N, S)`, `concentration (B, N)`, `charge (B, N)`,
`density (S,)`, and `volume (B,)`. The arrays must not share writable memory.
An inactive slot has zero concentration and literal-zero mass and charge.
These are CPU-only planning and commit helpers, not a CPU-to-Warp transfer or
parity interface.

GPU activation scaling consumes writable, contiguous, nonaliasing NumPy sidecars:

| Sidecar | CPU dtype and shape |
| --- | --- |
| `provisional_source_demand`, `requested_scale`, `minimum_scale`, `minimum_volume`, `resolved_scale` | `float64 (B,)` |
| `scaling_required` | `bool (B,)` |

CPU GPU activation rejects invalid schemas, physical values, bounds, aliasing, or selected
minimum volumes before diagnostic or particle writes. A row is selected only
when `scaling_required` is true and demand is positive. Even when no rows are
selected, a successful call writes `resolved_scale`: it is `1.0` for unselected
rows.

## Direct Warp primitives

The only exhaustion primitive exported by `particula.gpu.kernels` is:

```python
from particula.gpu.kernels import resampling_step_gpu
from particula.gpu.kernels.exhaustion import (
    ResamplingBuffers,
    representative_volume_scaling_step_gpu,
)
```

`ResamplingBuffers` and `representative_volume_scaling_step_gpu` are
concrete-module-only. The direct functions consume caller-owned, same-device,
fixed-shape particle state, counts, sidecars, and buffers. Callers perform
CPU/Warp transfer and synchronization. In particular, synchronize Warp before
observing successful asynchronous GPU activation results.

Warp CPU activation accepts same-device `int32 (B,)` `required_release_counts`. Its
caller-owned `ResamplingBuffers` fields are all distinct, same-device arrays:

| Fields | dtype and shape |
| --- | --- |
| `retained_counts`, `released_counts`, `planning_status` | `int32 (B,)` |
| `retained_indices`, `released_indices`, `sorted_indices` | `int32 (B, N)` |
| `replacement_masses` | `float64 (B, N, S)` |
| `replacement_concentration`, `replacement_charge`, `source_radii` | `float64 (B, N)` |
| `radius_cubed_relative_error`, `mean_radius_relative_error`, `surface_relative_error`, `diversity_absolute_error` | `float64 (B,)` |

After successful nonzero-demand planning, unused index lanes are `-1` and
unused replacement lanes are zero. CPU activation preflight rejects before caller writes.
A zero-demand call is write-free. Planning may alter documented buffer lanes;
if a planning diagnostic fails, particles remain unchanged and the commit is
skipped. No rollback is promised after a commit launch.

Warp GPU activation sidecars are same-device, contiguous, nonaliasing arrays: all numeric
sidecars (`provisional_source_demand`, `requested_scale`, `minimum_scale`,
`minimum_volume`, and `resolved_scale`) are `float64 (B,)`, while
`scaling_required` is `int32 (B,)` containing `0` or `1`. Preflight rejects
before a caller write. Successful GPU activation writes `resolved_scale` as the requested
factor for selected rows and `1.0` otherwise; no-selected-row calls are
otherwise write-free.

## Mutation boundaries

CPU slot classification resolution and CPU CPU activation `plan_resampling` are read-only. CPU CPU activation
`apply_resampling` validates every target box and its detached plan before its
first assignment, then commits the all-box remap. CPU GPU activation rejects invalid input
before writing either diagnostics or particle state.

Warp CPU activation and GPU activation preflight rejection likewise preserves caller-owned state. A
zero-demand Warp CPU activation call is write-free. A failed Warp CPU activation planning diagnostic
may alter documented buffer lanes, but it preserves particles and skips the
commit. Neither direct Warp primitive promises rollback after a commit kernel
has launched.

## Accounting and diagnostics

The CPU float64 ledger uses concentration-weighted reductions:

\[
N = \sum w, \qquad M_s = \sum w m_s, \qquad Q = \sum w q.
\]

Here, $w$ is the dimensionless computational particle weight/count, $m_s$ is
species mass `[kg]`, and $V$ is box volume `[m^3]`. Thus $N$, $M_s$, and $Q$
are box totals, while the helper-normalized values $N/V$, $M_s/V$, and $Q/V$
are their corresponding volume densities. Supported resolved storage contains
counts; apply volume normalization exactly once when comparing with gas density.

CPU activation has tolerance-bounded conservation diagnostics. Its radius-cubed,
mean-radius, surface-area, and Riemer-diversity diagnostics are separate,
non-exact properties, not extra physics guarantees. The four configurable
bounds are `radius_cubed_relative_error`, `mean_radius_relative_error`,
`surface_relative_error`, and `diversity_absolute_error`. Warp CPU activation records the
result in `planning_status` (`0` is success; nonzero values identify inventory,
moment, diversity, or non-finite planning failures).

For a selected GPU activation row, with $s = requested\_scale$,

\[
V_{new} = s V_{old}, \qquad w_{new} = s w_{old}, \qquad
demand_{new} = s\,demand_{old},
\]

where $0 < minimum\_scale \le requested\_scale = s \le 1$ and
`minimum_volume` is in `[m^3]`. `resolved_scale` is $s$ for a selected row and
`1.0` otherwise. Masses, charge, density, configuration sidecars, and
unselected rows are preserved. Later unscaled source work targets
`pre_state + source`; selected GPU activation work targets `s * pre_state + source`.
These primitives do not construct a source, deplete gas, activate slots, or
implement nucleation physics.

## Explicitly deferred capabilities

The primitives do not provide dynamic allocation, resizing, compaction,
particle-state transfers, CPU fallback, a high-level runnable, scheduler,
backend orchestration, graph capture, autodiff, performance guarantees, or
exact CPU/Warp/CUDA RNG replay. Validation and planning may perform bounded
scalar status readbacks. They do not establish a loop that invokes a policy
before exhaustion.

For the delivered-versus-deferred GPU roadmap, see
[Data-Oriented GPU Roadmap](Roadmap/data-oriented-gpu.md).

## Focused validation

```bash
pytest particula/particles/tests/exhaustion_test.py \
  particula/gpu/kernels/tests/exhaustion_test.py -q -Werror
mkdocs build --strict
```
