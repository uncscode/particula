# particula.gpu.kernels.coagulation

*module*

Native equal-count coagulation with tiled exact additive pair reductions.

All 15 nonempty mechanism subsets share one candidate and acceptance stream.
Work budgets reject before particle, collision-output or RNG writers. Direct
calls observe compact admission diagnostics; resident calls latch device errors.
Configuration and preflight helpers are concrete-module APIs. No CPU fallback,
hidden particle transfer, box subdivision or weight change is performed.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1-L1)

## particula.gpu.kernels.coagulation.MAX_COLLISION_PAIR_BUFFER_BYTES

*attribute* · *module attribute*

```python
MAX_COLLISION_PAIR_BUFFER_BYTES = 256 * 1024 * 1024
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L114-L114)

## particula.gpu.kernels.coagulation.MAX_SCHEDULED_TRIALS_PER_BOX

*attribute* · *module attribute*

```python
MAX_SCHEDULED_TRIALS_PER_BOX = 65536
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L115-L115)

## particula.gpu.kernels.coagulation.AdaptiveCoagulationResult

*class* · *dataclass*

```python
class AdaptiveCoagulationResult
```

Outer interval output with ordered pairs and per-box boundaries.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2958-L2965)

### particula.gpu.kernels.coagulation.AdaptiveCoagulationResult.collision_pairs

*attribute* · *instance attribute*

```python
collision_pairs: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2963-L2963)

### particula.gpu.kernels.coagulation.AdaptiveCoagulationResult.diagnostics

*attribute* · *instance attribute*

```python
diagnostics: dict[str, Any]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2965-L2965)

### particula.gpu.kernels.coagulation.AdaptiveCoagulationResult.n_collisions

*attribute* · *instance attribute*

```python
n_collisions: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2964-L2964)

### particula.gpu.kernels.coagulation.AdaptiveCoagulationResult.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2962-L2962)

### particula.gpu.kernels.coagulation.AdaptiveCoagulationResult.__init__

*method*

```python
def __init__(particles: Any, collision_pairs: Any, n_collisions: Any, diagnostics: dict[str, Any]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L0-L0)

## particula.gpu.kernels.coagulation._PreparedCoagulationCall

*class* · *dataclass*

```python
class _PreparedCoagulationCall
```

Privately freeze validated coagulation inputs and scratch for enqueue.

The carrier stores the exact caller-owned fields, outputs, device, and
selector arguments established during preparation. Enqueue therefore does
not repeat validation, allocation, normalization, RNG initialization, or
resource lookup. Rebinding a source-container attribute cannot redirect an
already prepared writer.

**Attributes**

- `particles` (`Any`) — Caller-owned particle container returned by the operation.
- `masses` (`Any`) — Frozen particle-mass array used by the apply kernel.
- `concentration` (`Any`) — Frozen particle-concentration array.
- `charge` (`Any`) — Frozen particle-charge array.
- `collision_pairs` (`Any`) — Collision-pair output buffer.
- `n_collisions` (`Any`) — Per-box collision-count output buffer.
- `n_boxes` (`int`) — Number of simulation boxes.
- `max_collisions` (`int`) — Effective collision capacity per box.
- `device` (`Any`) — Active Warp device for all launches.
- `selector_inputs` (`tuple[Any, ...]`) — Frozen arguments for the selector kernel.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2370-L2420)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.adaptive

*attribute* · *class attribute* · *instance attribute*

```python
adaptive: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2406-L2406)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.brownian_parallel

*attribute* · *class attribute* · *instance attribute*

```python
brownian_parallel: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2404-L2404)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.budget_failure

*attribute* · *class attribute* · *instance attribute*

```python
budget_failure: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2408-L2408)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.budget_report

*attribute* · *class attribute* · *instance attribute*

```python
budget_report: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2407-L2407)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.charge

*attribute* · *instance attribute*

```python
charge: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2396-L2396)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.collision_pairs

*attribute* · *instance attribute*

```python
collision_pairs: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2397-L2397)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.concentration

*attribute* · *instance attribute*

```python
concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2395-L2395)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2401-L2401)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.failure

*attribute* · *class attribute* · *instance attribute*

```python
failure: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2403-L2403)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.masses

*attribute* · *instance attribute*

```python
masses: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2394-L2394)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.max_collisions

*attribute* · *instance attribute*

```python
max_collisions: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2400-L2400)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.n_boxes

*attribute* · *instance attribute*

```python
n_boxes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2399-L2399)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.n_collisions

*attribute* · *instance attribute*

```python
n_collisions: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2398-L2398)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2393-L2393)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.rng_pending

*attribute* · *class attribute* · *instance attribute*

```python
rng_pending: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2409-L2409)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.rng_seed

*attribute* · *class attribute* · *instance attribute*

```python
rng_seed: int = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2410-L2410)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.selector_inputs

*attribute* · *instance attribute*

```python
selector_inputs: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2402-L2402)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.work_config

*attribute* · *class attribute* · *instance attribute*

```python
work_config: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2405-L2405)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.work_counts

*attribute* · *class attribute* · *instance attribute*

```python
work_counts: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2411-L2411)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.__init__

*method*

```python
def __init__(particles: Any, masses: Any, concentration: Any, charge: Any, collision_pairs: Any, n_collisions: Any, n_boxes: int, max_collisions: int, device: Any, selector_inputs: tuple[Any, ...], failure: Any = None, brownian_parallel: Any = None, work_config: Any = None, adaptive: Any = None, budget_report: Any = None, budget_failure: Any = None, rng_pending: Any = None, rng_seed: int = 0, work_counts: Any = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L0-L0)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.check_admission

*method*

```python
def check_admission() -> None
```

Observe compact diagnostics at an explicit host boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2413-L2420)

## particula.gpu.kernels.coagulation._bound_scheduled_trials

*function*

```python
def _bound_scheduled_trials(expected_trials: Any) -> Any
```

Retain the legacy trial clamp for diagnostic oracles, not production.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L120-L130)

## particula.gpu.kernels.coagulation._charged_majorant_from_active_pairs

*function*

```python
def _charged_majorant_from_active_pairs(active_indices: Any, box_idx: Any, active_count: Any, radii: Any, total_masses: Any, charges: Any, temperature: Any, pressure: Any, boltzmann_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, gas_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any) -> Any
```

Return the finite charged-rate maximum over unique compact active pairs.

This read-only helper resolves compact ranks through ``active_indices`` and
sanitizes every charged hard-sphere candidate. Charge and mass dependence
prevents a proved extrema-only bound, so the intentional O(n²) scan favors
a correctness-first majorant. Fewer than two active ranks and invalid,
nonpositive candidates contribute zero.

**Parameters**

- `active_indices` (`Any`) — Compact particle indices ``(n_boxes, n_particles)``.
- `box_idx` (`Any`) — Index of the simulation box to scan.
- `active_count` (`Any`) — Number of compact active ranks in ``box_idx``.
- `radii` (`Any`) — Particle radii ``(n_boxes, n_particles)`` [m].
- `total_masses` (`Any`) — Total particle masses ``(n_boxes, n_particles)`` [kg].
- `charges` (`Any`) — Particle charges ``(n_boxes, n_particles)`` in elementary charge counts.
- `temperature` (`Any`) — Per-box gas temperatures ``(n_boxes,)`` [K].
- `pressure` (`Any`) — Per-box gas pressures ``(n_boxes,)`` [Pa].
- `boltzmann_constant` (`Any`) — Boltzmann constant [J/K].
- `elementary_charge_value` (`Any`) — Elementary charge [C].
- `electric_permittivity` (`Any`) — Vacuum permittivity [F/m].
- `gas_constant` (`Any`) — Universal gas constant [J/(mol·K)].
- `molecular_weight_air` (`Any`) — Molecular weight of air [kg/mol].
- `ref_viscosity` (`Any`) — Reference gas viscosity [Pa·s].
- `ref_temperature` (`Any`) — Reference viscosity temperature [K].
- `sutherland_constant` (`Any`) — Sutherland viscosity constant [K].

**Returns**

- (`Any`) — Finite, nonnegative maximum charged hard-sphere rate for the selected
- (`Any`) — box, or zero when no positive finite selected-pair rate exists.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L336-L419)

## particula.gpu.kernels.coagulation._checked_positive_finite_add

*function*

```python
def _checked_positive_finite_add(left: Any, right: Any) -> Any
```

Add two sanitized fp64 terms, failing closed on invalid arithmetic.

Additive mechanism terms are independently optional, but an overflowing
aggregate cannot safely be used as a selector rate or majorant.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L146-L157)

## particula.gpu.kernels.coagulation._coagulation_selector_impl

*function*

```python
def _coagulation_selector_impl(masses: Any, concentration: Any, density: Any, volume: Any, temperature: Any, pressure: Any, gas_constant: Any, boltzmann_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, time_step: Any, radii: Any, diffusivities: Any, g_terms: Any, speeds: Any, settling_velocities: Any, turbulent_dissipation: Any, fluid_density: Any, total_masses: Any, charge: Any, active_indices: Any, collision_pairs: Any, n_collisions: Any, rng_states: Any, mechanism_mask: Any, collision_capacity: Any, box_idx: int) -> None
```

Select stochastic coagulation pairs for each box.

Candidate pairs are drawn by rank within the current active set so each
scheduled trial proposes two distinct active particles without retrying on
inactive slots. The sampler computes one finite positive total majorant per
box and one finite positive total pair rate per candidate. A valid candidate
receives exactly one acceptance draw; invalid rates, invalid majorants, and
rates above the majorant are skipped without collision-buffer or active-set
mutation. All fifteen nonempty mechanism masks share this diagnostic
candidate/acceptance path. Production obtains maxima from tiled reductions.

**Parameters**

- `masses` (`Any`) — Particle masses array ``(n_boxes, n_particles, n_species)``.
- `concentration` (`Any`) — Particle concentrations ``(n_boxes, n_particles)``.
- `density` (`Any`) — Species densities ``(n_species,)``.
- `volume` (`Any`) — Per-box volumes ``(n_boxes,)``.
- `temperature` (`Any`) — Per-box gas temperatures ``(n_boxes,)`` [K].
- `pressure` (`Any`) — Per-box gas pressures ``(n_boxes,)`` [Pa].
- `gas_constant` (`Any`) — Universal gas constant [J/(mol·K)].
- `boltzmann_constant` (`Any`) — Boltzmann constant [J/K].
- `molecular_weight_air` (`Any`) — Molecular weight of air [kg/mol].
- `ref_viscosity` (`Any`) — Reference viscosity at STP [Pa·s].
- `ref_temperature` (`Any`) — Reference temperature [K].
- `sutherland_constant` (`Any`) — Sutherland constant [K].
- `elementary_charge_value` (`Any`) — Elementary-charge magnitude [C].
- `electric_permittivity` (`Any`) — Vacuum electric permittivity [F/m].
- `time_step` (`Any`) — Coagulation time step [s].
- `radii` (`Any`) — Output particle radii ``(n_boxes, n_particles)``.
- `diffusivities` (`Any`) — Output diffusivities ``(n_boxes, n_particles)``.
- `g_terms` (`Any`) — Output collection terms ``(n_boxes, n_particles)``.
- `speeds` (`Any`) — Output mean thermal speeds ``(n_boxes, n_particles)``.
- `settling_velocities` (`Any`) — Private fp64 settling-velocity scratch ``(n_boxes, n_particles)``. When separately allocated for SP2016 sedimentation, this scratch is zero-initialized before selector launch. The selector populates valid active slots and does not clear this scratch.
- `turbulent_dissipation` (`Any`) — Per-box turbulent kinetic-energy dissipation rate ``(n_boxes,)`` [m^2/s^3]. Read when ST1956 turbulent shear is enabled.
- `fluid_density` (`Any`) — Per-box carrier-fluid density ``(n_boxes,)`` [kg/m^3]. Read when ST1956 turbulent shear is enabled to derive kinematic viscosity.
- `active_indices` (`Any`) — Output compact active indices ``(n_boxes, n_particles)``.
- `collision_pairs` (`Any`) — Output collision indices ``(n_boxes, max_collisions, 2)``.
- `n_collisions` (`Any`) — Output collision counts ``(n_boxes,)``.
- `rng_states` (`Any`) — Per-box RNG states ``(n_boxes,)`` mutated in place during pair selection. Reusing this buffer across calls preserves caller-owned persistent state unless it is reset before launch.
- `total_masses` (`Any`) — Private ``wp.float64`` total-mass scratch ``(n_boxes, n_particles)``. The kernel clears each slot and stores the sum over species only for active particles.
- `charge` (`Any`) — Caller-owned signed elementary-charge counts ``(n_boxes, n_particles)``.
- `mechanism_mask` (`Any`) — Fixed internal sampler mask. Public singleton, two-way, and four-way approved masks are executable.
- `collision_capacity` (`Any`) — Maximum accepted collisions per box for this call.
- `box_idx` (`int`) — Box index supplied by the kernel launch entry point.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L907-L1208)

## particula.gpu.kernels.coagulation._compact_applied_collision_pairs_guarded

*function*

```python
def _compact_applied_collision_pairs_guarded(collision_pairs: Any, n_collisions: Any, failure: wp.array(dtype=wp.int32)) -> None
```

Remove rejected collision entries after the apply pass.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1707-L1729)

## particula.gpu.kernels.coagulation._compact_applied_collision_pairs_kernel

*function*

```python
def _compact_applied_collision_pairs_kernel(collision_pairs: Any, n_collisions: Any) -> None
```

Remove rejected collision entries after the apply pass.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1684-L1704)

## particula.gpu.kernels.coagulation._enqueue_prepared_coagulation_call

*function*

```python
def _enqueue_prepared_coagulation_call(prepared: _PreparedCoagulationCall) -> tuple[Any, Any, Any]
```

Evaluate rates, admit all boxes, then enqueue guarded collision writers.

Execution uses retained resources without allocation or host observation.
The direct public wrapper observes compact budget diagnostics before
returning; private prepared owners call check_admission explicitly.
Failure preserves caller particle/output/RNG arrays for this operation;
earlier successful operations are not rolled back.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2737-L2803)

## particula.gpu.kernels.coagulation._ensure_turbulent_input_array

*function*

```python
def _ensure_turbulent_input_array(name: str, value: Any, n_boxes: int, device: Any) -> Any
```

Validate and normalize a turbulent-shear numerical validation per-box
    input.

This private helper validates explicitly supplied state only; it neither
infers turbulent properties from shared containers nor transfers host data.
``coagulation_step_gpu`` uses it for every executable mask containing
ST1956 turbulent shear.

**Parameters**

- `name` (`str`) — Input name used in contract errors.
- `value` (`Any`) — Positive finite floating scalar or Warp array. It represents turbulent dissipation [m^2/s^3] or fluid density [kg/m^3], according to ``name``.
- `n_boxes` (`int`) — Number of boxes.
- `device` (`Any`) — Active particle device.

**Returns**

- (`Any`) — A same-device Warp array with shape ``(n_boxes,)``. Valid supplied
- (`Any`) — arrays are returned by identity; scalar broadcasts use private storage.

**Raises**

- `ValueError` — If the value is not a supported floating scalar or a
same-device ``wp.float64`` Warp array with shape ``(n_boxes,)``
and positive finite values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2240-L2272)

## particula.gpu.kernels.coagulation._ensure_volume_array

*function*

```python
def _ensure_volume_array(volume: float | Any, n_boxes: int, device: Any) -> Any
```

Ensure a GPU volume array is available.

**Parameters**

- `volume` (`float | Any`) — Volume scalar or Warp array.
- `n_boxes` (`int`) — Number of boxes.
- `device` (`Any`) — Warp device.

**Returns**

- (`Any`) — Warp array of volumes ``(n_boxes,)``.

**Raises**

- `ValueError` — If volume array has mismatched shape, device, or domain.

> **Notes**
>
> Valid Warp arrays are returned unchanged so launch code can reuse
> caller-owned per-box volume buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2322-L2367)

## particula.gpu.kernels.coagulation._initialize_rng_states

*function*

```python
def _initialize_rng_states(seed: Any, rng_states: Any) -> None
```

Initialize or reset per-box RNG states from a seed.

**Parameters**

- `seed` (`Any`) — RNG seed value.
- `rng_states` (`Any`) — RNG state buffer ``(n_boxes,)`` written in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L733-L744)

## particula.gpu.kernels.coagulation._native_coagulation_kernel

*function*

```python
def _native_coagulation_kernel(masses: Any, concentration: Any, density: Any, volume: Any, temperature: Any, pressure: Any, gas_constant: Any, boltzmann_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, time_step: Any, radii: Any, diffusivities: Any, g_terms: Any, speeds: Any, settling_velocities: Any, turbulent_dissipation: Any, fluid_density: Any, total_masses: Any, charge: Any, active_indices: Any, collision_pairs: Any, n_collisions: Any, rng_states: Any, mechanism_mask: Any, collision_capacity: Any) -> None
```

Schedule every native mechanism with its common raw particle count.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1421-L1488)

## particula.gpu.kernels.coagulation._native_coagulation_kernel_guarded

*function*

```python
def _native_coagulation_kernel_guarded(masses: Any, concentration: Any, density: Any, volume: Any, temperature: Any, pressure: Any, gas_constant: Any, boltzmann_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, time_step: Any, radii: Any, diffusivities: Any, g_terms: Any, speeds: Any, settling_velocities: Any, turbulent_dissipation: Any, fluid_density: Any, total_masses: Any, charge: Any, active_indices: Any, collision_pairs: Any, n_collisions: Any, rng_states: Any, mechanism_mask: Any, collision_capacity: Any, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1491-L1561)

## particula.gpu.kernels.coagulation._normalize_turbulent_input_array

*function*

```python
def _normalize_turbulent_input_array(value: Any, n_boxes: int, device: Any) -> Any
```

Normalize an already-validated turbulent numerical validation input.

This helper does not repeat validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2275-L2286)

## particula.gpu.kernels.coagulation._preflight_enabled_coagulation_terms

*function*

```python
def _preflight_enabled_coagulation_terms(resolved: _ResolvedCoagulationMechanismConfig, particles: Any, n_boxes: int, n_particles: int, device: Any, turbulent_dissipation: Any, fluid_density: Any, validate_charge_finite: bool) -> None
```

Run enabled-term validation before executable resource preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1975-L2019)

## particula.gpu.kernels.coagulation._prepare_coagulation_step_gpu

*function*

```python
def _prepare_coagulation_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: object, volume: float | Any | None = None, max_collisions: object = 256, rng_seed: int = 0, collision_pairs: Any | None = None, n_collisions: Any | None = None, rng_states: Any | None = None, *, mechanism_config: CoagulationMechanismConfig | None = None, work_config: CoagulationWorkConfig | None = None, work_counts: Any | None = None, initialize_rng: bool = False, environment: Any | None = None, validate_charge_finite: bool = False, turbulent_dissipation: float | Any | None = None, fluid_density: float | Any | None = None) -> _PreparedCoagulationCall
```

Prepare fixed resources for native equal-count resolved coagulation.

All nonempty subsets of Brownian, charged Brownian, SP2016 sedimentation
and ST1956 shear use tiled exact additive majorants. Particle counts have
no algorithmic cap; slot indices remain signed int32 and pair ranks int64.
Fixed buffers retain caller identity. This phase validates and allocates
workspace but never resets supplied RNG or writes collision outputs.
RNG initialization and all simulation writers follow per-step admission.

The concrete work configuration controls trial and collision byte budgets.
Direct calls report budget rejection after compact diagnostic readback;
resident calls latch a device failure before all dependent writers.
Particle concentration is represented count, volume is m^3, time is s,
dissipation is m^2/s^3 and fluid density is kg/m^3. Positive represented
counts must agree within each box. Turbulent inputs are required only for
masks containing shear. Configuration validation precedes particle access.

**Returns**

- (`_PreparedCoagulationCall`) — A frozen binding owning reusable property, reduction, admission and
- (`_PreparedCoagulationCall`) — RNG-initialization scratch alongside caller-owned simulation arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2423-L2734)

## particula.gpu.kernels.coagulation._remove_active_pair_by_rank_swap_pop

*function*

```python
def _remove_active_pair_by_rank_swap_pop(active_indices: Any, box_idx: Any, active_count: Any, rank_i: Any, adjusted_rank_j: Any) -> Any
```

Remove two active ranks from a compact active-index buffer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L675-L700)

## particula.gpu.kernels.coagulation._resolve_active_pair_by_rank

*function*

```python
def _resolve_active_pair_by_rank(active_indices: Any, box_idx: Any, n_particles: Any, rank_i: Any, adjusted_rank_j: Any) -> Any
```

Resolve two active-set ranks to particle indices in one scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L703-L730)

## particula.gpu.kernels.coagulation._resolve_collision_capacity

*function*

```python
def _resolve_collision_capacity(max_collisions: object, n_boxes: int, n_particles: int, byte_budget: int = MAX_COLLISION_PAIR_BUFFER_BYTES) -> int
```

Resolve a bounded accepted-collision capacity for one call.

**Parameters**

- `max_collisions` (`object`) — Requested maximum collisions per box.
- `n_boxes` (`int`) — Number of boxes in the current launch.
- `n_particles` (`int`) — Number of particles per box.

**Returns**

- (`int`) — Effective accepted-collision capacity per box.

> **Notes**
>
> Accepted collisions cannot exceed ``n_particles // 2`` because each
> accepted collision removes two active particles from the candidate set.
> Allocation is also bounded by an explicit buffer-byte budget so invalid
> requests cannot trigger unbounded internal allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2161-L2194)

## particula.gpu.kernels.coagulation._safe_acceptance_probability

*function*

```python
def _safe_acceptance_probability(total_rate: Any, total_majorant: Any) -> Any
```

Return a safe acceptance ratio or zero before an acceptance RNG draw.

A rate may exceed its majorant only by an eight-machine-epsilon relative
tolerance; that narrow case maps exactly to one. Material violations and
non-finite tolerance arithmetic fail closed before the caller draws an
acceptance variate. Candidate-selection draws precede this calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L160-L195)

## particula.gpu.kernels.coagulation._sanitize_positive_finite

*function*

```python
def _sanitize_positive_finite(value: Any) -> Any
```

Return a finite, strictly positive term or zero for safe accumulation.

Invalid, zero, and negative terms make no contribution to an additive
pair-rate or majorant total.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L133-L143)

## particula.gpu.kernels.coagulation._sedimentation_majorant_from_active_pairs

*function*

```python
def _sedimentation_majorant_from_active_pairs(active_indices: Any, box_idx: Any, active_count: Any, radii: Any, settling_velocities: Any) -> Any
```

Return the finite sedimentation-rate maximum over active pairs.

This helper exhaustively resolves compact active ranks, so sparse source
slots cannot affect the exact SP2016 sedimentation sampler majorant.
Invalid, zero, and nonpositive rates contribute zero.

**Parameters**

- `active_indices` (`Any`) — Compact particle indices ``(n_boxes, n_particles)``.
- `box_idx` (`Any`) — Index of the simulation box to scan.
- `active_count` (`Any`) — Number of compact active ranks in ``box_idx``.
- `radii` (`Any`) — Particle radii ``(n_boxes, n_particles)`` [m].
- `settling_velocities` (`Any`) — Private particle settling velocities ``(n_boxes, n_particles)`` [m/s].

**Returns**

- (`Any`) — Finite, nonnegative maximum sedimentation rate for the selected box,
- (`Any`) — or zero when no positive finite selected-pair rate exists.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L422-L467)

## particula.gpu.kernels.coagulation._select_active_pair_by_rank

*function*

```python
def _select_active_pair_by_rank(active_indices: Any, box_idx: Any, rank_i: Any, adjusted_rank_j: Any) -> Any
```

Resolve two active-set ranks from a compact active-index buffer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L660-L672)

## particula.gpu.kernels.coagulation._select_coagulation_pairs_impl

*function*

```python
def _select_coagulation_pairs_impl(masses: Any, concentration: Any, density: Any, volume: Any, temperature: Any, pressure: Any, gas_constant: Any, boltzmann_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, time_step: Any, radii: Any, diffusivities: Any, g_terms: Any, speeds: Any, settling_velocities: Any, turbulent_dissipation: Any, fluid_density: Any, total_masses: Any, charge: Any, active_indices: Any, collision_pairs: Any, n_collisions: Any, rng_states: Any, mechanism_mask: Any, collision_capacity: Any, box_idx: int, majorant_total: wp.float64, active_count: wp.int32, kinematic_viscosity: wp.float64, ion_data: IonPairData, capture: CaptureData) -> wp.vec2i
```

Preserve sequential pair selection and the exact persistent RNG path.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1211-L1418)

## particula.gpu.kernels.coagulation._total_majorant

*function*

```python
def _total_majorant(mechanism_mask: Any, radius_min: Any, radius_max: Any, diffusivity_min: Any, diffusivity_max: Any, g_term_min: Any, g_term_max: Any, speed_min: Any, speed_max: Any, active_indices: Any, box_idx: Any, active_count: Any, radii: Any, settling_velocities: Any, turbulent_dissipation: Any, kinematic_viscosity: Any, total_masses: Any, charges: Any, temperature: Any, pressure: Any, boltzmann_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, gas_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any) -> Any
```

Return a checked sum of enabled finite, positive majorant components.

Component maxima can occur on different active pairs, so each enabled
component is bounded independently and then added with checked fp64
arithmetic. An invalid or overflowing aggregate returns zero. This helper
supports deterministic internal probes and public approved-mask execution;
its dispatch does not make three-way masks executable. Brownian-containing
production selection instead uses one exact active-pair scan through
``_total_pair_rate``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L519-L657)

## particula.gpu.kernels.coagulation._total_pair_rate

*function*

```python
def _total_pair_rate(mechanism_mask: Any, radius_i: Any, radius_j: Any, diffusivity_i: Any, diffusivity_j: Any, g_term_i: Any, g_term_j: Any, speed_i: Any, speed_j: Any, settling_velocity_i: Any, settling_velocity_j: Any, turbulent_dissipation: Any, kinematic_viscosity: Any, total_mass_i: Any, total_mass_j: Any, charge_i: Any, charge_j: Any, temperature: Any, pressure: Any, boltzmann_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, gas_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any) -> Any
```

Return the enabled finite, positive pair rate.

This helper dispatches fixed flag bits without dynamic mechanism iteration.
Each enabled term is independently sanitized and added with checked fp64
arithmetic. An invalid component contributes zero, while an invalid or
overflowing aggregate returns zero. This helper's bit dispatch does not
itself grant sampler support; the entry-point capability gate admits only
approved singleton, two-way, and four-way masks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L198-L333)

## particula.gpu.kernels.coagulation._turbulent_majorant_from_active_radii

*function*

```python
def _turbulent_majorant_from_active_radii(active_indices: Any, box_idx: Any, active_count: Any, radii: Any, turbulent_dissipation: Any, kinematic_viscosity: Any) -> Any
```

Return an ST1956 majorant from the two largest compact active radii.

The ST1956 prefactor is constant within a box and its diameter-sum cubed
term is monotonic, making this O(A) compact-rank scan a safe majorant.

**Parameters**

- `active_indices` (`Any`) — Compact particle indices ``(n_boxes, n_particles)``.
- `box_idx` (`Any`) — Index of the simulation box to scan.
- `active_count` (`Any`) — Number of compact active ranks in ``box_idx``.
- `radii` (`Any`) — Particle radii ``(n_boxes, n_particles)`` [m].
- `turbulent_dissipation` (`Any`) — Per-box turbulent kinetic-energy dissipation rate [m^2/s^3].
- `kinematic_viscosity` (`Any`) — Per-box kinematic viscosity [m^2/s].

**Returns**

- (`Any`) — Finite, nonnegative ST1956 majorant for the selected box, or zero when
- (`Any`) — fewer than two active particles exist or no finite positive rate exists.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L470-L516)

## particula.gpu.kernels.coagulation._validate_charge_finite

*function*

```python
def _validate_charge_finite(charge: Any, n_boxes: int, n_particles: int, device: Any) -> None
```

Reject non-finite caller-owned charge through a read-only device scan.

This helper runs only after the public entry point validates the charge
shape, dtype, and device. It allocates a private status scalar, launches
one ``O(n_boxes * n_particles)`` read-only scan, and reads back only that
scalar. It neither copies nor mutates the caller-owned charge array.

**Parameters**

- `charge` (`Any`) — Caller-owned ``wp.float64`` charge array with shape ``(n_boxes, n_particles)``.
- `n_boxes` (`int`) — Number of simulation boxes.
- `n_particles` (`int`) — Number of particles per box.
- `device` (`Any`) — Warp device that owns ``charge`` and the private status buffer.

**Raises**

- `ValueError` — If any charge value is NaN or infinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1803-L1835)

## particula.gpu.kernels.coagulation._validate_charge_finite_kernel

*function*

```python
def _validate_charge_finite_kernel(charge: Any, invalid: Any) -> None
```

Record non-finite charge in a private status buffer without mutation.

**Parameters**

- `charge` (`Any`) — Caller-owned particle charge array ``(n_boxes, n_particles)`` read without modification.
- `invalid` (`Any`) — Private one-element ``wp.int32`` status buffer set to one when any charge value is non-finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L747-L761)

## particula.gpu.kernels.coagulation._validate_charged_density_kernel

*function*

```python
def _validate_charged_density_kernel(density: Any, invalid: Any) -> None
```

Record invalid charged species densities without mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L799-L807)

## particula.gpu.kernels.coagulation._validate_charged_particle_physics

*function*

```python
def _validate_charged_particle_physics(particles: Any, n_boxes: int, n_particles: int, device: Any) -> None
```

Reject invalid charged particle state before allocating writer resources.

Mass and concentration must be finite and nonnegative; density must be
finite and positive. Physical population size is independent of work tiles.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1838-L1881)

## particula.gpu.kernels.coagulation._validate_charged_particle_physics_kernel

*function*

```python
def _validate_charged_particle_physics_kernel(masses: Any, concentration: Any, density: Any, invalid: Any, active_counts: Any) -> None
```

Validate charged particle inputs and count selector-eligible slots.

Masses and concentrations may be zero for inactive slots, but all particle
state must be finite and nonnegative; species density must be finite and
strictly positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L764-L796)

## particula.gpu.kernels.coagulation._validate_collision_counts

*function*

```python
def _validate_collision_counts(n_collisions: Any, expected_shape: tuple[int], expected_device: Any) -> None
```

Validate collision count buffer shape and device.

**Parameters**

- `n_collisions` (`Any`) — Collision counts buffer.
- `expected_shape` (`tuple[int]`) — Expected ``(n_boxes,)`` shape.
- `expected_device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If the buffer shape or device mismatches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2049-L2071)

## particula.gpu.kernels.coagulation._validate_collision_pairs

*function*

```python
def _validate_collision_pairs(collision_pairs: Any, expected_shape: tuple[int, int, int], expected_device: Any) -> None
```

Validate collision pair buffer shape and device.

**Parameters**

- `collision_pairs` (`Any`) — Collision pairs buffer.
- `expected_shape` (`tuple[int, int, int]`) — Expected ``(n_boxes, max_collisions, 2)`` shape.
- `expected_device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If the buffer shape or device mismatches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2022-L2046)

## particula.gpu.kernels.coagulation._validate_device_arrays

*function*

```python
def _validate_device_arrays(particles: Any, device: Any) -> None
```

Validate particle arrays, including charge, share one Warp device.

**Parameters**

- `particles` (`Any`) — GPU particle data container.
- `device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If any particle array is on a different device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1784-L1800)

## particula.gpu.kernels.coagulation._validate_device_match

*function*

```python
def _validate_device_match(name: str, array: Any, expected_device: Any) -> None
```

Validate that a Warp array is on the expected device.

**Parameters**

- `name` (`str`) — Array label for error messages.
- `array` (`Any`) — Warp array to validate.
- `expected_device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If the array is not on the expected device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1768-L1781)

## particula.gpu.kernels.coagulation._validate_max_collisions

*function*

```python
def _validate_max_collisions(max_collisions: object) -> int
```

Validate the supported collision-buffer length before allocation.

**Parameters**

- `max_collisions` (`object`) — Proposed maximum accepted collisions per box.

**Returns**

- (`int`) — The validated collision limit as a Python ``int``.

**Raises**

- `ValueError` — If ``max_collisions`` is not a supported positive integer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2125-L2158)

## particula.gpu.kernels.coagulation._validate_particle_arrays

*function*

```python
def _validate_particle_arrays(particles: Any, n_boxes: int, n_particles: int, n_species: int) -> None
```

Validate particle array shapes, including the charge schema.

**Parameters**

- `particles` (`Any`) — GPU particle data container.
- `n_boxes` (`int`) — Expected number of boxes.
- `n_particles` (`int`) — Expected number of particles per box.
- `n_species` (`int`) — Expected number of species.

**Raises**

- `ValueError` — If particle arrays do not match expected shapes, or charge
is not ``wp.float64``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1732-L1765)

## particula.gpu.kernels.coagulation._validate_rng_states

*function*

```python
def _validate_rng_states(rng_states: Any, expected_shape: tuple[int], expected_device: Any) -> None
```

Validate RNG state buffer shape and device.

**Parameters**

- `rng_states` (`Any`) — RNG state buffer.
- `expected_shape` (`tuple[int]`) — Expected ``(n_boxes,)`` shape.
- `expected_device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If the buffer shape or device mismatches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2074-L2096)

## particula.gpu.kernels.coagulation._validate_sedimentation_particle_physics

*function*

```python
def _validate_sedimentation_particle_physics(particles: Any, n_boxes: int, n_particles: int, device: Any) -> None
```

Validate sedimentation state through a read-only device scan.

Reject nonfinite charge, invalid mass/density/counts, unequal positive
counts and invalid derived mass or volume. No active-population cap is
imposed. Only compact error status is transferred to the host.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1884-L1934)

## particula.gpu.kernels.coagulation._validate_sedimentation_particle_physics_kernel

*function*

```python
def _validate_sedimentation_particle_physics_kernel(masses: Any, concentration: Any, density: Any, charge: Any, invalid: Any) -> None
```

Validate all SP2016 inputs with one box-local read-only reduction.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L852-L904)

## particula.gpu.kernels.coagulation._validate_time_step

*function*

```python
def _validate_time_step(time_step: object) -> float
```

Validate the coagulation time step before any mutation.

**Parameters**

- `time_step` (`object`) — Proposed coagulation time step in seconds.

**Returns**

- (`float`) — The validated time step as a Python ``float``.

**Raises**

- `ValueError` — If ``time_step`` is not a finite nonnegative real value.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2099-L2122)

## particula.gpu.kernels.coagulation._validate_turbulent_input

*function*

```python
def _validate_turbulent_input(name: str, value: Any, n_boxes: int, device: Any) -> None
```

Validate a turbulent-shear numerical validation input without scalar
    normalization.

This validation-only boundary checks turbulent requests before allocating
private broadcast storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2289-L2319)

## particula.gpu.kernels.coagulation._validate_turbulent_particle_physics

*function*

```python
def _validate_turbulent_particle_physics(particles: Any, n_boxes: int, device: Any) -> None
```

Reject invalid ST1956 particle physics before mutable runtime work.

Turbulent-shear selection uses the same physical particle fields as the
other direct mechanisms, but does not impose their concentration uniformity
or active-count constraints. It validates finite charge because the shared
merge application consumes it, and rejects overflowed active mass/volume
totals before rate sanitization could turn them into no-collision behavior.
This read-only scan runs before time, environment, output, or RNG
processing.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1937-L1972)

## particula.gpu.kernels.coagulation._validate_turbulent_particle_physics_kernel

*function*

```python
def _validate_turbulent_particle_physics_kernel(masses: Any, concentration: Any, density: Any, charge: Any, invalid: Any) -> None
```

Record invalid ST1956 particle state without mutating caller inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L810-L849)

## particula.gpu.kernels.coagulation.adaptive_coagulation_step_gpu

*function*

```python
def adaptive_coagulation_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float, *, subcycling: CoagulationSubcyclingConfig, work_config: CoagulationWorkConfig | None = None, **kwargs: Any) -> AdaptiveCoagulationResult
```

Run a synchronous, atomic direct interval on the particle device.

Accepts the fixed call's optional inputs. Supplied collision and work
buffers hold the final accepted pass. Returned device pairs span the whole
interval in application order. Each diagnostics pair_end_offsets row gives
the exclusive end of that pass per box. Reuse rng_states for continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2968-L3076)

## particula.gpu.kernels.coagulation.apply_coagulation_kernel

*function*

```python
def apply_coagulation_kernel(masses: Any, concentration: Any, charge: Any, collision_pairs: Any, n_collisions: Any) -> None
```

Run the shared device operation through the direct kernel API.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1637-L1656)

## particula.gpu.kernels.coagulation.apply_coagulation_kernel_body

*function*

```python
def apply_coagulation_kernel_body(masses: Any, concentration: Any, charge: Any, collision_pairs: Any, n_collisions: Any, box_idx: wp.int32, collision_idx: wp.int32) -> None
```

Apply representable coagulation merges to caller-owned state in place.

A pair is applied only when the summed charge and every summed species mass
are finite and nonnegative. Applied merges add donor mass and charge to the
recipient, then clear the donor mass, concentration, and charge. A rejected
unrepresentable pair leaves both particle inventories unchanged and is
removed from the collision output during the following compaction pass.

**Parameters**

- `masses` (`Any`) — Particle masses array ``(n_boxes, n_particles, n_species)``.
- `concentration` (`Any`) — Particle concentrations ``(n_boxes, n_particles)``.
- `charge` (`Any`) — Particle charges with ``wp.float64`` dtype and shape ``(n_boxes, n_particles)``. Donor charge is added to its recipient and then cleared.
- `collision_pairs` (`Any`) — Collision index buffer ``(n_boxes, max_collisions, 2)``.
- `n_collisions` (`Any`) — Collision counts ``(n_boxes,)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1564-L1634)

## particula.gpu.kernels.coagulation.apply_coagulation_kernel_guarded

*function*

```python
def apply_coagulation_kernel_guarded(masses: Any, concentration: Any, charge: Any, collision_pairs: Any, n_collisions: Any, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1659-L1681)

## particula.gpu.kernels.coagulation.coagulation_preflight_gpu

*function*

```python
def coagulation_preflight_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: object, **kwargs: Any) -> dict[str, Any]
```

Inspect native capability, resources and trial demand without advancing.

Accepts the direct step's keyword arguments. The report downloads compact
per-box data, not particle arrays. Preparation allocates temporary workspace
but leaves caller particle, collision-output and RNG buffers unchanged.
Required bytes count unique bound collision, RNG, normalized environment,
property, reduction and admission arrays (including caller-supplied
sidecars). They exclude particle storage and runtime/compiler allocations;
this is bound storage, not an estimate of incremental allocation.

**Returns**

- (`dict[str, Any]`) — Report containing active counts, unordered pair evaluations, expected
- (`dict[str, Any]`) — trials, conservative required trials, available trial capacity,
- (`dict[str, Any]`) — budget-admissible timestep [s], per-box admission and workspace bytes.
- (`dict[str, Any]`) — The admissible timestep is a resource bound, not an accuracy estimate.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2867-L2955)

## particula.gpu.kernels.coagulation.coagulation_step_gpu

*function*

```python
def coagulation_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: object, volume: float | Any | None = None, max_collisions: object = 256, rng_seed: int = 0, collision_pairs: Any | None = None, n_collisions: Any | None = None, rng_states: Any | None = None, *, mechanism_config: CoagulationMechanismConfig | None = None, work_config: CoagulationWorkConfig | None = None, work_counts: Any | None = None, initialize_rng: bool = False, environment: Any | None = None, validate_charge_finite: bool = False, turbulent_dissipation: float | Any | None = None, fluid_density: float | Any | None = None) -> tuple[Any, Any, Any]
```

Prepare and enqueue one direct particle-resolved coagulation step.

Preparation validates inputs and allocates workspace. Enqueue uses the
frozen call record; optional RNG initialization follows successful device
admission. This direct wrapper observes compact admission diagnostics.

The optional ``work_counts`` buffer has shape (n_boxes, 2), dtype int32,
and shares the particle device. Its columns contain scheduled and attempted
candidate trials for this step. It must not overlap any other input/output.
Admission rejection preserves it; successful calls overwrite every box.

**Returns**

- (`Any`) — The particle container, collision-pair output, and per-box collision
- (`Any`) — counts returned by the direct kernel.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2806-L2864)

## particula.gpu.kernels.coagulation.initialize_coagulation_rng_states

*function*

```python
def initialize_coagulation_rng_states(rng_seed: int, rng_states: Any, *, device: Any | None = None) -> Any
```

Initialize a caller-owned coagulation RNG state buffer from a seed.

**Parameters**

- `rng_seed` (`int`) — Seed used to initialize the per-box Warp RNG state.
- `rng_states` (`Any`) — Caller-owned ``(n_boxes,)`` buffer with dtype ``wp.uint32``.
- `device` (`Any | None`) (default: `None`) — Optional Warp device override. When omitted, uses ``rng_states.device``.

**Returns**

- (`Any`) — The same ``rng_states`` buffer after in-place initialization.

**Raises**

- `ValueError` — If ``rng_states`` is not a one-dimensional ``wp.uint32``
buffer or if the provided device mismatches the buffer device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2197-L2237)
