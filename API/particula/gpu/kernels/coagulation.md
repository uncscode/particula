# particula.gpu.kernels.coagulation

*module*

GPU direct-kernel utilities for additive particle-resolved coagulation.

This module composes Warp ``@wp.func`` building blocks into a direct-kernel
coagulation pipeline. The executable particle-resolved masks are the four
singletons ``1``, ``2``, ``4``, and ``8``; the six unordered two-way masks
``3``, ``5``, ``6``, ``9``, ``10``, and ``12``; and the four-way mask ``15``.
Mask ``7`` rejects at capability preflight before particle metadata or
enabled-term validation. Turbulent masks ``11``, ``13``, and ``14`` validate
particle metadata and enabled terms, then raise
``ValueError("Additive coagulation execution is deferred.")`` before downstream
work.
Mechanism tuples normalize to Brownian, charged hard-sphere, SP2016
sedimentation, then ST1956 turbulent shear.

Enabled fp64 component rates are sanitized and summed. A safe,
potentially-conservative sum of component majorants satisfies
``sum_m K_m(i, j) <= sum_m M_m``; component maxima may occur on different
pairs. Production selection obtains its majorant from an exact compact-active
scan of the summed rate. One shared candidate stream, acceptance stream,
collision output, RNG stream, and apply pass serve every approved mask.
Entry-point validation accepts scalar direct inputs, explicit
``(n_boxes,)`` Warp arrays, or a ``WarpEnvironmentData`` container. Particle
metadata and device checks run before normalizing environment inputs, setting up
volume, initializing RNG state, or executing Brownian work. All native calls
require explicit resolved metadata, equal positive active counts per box, and
finite charge. Private prepared calls freeze the selected mechanism and pinned
arrays after native preflight. Their enqueue performs retained device work only,
with no validation,
allocation, readback, transfer, synchronization, lookup, rebinding, or RNG
initialization/reset. Rejection is pre-writer; rollback is not promised after a
writer launches. The kernels
operate on GPU-resident particle data and produce collision pairs that are
applied in place: recipient particles receive donor mass and charge, while
donor mass, concentration, and charge are cleared.

``CoagulationMechanismConfig`` and its resolver are concrete-module APIs:
import them from ``particula.gpu.kernels.coagulation``, not
``particula.gpu.kernels``. The immutable, keyword-only configuration is
host-side metadata, not device-resident simulation state. The public step
defaults to Brownian, particle-resolved execution and accepts every singleton,
two-way combination, and the four-way combination. Three-way masks remain
deferred. Malformed, duplicate, unknown, and non-particle-resolved
configurations reject before runtime state is accessed or mutated.
Turbulent-shear-enabled masks validate and normalize explicit
turbulent dissipation and fluid-density inputs after particle schema/device
metadata is available. Supplied particles, collision outputs, and RNG sidecars
are caller-owned same-device Warp resources.

The ``SEDIMENTATION_SP2016_MECHANISM_FLAG`` is a public direct-kernel mode for
particle-resolved, unit-efficiency SP2016 scheduling. It shares the bounded
sampler and persistent RNG path. Sedimentation-enabled calls read-only validate
finite, nonnegative masses and concentrations and finite, positive densities
before any output, RNG, or particle-state mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1-L1)

## particula.gpu.kernels.coagulation.MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX

*attribute* · *module attribute*

```python
MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX = 256
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L142-L142)

## particula.gpu.kernels.coagulation.MAX_COLLISION_PAIR_BUFFER_BYTES

*attribute* · *module attribute*

```python
MAX_COLLISION_PAIR_BUFFER_BYTES = 256 * 1024 * 1024
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L137-L137)

## particula.gpu.kernels.coagulation.MAX_SCHEDULED_TRIALS_PER_BOX

*attribute* · *module attribute*

```python
MAX_SCHEDULED_TRIALS_PER_BOX = 65536
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L138-L138)

## particula.gpu.kernels.coagulation.MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX

*attribute* · *module attribute*

```python
MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX = 256
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L145-L145)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2182-L2214)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.charge

*attribute* · *instance attribute*

```python
charge: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2208-L2208)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.collision_pairs

*attribute* · *instance attribute*

```python
collision_pairs: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2209-L2209)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.concentration

*attribute* · *instance attribute*

```python
concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2207-L2207)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2213-L2213)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.masses

*attribute* · *instance attribute*

```python
masses: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2206-L2206)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.max_collisions

*attribute* · *instance attribute*

```python
max_collisions: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2212-L2212)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.n_boxes

*attribute* · *instance attribute*

```python
n_boxes: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2211-L2211)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.n_collisions

*attribute* · *instance attribute*

```python
n_collisions: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2210-L2210)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2205-L2205)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.selector_inputs

*attribute* · *instance attribute*

```python
selector_inputs: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2214-L2214)

### particula.gpu.kernels.coagulation._PreparedCoagulationCall.__init__

*method*

```python
def __init__(particles: Any, masses: Any, concentration: Any, charge: Any, collision_pairs: Any, n_collisions: Any, n_boxes: int, max_collisions: int, device: Any, selector_inputs: tuple[Any, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L0-L0)

## particula.gpu.kernels.coagulation._bound_scheduled_trials

*function*

```python
def _bound_scheduled_trials(expected_trials: Any) -> Any
```

Clamp scheduled-trial counts before any int32 conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L148-L158)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L363-L446)

## particula.gpu.kernels.coagulation._checked_positive_finite_add

*function*

```python
def _checked_positive_finite_add(left: Any, right: Any) -> Any
```

Add two sanitized fp64 terms, failing closed on invalid arithmetic.

Additive mechanism terms are independently optional, but an overflowing
aggregate cannot safely be used as a selector rate or majorant.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L174-L185)

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
mutation. Supported singleton, two-way, and four-way masks share this single
candidate/acceptance/apply path; deferred masks return before scheduling or
accessing the RNG state.

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L937-L1348)

## particula.gpu.kernels.coagulation._compact_applied_collision_pairs_kernel

*function*

```python
def _compact_applied_collision_pairs_kernel(collision_pairs: Any, n_collisions: Any) -> None
```

Remove rejected collision entries after the apply pass.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1493-L1513)

## particula.gpu.kernels.coagulation._enqueue_prepared_coagulation_call

*function*

```python
def _enqueue_prepared_coagulation_call(prepared: _PreparedCoagulationCall) -> tuple[Any, Any, Any]
```

Issue only the frozen selector, apply, and compaction kernels.

This private enqueue boundary performs no validation, allocation,
normalization, RNG initialization, or resource lookup.

**Parameters**

- `prepared` (`_PreparedCoagulationCall`) — Validated call record produced by ``_prepare_coagulation_step_gpu``.

**Returns**

- (`Any`) — The pinned particle container, collision-pair buffer, and
- (`Any`) — collision-count buffer, in the same order as the public direct-kernel
- (`Any`) — API.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2736-L2779)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2051-L2083)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2134-L2179)

## particula.gpu.kernels.coagulation._initialize_rng_states

*function*

```python
def _initialize_rng_states(seed: Any, rng_states: Any) -> None
```

Initialize or reset per-box RNG states from a seed.

**Parameters**

- `seed` (`Any`) — RNG seed value.
- `rng_states` (`Any`) — RNG state buffer ``(n_boxes,)`` written in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L760-L771)

## particula.gpu.kernels.coagulation._native_coagulation_kernel

*function*

```python
def _native_coagulation_kernel(masses: Any, concentration: Any, density: Any, volume: Any, temperature: Any, pressure: Any, gas_constant: Any, boltzmann_constant: Any, molecular_weight_air: Any, ref_viscosity: Any, ref_temperature: Any, sutherland_constant: Any, elementary_charge_value: Any, electric_permittivity: Any, time_step: Any, radii: Any, diffusivities: Any, g_terms: Any, speeds: Any, settling_velocities: Any, turbulent_dissipation: Any, fluid_density: Any, total_masses: Any, charge: Any, active_indices: Any, collision_pairs: Any, n_collisions: Any, rng_states: Any, mechanism_mask: Any, collision_capacity: Any) -> None
```

Schedule every native mechanism with its common raw particle count.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1351-L1418)

## particula.gpu.kernels.coagulation._normalize_turbulent_input_array

*function*

```python
def _normalize_turbulent_input_array(value: Any, n_boxes: int, device: Any) -> Any
```

Normalize an already-validated turbulent numerical validation input.

This helper does not repeat validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2086-L2097)

## particula.gpu.kernels.coagulation._preflight_enabled_coagulation_terms

*function*

```python
def _preflight_enabled_coagulation_terms(resolved: _ResolvedCoagulationMechanismConfig, particles: Any, n_boxes: int, n_particles: int, device: Any, turbulent_dissipation: Any, fluid_density: Any, validate_charge_finite: bool) -> None
```

Run enabled-term validation before executable resource preparation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1793-L1834)

## particula.gpu.kernels.coagulation._prepare_coagulation_step_gpu

*function*

```python
def _prepare_coagulation_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: object, volume: float | Any | None = None, max_collisions: object = 256, rng_seed: int = 0, collision_pairs: Any | None = None, n_collisions: Any | None = None, rng_states: Any | None = None, *, mechanism_config: CoagulationMechanismConfig | None = None, initialize_rng: bool = False, environment: Any | None = None, validate_charge_finite: bool = False, turbulent_dissipation: float | Any | None = None, fluid_density: float | Any | None = None) -> _PreparedCoagulationCall
```

Privately prepare one direct, particle-resolved Warp coagulation step.

This private helper performs the public wrapper's validation, normalization,
allocation, and optional RNG initialization, then freezes launch references.
It is not a CPU strategy-composition or ``Runnable`` API.
It supports only ``"particle_resolved"`` execution: omission selects
Brownian; the executable singleton masks are ``1``, ``2``, ``4``, and ``8``;
unordered two-way masks are ``3``, ``5``, ``6``, ``9``, ``10``, and ``12``;
and the four-way mask is ``15``. Supported mechanism tuples normalize to
Brownian, charged hard-sphere, SP2016 sedimentation, and ST1956 turbulent
shear. Mask ``7`` raises the stable deferred error at capability preflight,
before particle metadata or enabled-term validation. Turbulent masks ``11``,
``13``, and ``14`` validate particle metadata and enabled terms before
raising that error, without downstream work.
``mechanism_config`` is immutable, keyword-only host metadata and is
structurally preflighted before any runtime input access, allocation,
normalization, RNG work, or launch. Enabled fp64 pair-rate components are
sanitized and summed. A component-majorant sum is safe but can be
conservative because component maxima can occur on different pairs;
production selection obtains the majorant by scanning the summed active-pair
rates. Every approved mask uses one candidate stream, acceptance stream,
collision-buffer set, RNG stream, and apply pass. Its active-pair work is
O(A²), while
collision and selector buffers are O(A), for active count A; this is bounded
implementation scope, not a throughput or scaling claim. Malformed,
duplicate, unknown, and unsupported-distribution combinations reject before
particle state is accessed. A structurally valid turbulent-shear request
reaches a later numerical validation boundary after particle metadata and
device checks.
Turbulent-shear-enabled executable masks normalize their explicit per-box
inputs and calculate ST1956 rates. After particle metadata and device
checks, direct temperature and pressure inputs are validated and normalized
before volume setup or selector launches.
``particles`` and supplied ``collision_pairs``, ``n_collisions``,
and ``rng_states`` are caller-owned same-device Warp resources; the step
mutates them in place as applicable. The three-item return tuple contains
no RNG state: supplied collision buffers are returned by identity, while
``rng_states`` remains caller-owned. Caller-owned RNG buffers are reset only
when ``initialize_rng=True`` explicitly opts in.

Explicitly tagged native inputs support resolved counts only. Every native
mechanism requires equal positive active counts within each box, native
fp64 primary arrays, and finite physical state. The selector multiplies its
schedule by the current common count for every supported mechanism. An
explicit volume must equal the particle container's positive physical
volume. All native writers must be disjoint from each other and protected
inputs. These read-only checks may allocate private flags and synchronize
during preparation; frozen enqueue adds no validation or host work.
Untagged populations reject before execution.

**Parameters**

- `particles` (`Any`) — Caller-owned, Warp-resident particle data. All required Warp arrays, including ``charge``, must be on the same device. ``charge`` must be ``wp.float64`` with shape ``(n_boxes, n_particles)``. Charged-containing calls scan it for finite values before output validation or allocation, RNG setup, or selector/apply work. Exact SP2016 sedimentation calls scan mass and concentration for finite nonnegative values and density for finite positive values before time-step, environment, output-buffer, or RNG work. Accepted collisions mutate mass, concentration, and charge in place: recipient particles receive donor mass and charge, and donor mass, concentration, and charge are cleared.
- `temperature` (`float | Any | None`) — Direct gas temperature as either a scalar or a Warp array with shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
- `pressure` (`float | Any | None`) — Direct gas pressure as either a scalar or a Warp array with shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
- `time_step` (`object`) — Coagulation time step in seconds.
- `volume` (`float | Any | None`) (default: `None`) — Per-box volume [m^3]. If None, uses ``particles.volume``.
- `max_collisions` (`object`) (default: `256`) — Maximum number of collisions per box.
- `rng_seed` (`int`) (default: `0`) — Seed used whenever this call initializes Warp RNG states. This always applies to the omitted-``rng_states`` convenience path. For caller-owned persistent buffers, the seed is only consumed when ``initialize_rng=True`` explicitly requests a reset.
- `collision_pairs` (`Any | None`) (default: `None`) — Optional caller-owned, same-device Warp collision output buffer. The buffer is written in place. When omitted, the function allocates a call-local convenience buffer for this call.
- `n_collisions` (`Any | None`) (default: `None`) — Optional caller-owned, same-device Warp per-box collision count output buffer. The buffer is written in place. When omitted, the function allocates a call-local convenience buffer for this call.
- `rng_states` (`Any | None`) (default: `None`) — Optional caller-owned, same-device Warp RNG state buffer. The buffer is mutated in place. When omitted, this function allocates a call-local same-device ``(n_boxes,)`` buffer, seeds it from ``rng_seed``, and uses it only for the current call. When provided, the caller owns the persistent GPU-resident sidecar buffer and it is reused as-is across repeated calls unless ``initialize_rng=True`` explicitly requests a reset from  ``rng_seed``.
- `mechanism_config` (`CoagulationMechanismConfig | None`) (default: `None`) — Optional immutable host-side ``CoagulationMechanismConfig`` imported from ``particula.gpu.kernels.coagulation``; it is not re-exported by ``particula.gpu.kernels``. This keyword-only configuration does not transfer, synchronize, or own Warp state. Omission selects Brownian, particle-resolved execution. The executable masks are ``1``, ``2``, ``4``, ``8``, ``3``, ``5``, ``6``, ``9``, ``10``, ``12``, and ``15``. Malformed configurations, unsupported distributions, and unrecognized mechanism combinations fail before particle state is accessed or mutated. Deferred mask ``7`` rejects at capability preflight before particle metadata or enabled-term validation. Turbulent masks ``11``, ``13``, and ``14`` validate particle metadata and enabled terms before rejecting. A wrong type raises exactly ``ValueError`` with the message ``"mechanism_config must be a CoagulationMechanismConfig."``; other errors are delegated to the resolver and capability gate.
- `initialize_rng` (`bool`) (default: `False`) — Explicit reset flag for caller-provided ``rng_states``. The default ``False`` path validates the buffer and reuses its existing state without reseeding, even if ``rng_seed`` matches an earlier call. Set ``True`` to launch ``_initialize_rng_states`` after validation and reset the persistent buffer from ``rng_seed``. This argument does not affect omitted ``rng_states`` convenience allocation, which always initializes the internal buffer for the current call.
- `environment` (`Any | None`) (default: `None`) — Optional ``WarpEnvironmentData`` with ``(n_boxes,)`` temperature and pressure arrays on the same device as ``particles``. This mode is supported when both direct inputs are ``None``.
- `validate_charge_finite` (`bool`) (default: `False`) — When True, explicitly scan charge on device and synchronize for host-visible rejection of NaN or infinity. The native population validation requires finite charge in every mode. Native tagged inputs always validate charge. Every charged mode performs this scan before caller output or RNG mutation.
- `turbulent_dissipation` (`float | Any | None`) (default: `None`) — Explicit turbulent kinetic-energy dissipation rate [m^2/s^3] for the turbulent-shear numerical validation boundary. For a structurally valid ST1956 turbulent request, accepts a positive finite Python or NumPy floating scalar, or a positive finite, active-device ``wp.float64`` Warp array with shape ``(n_boxes,)``. Supplied arrays retain identity; scalar broadcasts use private helper-owned storage. This input is not inferred from shared containers and is ignored by non-turbulent masks.
- `fluid_density` (`float | Any | None`) (default: `None`) — Explicit carrier-fluid density [kg/m^3] for the turbulent-shear numerical validation boundary. It has the same conditional scalar and active-device ``wp.float64`` ``(n_boxes,)`` Warp-array contract as ``turbulent_dissipation``, is not inferred from shared containers, and is ignored by non-turbulent masks.

**Returns**

- (`_PreparedCoagulationCall`) — A private frozen call record for the matching enqueue helper.

**Raises**

- `ValueError` — If array shapes or devices mismatch expectations.
- `ValueError` — If direct ``temperature`` or ``pressure`` inputs are mixed
with ``environment``.
- `ValueError` — If direct inputs are missing when ``environment`` is
omitted.
- `ValueError` — If environment arrays do not match ``(n_boxes,)`` or the
caller device.
- `ValueError` — If mechanism_config has the wrong type, is malformed, or
requests an unsupported distribution or unrecognized mechanism
combination.
- `ValueError` — If an additive mechanism configuration is unsupported.
- `ValueError` — If a structurally valid turbulent-shear request omits
either explicit turbulent input or supplies an input with an
unsupported type, shape, device, ``wp.float64`` dtype, or value
domain. Deferred turbulent three-way combinations reject after this
Numerical validation boundary.
- `ValueError` — If particle charge does not have shape
``(n_boxes, n_particles)``, dtype ``wp.float64``, or the particle
device. Charged-containing and exact SP2016 sedimentation
execution, and Brownian with ``validate_charge_finite=True``, also
raise for NaN or infinity.
- `ValueError` — If charged-containing masses or concentrations are
non-finite or negative, species density is non-finite or
nonpositive, or a box exceeds
``MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX`` active particles.
- `ValueError` — If SP2016 sedimentation masses or concentrations are
non-finite or negative, positive concentrations differ within a
box, species density is non-finite or nonpositive, or a box exceeds
``MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX`` active particles.

> **Notes**
>
> Accepted environment sources are scalar direct inputs, direct
> ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
> inputs, or keyword-only ``environment=...`` execution.
> 
> ``mechanism_config``, ``initialize_rng``, ``environment``,
> ``validate_charge_finite``, ``turbulent_dissipation``, and
> ``fluid_density`` remain keyword-only so existing positional scalar
> callers stay source-compatible. Structural configuration preflight
> occurs before all runtime input access, allocation, normalization, RNG
> work, and launches. Deferred turbulent three-way masks validate their
> explicit numerical validation inputs after particle schema/device
> metadata, but cannot
> reach downstream simulation, selector, apply launches, allocation,
> normalization, RNG work, or caller-owned mutation. It does not make
> unrelated later validation failures atomic.
> 
> Structurally valid turbulent-shear requests require explicit
> ``turbulent_dissipation`` and ``fluid_density`` only after structural
> configuration resolution and particle schema/device metadata validation.
> Deferred turbulent three-way combinations then reject before
> normalization, allocation, rate dispatch, sampling, majorant
> construction, RNG work, or mutation. Executable turbulent masks
> normalize both inputs and use their active-device values for ST1956
> execution. Non-turbulent masks do not inspect or validate either input.
> 
> validation runs before volume normalization, RNG setup, and Brownian or
> apply launches so invalid ``time_step``, shape, dtype, or device
> combinations fail without mutating particle state or allocating
> downstream work. The normalized environment arrays are forwarded
> directly into the launch path.
> 
> Charged-containing execution read-only-validates finite/nonnegative
> masses and concentrations, finite/positive density, its bounded active
> count, and finite charge before time-step or environment normalization.
> The active bound caps its exact compact-active O(A²) majorant at 32,640
> pairs per box. Native population admission requires finite charge for
> every supported mechanism, including Brownian execution.
> 
> Exact SP2016 sedimentation read-only-validates finite charge,
> finite/nonnegative masses and concentrations, finite/positive density,
> compatible positive concentrations, and its bounded active count before
> time-step, environment, output, or RNG work. Its exact compact-active
> O(A²) majorant is capped at 32,640 pairs per box. Zero masses and
> concentrations remain valid inactive-slot state; this mode does not
> impose charged-only constraints.
> 
> A successful call allocates private, call-local selector scratch,
> including a ``wp.float64`` ``(n_boxes, n_particles)`` total-mass array.
> Sedimentation-enabled masks allocate a separate zero-initialized
> settling-velocity array; non-sedimentation paths alias that storage to
> radii. The selector clears its own scratch slots and sums each active
> particle's species masses once for charged pair rates and their
> compact-active majorant. It derives radii for ST1956 turbulent-shear
> rates and populates settling velocities for valid active slots whenever
> sedimentation is enabled, including mixed masks.
> This scratch is not a caller-owned output, is not returned, and is
> allocated only after caller-resource preflight succeeds.
> 
> Supported RNG setup cases are:
> 
> - Omitted ``rng_states``: allocate a call-local internal buffer and
>   initialize it from ``rng_seed`` for this call.
> - Provided ``rng_states`` with ``initialize_rng=False``: validate the
>   caller-owned persistent buffer and reuse it without resetting.
> - Provided ``rng_states`` with ``initialize_rng=True``: validate the
>   caller-owned persistent buffer, then reset it from ``rng_seed``.
> - Validation failure: raise before RNG initialization or other Warp
>   launches mutate state.
> 
> Reusing the same ``rng_seed`` alongside a persistent ``rng_states``
> buffer does not reseed that buffer on later calls unless
> ``initialize_rng=True`` is passed. Initialize caller-owned buffers once
> before repeated timesteps, then reuse them without hidden per-step
> resets. Graph-capture support remains deferred.
> 
> After a successful preflight, this is an in-place asynchronous Warp API.
> A device-launch or runtime failure has no rollback guarantee for
> particles, collision outputs, or RNG state. Callers must synchronize and
> check Warp errors, then restore or discard their own snapshots before
> retrying.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2217-L2733)

## particula.gpu.kernels.coagulation._remove_active_pair_by_rank_swap_pop

*function*

```python
def _remove_active_pair_by_rank_swap_pop(active_indices: Any, box_idx: Any, active_count: Any, rank_i: Any, adjusted_rank_j: Any) -> Any
```

Remove two active ranks from a compact active-index buffer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L702-L727)

## particula.gpu.kernels.coagulation._resolve_active_pair_by_rank

*function*

```python
def _resolve_active_pair_by_rank(active_indices: Any, box_idx: Any, n_particles: Any, rank_i: Any, adjusted_rank_j: Any) -> Any
```

Resolve two active-set ranks to particle indices in one scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L730-L757)

## particula.gpu.kernels.coagulation._resolve_collision_capacity

*function*

```python
def _resolve_collision_capacity(max_collisions: object, n_boxes: int, n_particles: int) -> int
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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1976-L2005)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L188-L223)

## particula.gpu.kernels.coagulation._sanitize_positive_finite

*function*

```python
def _sanitize_positive_finite(value: Any) -> Any
```

Return a finite, strictly positive term or zero for safe accumulation.

Invalid, zero, and negative terms make no contribution to an additive
pair-rate or majorant total.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L161-L171)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L449-L494)

## particula.gpu.kernels.coagulation._select_active_pair_by_rank

*function*

```python
def _select_active_pair_by_rank(active_indices: Any, box_idx: Any, rank_i: Any, adjusted_rank_j: Any) -> Any
```

Resolve two active-set ranks from a compact active-index buffer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L687-L699)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L546-L684)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L226-L360)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L497-L543)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1587-L1619)

## particula.gpu.kernels.coagulation._validate_charge_finite_kernel

*function*

```python
def _validate_charge_finite_kernel(charge: Any, invalid: Any) -> None
```

Record non-finite charge in a private status buffer without mutation.

**Parameters**

- `charge` (`Any`) — Caller-owned particle charge array ``(n_boxes, n_particles)`` read without modification.
- `invalid` (`Any`) — Private one-element ``wp.int32`` status buffer set to one when any charge value is non-finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L774-L788)

## particula.gpu.kernels.coagulation._validate_charged_density_kernel

*function*

```python
def _validate_charged_density_kernel(density: Any, invalid: Any) -> None
```

Record invalid charged species densities without mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L826-L834)

## particula.gpu.kernels.coagulation._validate_charged_particle_physics

*function*

```python
def _validate_charged_particle_physics(particles: Any, n_boxes: int, n_particles: int, device: Any) -> None
```

Reject invalid charged particle inputs and excessive active work.

The exact charged majorant enumerates all unique active pairs in a single
selector thread. This read-only preflight therefore bounds each box to
``MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX`` active particles before any output
allocation, RNG initialization, selector launch, or caller-state mutation.

**Raises**

- `ValueError` — If mass or concentration is non-finite/negative, density is
non-finite/nonpositive, or a box exceeds the charged active limit.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1622-L1676)

## particula.gpu.kernels.coagulation._validate_charged_particle_physics_kernel

*function*

```python
def _validate_charged_particle_physics_kernel(masses: Any, concentration: Any, density: Any, invalid: Any, active_counts: Any) -> None
```

Validate charged particle inputs and count selector-eligible slots.

Masses and concentrations may be zero for inactive slots, but all particle
state must be finite and nonnegative; species density must be finite and
strictly positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L791-L823)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1864-L1886)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1837-L1861)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1568-L1584)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1552-L1565)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1940-L1973)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1516-L1549)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1889-L1911)

## particula.gpu.kernels.coagulation._validate_sedimentation_particle_physics

*function*

```python
def _validate_sedimentation_particle_physics(particles: Any, n_boxes: int, n_particles: int, device: Any) -> None
```

Reject invalid or excessive SP2016 inputs through read-only scans.

The fixed-size error state permits one device-to-host readback while
retaining field-specific errors. A single box-local reduction validates
charge, particle state, derived totals, and bounded exact-pair work.
Positive concentrations must agree within a box because this
particle-resolved merge representation stores one concentration per slot.
Zero mass and concentration are valid, allowing inactive slots without
applying charged-only constraints. This preflight runs before time-step,
environment, output-buffer, or RNG work.

**Parameters**

- `particles` (`Any`) — Caller-owned Warp particle data with validated shape and device placement.
- `n_boxes` (`int`) — Number of simulation boxes.
- `n_particles` (`int`) — Number of particle slots per box.
- `device` (`Any`) — Warp device that owns the particle arrays and status buffer.

**Raises**

- `ValueError` — If mass or concentration is non-finite or negative, density
is non-finite or nonpositive, concentrations are incompatible, or a
box exceeds the sedimentation active limit.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1679-L1752)

## particula.gpu.kernels.coagulation._validate_sedimentation_particle_physics_kernel

*function*

```python
def _validate_sedimentation_particle_physics_kernel(masses: Any, concentration: Any, density: Any, charge: Any, active_limit: Any, invalid: Any) -> None
```

Validate all SP2016 inputs with one box-local read-only reduction.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L879-L934)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1914-L1937)

## particula.gpu.kernels.coagulation._validate_turbulent_input

*function*

```python
def _validate_turbulent_input(name: str, value: Any, n_boxes: int, device: Any) -> None
```

Validate a turbulent-shear numerical validation input without scalar
    normalization.

This validation-only boundary validates turbulent requests before deferred
three-way-mask capability rejection. Valid scalars do not create
private broadcast storage here.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2100-L2131)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1755-L1790)

## particula.gpu.kernels.coagulation._validate_turbulent_particle_physics_kernel

*function*

```python
def _validate_turbulent_particle_physics_kernel(masses: Any, concentration: Any, density: Any, charge: Any, invalid: Any) -> None
```

Record invalid ST1956 particle state without mutating caller inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L837-L876)

## particula.gpu.kernels.coagulation.apply_coagulation_kernel

*function*

```python
def apply_coagulation_kernel(masses: Any, concentration: Any, charge: Any, collision_pairs: Any, n_collisions: Any) -> None
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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L1421-L1490)

## particula.gpu.kernels.coagulation.coagulation_step_gpu

*function*

```python
def coagulation_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: object, volume: float | Any | None = None, max_collisions: object = 256, rng_seed: int = 0, collision_pairs: Any | None = None, n_collisions: Any | None = None, rng_states: Any | None = None, *, mechanism_config: CoagulationMechanismConfig | None = None, initialize_rng: bool = False, environment: Any | None = None, validate_charge_finite: bool = False, turbulent_dissipation: float | Any | None = None, fluid_density: float | Any | None = None) -> tuple[Any, Any, Any]
```

Prepare and enqueue one direct particle-resolved coagulation step.

This public wrapper preserves the direct API while keeping all validation,
allocation, and optional RNG initialization in the private preparation
phase. The enqueue phase uses only the resulting frozen call record.

**Returns**

- (`Any`) — The particle container, collision-pair output, and per-box collision
- (`Any`) — counts returned by the direct kernel.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2782-L2830)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/coagulation.py#L2008-L2048)
