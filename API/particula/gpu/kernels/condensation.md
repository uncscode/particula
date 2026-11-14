# particula.gpu.kernels.condensation

*module*

GPU condensation kernels and orchestration utilities.

This module composes condensation ``@wp.func`` building blocks into end-to-end
kernels and provides the high-level ``condensation_step_gpu`` API. The private
Inventory finalization inventory finalizer is direct-test-only: it accepts an
already partitioning-gated raw
proposal, bounds and finalizes it, and mutates particle masses and inventory
finalization scratch
sidecars without coupling gas. The public step performs that coupling after
each finalized inventory finalization transfer.

Tagged native calls require an explicit ``CondensationSpeciesBinding``.
Particle-material arrays retain full composition; gas properties, partitioning,
limiting reductions and energy outputs retain full gas order. Resolved counts
use N/V for gas density coupling and raw N for extensive energy. Discrete PMF
uses stored density and requires V=1. GPU PDF and untagged populations are
unsupported; both reject before scientific execution.

Entry-point validation accepts scalar direct inputs, explicit ``(n_boxes,)``
Warp arrays, or a ``WarpEnvironmentData`` container. Aggregate preflight,
including supplied ``CondensationScratchBuffers`` metadata, completes before
fallback allocation, vapor-pressure refresh, or mutation of caller-owned
State. Physical validation may launch scans and read private status flags.
Private preparation freezes the exact validated primary
arrays, supplied sidecars, and resolved fallback storage for a subsequent
device-only enqueue. A required keyword-only ``ThermodynamicsConfig`` then
executes exactly four equal substeps. Each substep refreshes the caller-owned
pure-vapor-pressure buffer, updates box-level environment properties, proposes
transfer from current particle and gas state, proposal gating-gates disabled
species and
zero-concentration slots, finalizes the inventory-limited applied transfer,
updates particle mass and gas concentration, and accumulates the finalized
whole-call total. Float32 temperatures are cast to a step-owned float64 device
buffer for refresh. Latent heat, when supplied, corrects each fixed substep.
An optional caller-owned energy-transfer output records signed whole-call
finalized-transfer energy on the active device, while thermal-work remains
validated caller-owned state reserved for later work.

Private prepared calls preserve this validation ordering, normalization, and
fallback allocation during setup; retained-reference enqueue performs no
Validation, allocation, readback, transfer, synchronization, lookup, rebinding,
or reset. Rejection is pre-writer; rollback is not promised after a writer
launches.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1-L1)

## particula.gpu.kernels.condensation.ACTIVITY_MODE_IDEAL

*attribute* · *module attribute*

```python
ACTIVITY_MODE_IDEAL = wp.int32(0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L126-L126)

## particula.gpu.kernels.condensation.ACTIVITY_MODE_KAPPA

*attribute* · *module attribute*

```python
ACTIVITY_MODE_KAPPA = wp.int32(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L127-L127)

## particula.gpu.kernels.condensation.SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED

*attribute* · *module attribute*

```python
SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED = wp.int32(1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L129-L129)

## particula.gpu.kernels.condensation.SURFACE_TENSION_MODE_STATIC

*attribute* · *module attribute*

```python
SURFACE_TENSION_MODE_STATIC = wp.int32(0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L128-L128)

## particula.gpu.kernels.condensation._DEFAULT_DIFFUSION_COEFFICIENT

*attribute* · *module attribute*

```python
_DEFAULT_DIFFUSION_COEFFICIENT = 2e-05
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L116-L116)

## particula.gpu.kernels.condensation._DEFAULT_MASS_ACCOMMODATION

*attribute* · *module attribute*

```python
_DEFAULT_MASS_ACCOMMODATION = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L115-L115)

## particula.gpu.kernels.condensation._DEFAULT_SURFACE_TENSION

*attribute* · *module attribute*

```python
_DEFAULT_SURFACE_TENSION = 0.072
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L114-L114)

## particula.gpu.kernels.condensation._GAS_CONSTANT

*attribute* · *module attribute*

```python
_GAS_CONSTANT = wp.constant(wp.float64(constants.GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L117-L117)

## particula.gpu.kernels.condensation._MIN_NATIVE_RADIUS

*attribute* · *module attribute*

```python
_MIN_NATIVE_RADIUS = wp.constant(wp.float64(1e-10))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L119-L119)

## particula.gpu.kernels.condensation._MOLECULAR_WEIGHT_AIR

*attribute* · *module attribute*

```python
_MOLECULAR_WEIGHT_AIR = wp.constant(wp.float64(constants.MOLECULAR_WEIGHT_AIR))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L118-L118)

## particula.gpu.kernels.condensation._REF_TEMPERATURE_STP

*attribute* · *module attribute*

```python
_REF_TEMPERATURE_STP = wp.constant(wp.float64(constants.REF_TEMPERATURE_STP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L123-L123)

## particula.gpu.kernels.condensation._REF_VISCOSITY_AIR_STP

*attribute* · *module attribute*

```python
_REF_VISCOSITY_AIR_STP = wp.constant(wp.float64(constants.REF_VISCOSITY_AIR_STP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L120-L122)

## particula.gpu.kernels.condensation._SUTHERLAND_CONSTANT

*attribute* · *module attribute*

```python
_SUTHERLAND_CONSTANT = wp.constant(wp.float64(constants.SUTHERLAND_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L124-L124)

## particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig

*class* · *dataclass*

```python
class CondensationActivitySurfaceConfig
```

Configure caller-owned activity and surface-tension sidecar inputs.

The frozen dataclass prevents rebinding fields, but does not make its Warp
arrays immutable. ``kappas`` and ``molar_mass_reference`` therefore remain
caller-owned mutable ``wp.float64`` arrays with shape ``(n_species,)``;
native species order is particle material order. Callers must retain
their lifetime and must not modify them concurrently
with a launch. Activity is evaluated only for ``water_species_index``;
every non-water vapor uses unit activity. Static tension uses the current
condensing-species index, while composition-weighted tension computes one
particle-wide value for every condensing species.

**Attributes**

- `activity_mode` (`Any`) — Integer selector: ``0`` for ideal or ``1`` for kappa water activity.
- `surface_tension_mode` (`Any`) — Integer selector: ``0`` for static or ``1`` for composition-weighted surface tension.
- `water_species_index` (`Any`) — Index of the sole species receiving activity; all other vapor species use unit activity.
- `kappas` (`Any`) — Caller-owned finite, non-negative ``wp.float64`` array of per-species kappa parameters with shape ``(n_species,)``.
- `molar_mass_reference` (`Any`) — Caller-owned positive ``wp.float64`` array of per-species molar masses [kg/mol], shaped ``(n_species,)`` and equal to mapped full-gas values for native calls. Unmapped material molar masses are independent. Legacy calls require gas order.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L132-L165)

### particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig.activity_mode

*attribute* · *instance attribute*

```python
activity_mode: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L161-L161)

### particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig.kappas

*attribute* · *instance attribute*

```python
kappas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L164-L164)

### particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig.molar_mass_reference

*attribute* · *instance attribute*

```python
molar_mass_reference: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L165-L165)

### particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig.surface_tension_mode

*attribute* · *instance attribute*

```python
surface_tension_mode: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L162-L162)

### particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig.water_species_index

*attribute* · *instance attribute*

```python
water_species_index: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L163-L163)

### particula.gpu.kernels.condensation.CondensationActivitySurfaceConfig.__init__

*method*

```python
def __init__(activity_mode: Any, surface_tension_mode: Any, water_species_index: Any, kappas: Any, molar_mass_reference: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L0-L0)

## particula.gpu.kernels.condensation.CondensationScratchBuffers

*class* · *dataclass*

```python
class CondensationScratchBuffers
```

Own reusable stable-shape buffers for a condensation-step update.

This concrete-module-only sidecar is intentionally not exported. Every
non-``None`` field is a caller-owned, active-device ``wp.float64`` array
whose shape must remain stable: transfer fields have shape
``(n_boxes, n_particles, n_species)``, property fields have shape
``(n_boxes,)``, and inventory finalization-only limiting sidecars have shape
``(n_boxes, n_gas)``. Native ``n_species`` is particle-material width and
``n_gas`` is complete gas width and may differ from material width.
Fields may be omitted independently; omitted fields are
step-local fallback allocations, while successful calls preserve every
supplied object's identity. Transfer roles are deliberately separate: work
retains the gated raw proposal from the final substep, while total records
the applied transfer over the complete four-substep step.

Each call performs four fixed updates. The total buffer is cleared once
    after preflight and accumulates the inventory-limited, inventory-limited
    transfer applied in each substep; a supplied ``total_mass_transfer`` is
    returned by identity.
Callers must keep supplied arrays alive and unmodified until launched work
completes, and may reuse them only after a successful completion. Complete
supplied-field metadata validation precedes allocation, environment
normalization, refresh, launch, and mutation, so a validation failure
leaves caller-owned state unchanged.

**Attributes**

- `work_mass_transfer` (`Any | None`) — Optional caller-owned work-transfer array with shape ``(n_boxes, n_particles, n_species)``.
- `total_mass_transfer` (`Any | None`) — Optional caller-owned applied-transfer accumulator with shape ``(n_boxes, n_particles, n_species)``.
- `dynamic_viscosity` (`Any | None`) — Optional caller-owned per-box dynamic-viscosity array with shape ``(n_boxes,)`` [Pa·s].
- `mean_free_path` (`Any | None`) — Optional caller-owned per-box mean-free-path array with shape ``(n_boxes,)`` [m].
- `positive_mass_transfer_demand` (`Any | None`) — Optional inventory finalization-only per-box, per-species reduction storage. Proposal gating validates but does not read or write it.
- `negative_mass_transfer_release` (`Any | None`) — Optional inventory finalization-only per-box, per-species reduction storage. Proposal gating validates but does not read or write it.
- `positive_mass_transfer_scale` (`Any | None`) — Optional inventory finalization-only per-box, per-species scale storage. Proposal gating validates but does not read or write it.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L168-L224)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.dynamic_viscosity

*attribute* · *class attribute* · *instance attribute*

```python
dynamic_viscosity: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L220-L220)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.mean_free_path

*attribute* · *class attribute* · *instance attribute*

```python
mean_free_path: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L221-L221)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.negative_mass_transfer_release

*attribute* · *class attribute* · *instance attribute*

```python
negative_mass_transfer_release: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L223-L223)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.positive_mass_transfer_demand

*attribute* · *class attribute* · *instance attribute*

```python
positive_mass_transfer_demand: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L222-L222)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.positive_mass_transfer_scale

*attribute* · *class attribute* · *instance attribute*

```python
positive_mass_transfer_scale: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L224-L224)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.total_mass_transfer

*attribute* · *class attribute* · *instance attribute*

```python
total_mass_transfer: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L219-L219)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.work_mass_transfer

*attribute* · *class attribute* · *instance attribute*

```python
work_mass_transfer: Any | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L218-L218)

### particula.gpu.kernels.condensation.CondensationScratchBuffers.__init__

*method*

```python
def __init__(work_mass_transfer: Any | None = None, total_mass_transfer: Any | None = None, dynamic_viscosity: Any | None = None, mean_free_path: Any | None = None, positive_mass_transfer_demand: Any | None = None, negative_mass_transfer_release: Any | None = None, positive_mass_transfer_scale: Any | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L0-L0)

## particula.gpu.kernels.condensation._PreparedCondensationCall

*class* · *dataclass*

```python
class _PreparedCondensationCall
```

Retain validated state and device operations for one enqueue.

Setup owns validation, normalization, and fallback allocation. The enqueue
phase reuses these retained references for its four device substeps without
validation, allocation, readback, transfer, synchronization, lookup,
rebinding, or reset. The record is private because it captures
implementation-specific Warp callables and scratch layouts rather than a
public execution protocol. Rejection is pre-writer; rollback is not promised
after a writer launches.

**Attributes**

- `particles` (`Any`) — Validated resident Warp particle container returned by enqueue.
- `gas` (`Any`) — Validated resident Warp gas container retained for call identity.
- `thermodynamics` (`Any`) — Validated thermodynamic refresh configuration retained for call identity.
- `total_mass_transfer` (`Any`) — Whole-call finalized-transfer accumulator.
- `work_mass_transfer` (`Any`) — Raw-proposal workspace reused by each substep.
- `dimensions` (`tuple[int, int, int]`) — Fixed ``(n_boxes, n_particles, n_species)`` shape.
- `device` (`Any`) — Active Warp device for all retained arrays.
- `substep_time_step` (`Any`) — Equal duration of each of four substeps [s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L227-L301)

### particula.gpu.kernels.condensation._PreparedCondensationCall.activity_enabled

*attribute* · *instance attribute*

```python
activity_enabled: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L270-L270)

### particula.gpu.kernels.condensation._PreparedCondensationCall.activity_mode

*attribute* · *instance attribute*

```python
activity_mode: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L271-L271)

### particula.gpu.kernels.condensation._PreparedCondensationCall.candidate

*attribute* · *instance attribute*

```python
candidate: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L263-L263)

### particula.gpu.kernels.condensation._PreparedCondensationCall.demand

*attribute* · *instance attribute*

```python
demand: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L260-L260)

### particula.gpu.kernels.condensation._PreparedCondensationCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L282-L282)

### particula.gpu.kernels.condensation._PreparedCondensationCall.diffusion_coefficient_vapor

*attribute* · *instance attribute*

```python
diffusion_coefficient_vapor: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L257-L257)

### particula.gpu.kernels.condensation._PreparedCondensationCall.dimensions

*attribute* · *instance attribute*

```python
dimensions: tuple[int, int, int]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L281-L281)

### particula.gpu.kernels.condensation._PreparedCondensationCall.distribution_kind

*attribute* · *instance attribute*

```python
distribution_kind: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L298-L298)

### particula.gpu.kernels.condensation._PreparedCondensationCall.dynamic_viscosity

*attribute* · *instance attribute*

```python
dynamic_viscosity: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L266-L266)

### particula.gpu.kernels.condensation._PreparedCondensationCall.effective_surface_tension

*attribute* · *instance attribute*

```python
effective_surface_tension: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L274-L274)

### particula.gpu.kernels.condensation._PreparedCondensationCall.energy_transfer

*attribute* · *instance attribute*

```python
energy_transfer: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L280-L280)

### particula.gpu.kernels.condensation._PreparedCondensationCall.failure

*attribute* · *class attribute* · *instance attribute*

```python
failure: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L300-L300)

### particula.gpu.kernels.condensation._PreparedCondensationCall.finalized

*attribute* · *instance attribute*

```python
finalized: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L264-L264)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas

*attribute* · *instance attribute*

```python
gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L253-L253)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_concentration

*attribute* · *instance attribute*

```python
gas_concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L287-L287)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_delta

*attribute* · *instance attribute*

```python
gas_delta: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L265-L265)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_molar_mass

*attribute* · *instance attribute*

```python
gas_molar_mass: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L289-L289)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_partitioning

*attribute* · *instance attribute*

```python
gas_partitioning: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L290-L290)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_species_count

*attribute* · *instance attribute*

```python
gas_species_count: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L299-L299)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_to_particle

*attribute* · *instance attribute*

```python
gas_to_particle: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L295-L295)

### particula.gpu.kernels.condensation._PreparedCondensationCall.gas_vapor_pressure

*attribute* · *instance attribute*

```python
gas_vapor_pressure: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L288-L288)

### particula.gpu.kernels.condensation._PreparedCondensationCall.kappas

*attribute* · *instance attribute*

```python
kappas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L268-L268)

### particula.gpu.kernels.condensation._PreparedCondensationCall.kernel_pressure

*attribute* · *instance attribute*

```python
kernel_pressure: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L276-L276)

### particula.gpu.kernels.condensation._PreparedCondensationCall.latent_heat

*attribute* · *instance attribute*

```python
latent_heat: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L277-L277)

### particula.gpu.kernels.condensation._PreparedCondensationCall.latent_heat_enabled

*attribute* · *instance attribute*

```python
latent_heat_enabled: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L279-L279)

### particula.gpu.kernels.condensation._PreparedCondensationCall.latent_heat_values

*attribute* · *instance attribute*

```python
latent_heat_values: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L278-L278)

### particula.gpu.kernels.condensation._PreparedCondensationCall.mass_accommodation

*attribute* · *instance attribute*

```python
mass_accommodation: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L256-L256)

### particula.gpu.kernels.condensation._PreparedCondensationCall.mean_free_path

*attribute* · *instance attribute*

```python
mean_free_path: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L267-L267)

### particula.gpu.kernels.condensation._PreparedCondensationCall.molar_mass_reference

*attribute* · *instance attribute*

```python
molar_mass_reference: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L269-L269)

### particula.gpu.kernels.condensation._PreparedCondensationCall.parallel_reductions

*attribute* · *class attribute* · *instance attribute*

```python
parallel_reductions: Any = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L301-L301)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particle_charge

*attribute* · *instance attribute*

```python
particle_charge: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L297-L297)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particle_concentration

*attribute* · *instance attribute*

```python
particle_concentration: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L285-L285)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particle_density

*attribute* · *instance attribute*

```python
particle_density: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L286-L286)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particle_masses

*attribute* · *instance attribute*

```python
particle_masses: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L284-L284)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particle_to_gas

*attribute* · *instance attribute*

```python
particle_to_gas: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L294-L294)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particle_volume

*attribute* · *instance attribute*

```python
particle_volume: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L296-L296)

### particula.gpu.kernels.condensation._PreparedCondensationCall.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L252-L252)

### particula.gpu.kernels.condensation._PreparedCondensationCall.refresh_temperature

*attribute* · *instance attribute*

```python
refresh_temperature: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L275-L275)

### particula.gpu.kernels.condensation._PreparedCondensationCall.release

*attribute* · *instance attribute*

```python
release: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L261-L261)

### particula.gpu.kernels.condensation._PreparedCondensationCall.scale

*attribute* · *instance attribute*

```python
scale: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L262-L262)

### particula.gpu.kernels.condensation._PreparedCondensationCall.species_binding

*attribute* · *instance attribute*

```python
species_binding: CondensationSpeciesBinding
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L293-L293)

### particula.gpu.kernels.condensation._PreparedCondensationCall.substep_time_step

*attribute* · *instance attribute*

```python
substep_time_step: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L283-L283)

### particula.gpu.kernels.condensation._PreparedCondensationCall.surface_tension

*attribute* · *instance attribute*

```python
surface_tension: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L255-L255)

### particula.gpu.kernels.condensation._PreparedCondensationCall.surface_tension_mode

*attribute* · *instance attribute*

```python
surface_tension_mode: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L272-L272)

### particula.gpu.kernels.condensation._PreparedCondensationCall.thermodynamics

*attribute* · *instance attribute*

```python
thermodynamics: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L254-L254)

### particula.gpu.kernels.condensation._PreparedCondensationCall.thermodynamics_modes

*attribute* · *instance attribute*

```python
thermodynamics_modes: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L291-L291)

### particula.gpu.kernels.condensation._PreparedCondensationCall.thermodynamics_parameters

*attribute* · *instance attribute*

```python
thermodynamics_parameters: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L292-L292)

### particula.gpu.kernels.condensation._PreparedCondensationCall.total_mass_transfer

*attribute* · *instance attribute*

```python
total_mass_transfer: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L258-L258)

### particula.gpu.kernels.condensation._PreparedCondensationCall.water_species_index

*attribute* · *instance attribute*

```python
water_species_index: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L273-L273)

### particula.gpu.kernels.condensation._PreparedCondensationCall.work_mass_transfer

*attribute* · *instance attribute*

```python
work_mass_transfer: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L259-L259)

### particula.gpu.kernels.condensation._PreparedCondensationCall.__init__

*method*

```python
def __init__(particles: Any, gas: Any, thermodynamics: Any, surface_tension: Any, mass_accommodation: Any, diffusion_coefficient_vapor: Any, total_mass_transfer: Any, work_mass_transfer: Any, demand: Any, release: Any, scale: Any, candidate: Any, finalized: Any, gas_delta: Any, dynamic_viscosity: Any, mean_free_path: Any, kappas: Any, molar_mass_reference: Any, activity_enabled: Any, activity_mode: Any, surface_tension_mode: Any, water_species_index: Any, effective_surface_tension: Any, refresh_temperature: Any, kernel_pressure: Any, latent_heat: Any, latent_heat_values: Any, latent_heat_enabled: Any, energy_transfer: Any, dimensions: tuple[int, int, int], device: Any, substep_time_step: Any, particle_masses: Any, particle_concentration: Any, particle_density: Any, gas_concentration: Any, gas_vapor_pressure: Any, gas_molar_mass: Any, gas_partitioning: Any, thermodynamics_modes: Any, thermodynamics_parameters: Any, species_binding: CondensationSpeciesBinding, particle_to_gas: Any, gas_to_particle: Any, particle_volume: Any, particle_charge: Any, distribution_kind: int, gas_species_count: int, failure: Any = None, parallel_reductions: Any = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L0-L0)

## particula.gpu.kernels.condensation._accumulate_finalized_mass_transfer_kernel

*function*

```python
def _accumulate_finalized_mass_transfer_kernel(finalized: Any, total_mass_transfer: Any) -> None
```

Add one inventory-limited transfer to the whole-call total.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1118-L1129)

## particula.gpu.kernels.condensation._bound_evaporation_candidate_kernel

*function*

```python
def _bound_evaporation_candidate_kernel(masses: Any, gated_transfer: Any, candidate: Any) -> None
```

Copy a gated proposal while bounding evaporation by owned mass.

**Parameters**

- `masses` (`Any`) — Particle masses ``(n_boxes, n_particles, n_species)`` [kg].
- `gated_transfer` (`Any`) — Already partitioning-gated transfer proposal matching ``masses`` [kg].
- `candidate` (`Any`) — Output bounded proposal matching ``masses`` [kg].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1185-L1207)

## particula.gpu.kernels.condensation._clear_energy_transfer_kernel

*function*

```python
def _clear_energy_transfer_kernel(energy_transfer: Any) -> None
```

Clear caller-owned whole-call energy diagnostic storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1107-L1115)

## particula.gpu.kernels.condensation._clear_mass_transfer_kernel

*function*

```python
def _clear_mass_transfer_kernel(total_mass_transfer: Any) -> None
```

Clear the fixed-shape accumulator before the four-substep update.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1094-L1104)

## particula.gpu.kernels.condensation._condensation_mass_transfer_impl

*function*

```python
def _condensation_mass_transfer_impl(masses: Any, concentration: Any, density: Any, gas_concentration: Any, vapor_pressure: Any, molar_mass: Any, surface_tension: Any, kappas: Any, molar_mass_reference: Any, effective_surface_tension: Any, activity_enabled: wp.int32, activity_mode: wp.int32, surface_tension_mode: wp.int32, water_species_index: wp.int32, mass_accommodation: Any, diffusion_coefficient_vapor: Any, latent_heat: Any, latent_heat_enabled: wp.int32, dynamic_viscosity: Any, mean_free_path: Any, gas_constant: Any, boltzmann_constant: Any, temperature: Any, time_step: Any, mass_transfer: Any, particle_to_gas: Any, native: wp.int32, box_idx: int, particle_idx: int) -> None
```

Shared raw proposal with explicit gas lookup only for native calls.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L493-L783)

## particula.gpu.kernels.condensation._copy_pressure_to_float64_kernel

*function*

```python
def _copy_pressure_to_float64_kernel(pressure: Any, output: Any) -> None
```

Copy normalized pressures into a float64 device buffer.

**Parameters**

- `pressure` (`Any`) — Normalized per-box pressure array ``(n_boxes,)`` [Pa].
- `output` (`Any`) — Float64 per-box output array ``(n_boxes,)`` [Pa].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L987-L1000)

## particula.gpu.kernels.condensation._copy_temperature_to_float64_kernel

*function*

```python
def _copy_temperature_to_float64_kernel(temperature: Any, output: Any) -> None
```

Copy normalized temperatures to a float64 device buffer.

This device-only helper prepares non-float64 normalized temperatures for
thermodynamic vapor-pressure refresh without changing the input buffer.

**Parameters**

- `temperature` (`Any`) — Normalized per-box temperature array ``(n_boxes,)`` [K].
- `output` (`Any`) — Float64 per-box output array ``(n_boxes,)`` [K].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L968-L984)

## particula.gpu.kernels.condensation._couple_finalized_transfer_to_gas_kernel

*function*

```python
def _couple_finalized_transfer_to_gas_kernel(gas_delta: Any, gas_concentration: Any) -> None
```

Run the shared device operation through the direct kernel API.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1156-L1166)

## particula.gpu.kernels.condensation._couple_finalized_transfer_to_gas_kernel_body

*function*

```python
def _couple_finalized_transfer_to_gas_kernel_body(gas_delta: Any, gas_concentration: Any, box_idx: wp.int32, species_idx: wp.int32) -> None
```

Couple a prevalidated finalized-transfer reduction to gas.

Each box-species lane subtracts a finite, concentration-weighted particle
transfer delta from gas. Negative finalized transfer therefore adds
evaporated inventory to gas.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1132-L1153)

## particula.gpu.kernels.condensation._couple_finalized_transfer_to_gas_kernel_guarded

*function*

```python
def _couple_finalized_transfer_to_gas_kernel_guarded(gas_delta: Any, gas_concentration: Any, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1169-L1182)

## particula.gpu.kernels.condensation._effective_surface_tension_kernel

*function*

```python
def _effective_surface_tension_kernel(masses: Any, density: Any, surface_tension: Any, output: Any) -> None
```

Compute one composition-weighted surface tension per particle.

The output is recalculated from the current mass in each substep and reused
for every condensing species of that particle, avoiding a full composition
reduction per species.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L949-L965)

## particula.gpu.kernels.condensation._enqueue_prepared_condensation_call

*function*

```python
def _enqueue_prepared_condensation_call(prepared: _PreparedCondensationCall) -> tuple[Any, Any]
```

Enqueue the bound four-substep device sequence without setup work.

This helper is the enqueue half of the preparation boundary. It reuses only
references retained by ``prepared`` and does not repeat public input
validation, normalization, fallback resolution, resource lookup, allocation,
readback, or synchronization. It uses exact primary and sidecar references
frozen by setup, so rebinding a container attribute after setup cannot alter
the launched work. The call clears supplied output accumulators once, then
executes the refresh, proposal, inventory finalization finalization,
particle update, gas
coupling, and diagnostic sequence four times.

**Parameters**

- `prepared` (`_PreparedCondensationCall`) — Fully validated call record returned by ``_prepare_condensation_step_gpu``.

**Returns**

- (`Any`) — A tuple containing the resident particle container and the accumulated
- (`Any`) — finalized mass-transfer buffer, both retained by the prepared record.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L2932-L3183)

## particula.gpu.kernels.condensation._finalize_and_apply_inventory_transfer_kernel

*function*

```python
def _finalize_and_apply_inventory_transfer_kernel(masses: Any, candidate: Any, scale: Any, finalized: Any) -> None
```

Finalize bounded evaporation and inventory-scaled uptake.

Negative candidates are retained and positive candidates are multiplied by
their box-species scale. This kernel has no gas-concentration argument and
therefore does not perform gas coupling.

**Parameters**

- `masses` (`Any`) — Particle masses used to determine the species count [kg].
- `candidate` (`Any`) — Bounded, already gated transfer proposal [kg].
- `scale` (`Any`) — Positive-transfer fractions by box and species.
- `finalized` (`Any`) — Output applied transfer matching ``masses`` [kg].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1282-L1307)

## particula.gpu.kernels.condensation._finalize_inventory_limited_mass_transfer

*function*

```python
def _finalize_inventory_limited_mass_transfer(particles: Any, gas: Any, gated_mass_transfer: Any, scratch_buffers: CondensationScratchBuffers | Any | None = None, *, energy_transfer: Any | None = None) -> Any
```

Finalize a pre-gated proposal under particle and gas inventory limits.

This private, direct-test-only inventory finalization primitive first bounds
evaporation by
owned particle mass. It then limits concentration-weighted positive uptake
for each box and species to gas inventory plus permitted evaporation
release. It mutates particle masses and any supplied inventory finalization
sidecars, but
deliberately does not mutate ``gas.concentration``. This primitive accepts
explicitly discrete number-density weights and equal particle/gas lanes.
The public mapped step owns its separate gas coupling and supports resolved
counts with volume normalization; this primitive does not interpret counts.

**Parameters**

- `particles` (`Any`) — GPU particle data with float64 masses and concentrations.
- `gas` (`Any`) — GPU gas data whose concentration supplies read-only inventory.
- `gated_mass_transfer` (`Any`) — Float64 ``(n_boxes, n_particles, n_species)`` proposal [kg] already partitioning-gated for partitioning and inactive slots.
- `scratch_buffers` (`CondensationScratchBuffers | Any | None`) (default: `None`) — Optional validated scratch sidecar. Only its inventory finalization demand, release, and scale fields are resolved or mutated.
- `energy_transfer` (`Any | None`) (default: `None`) — Optional public-step energy output to protect from inventory finalization sidecar aliasing. It is never read or written by this helper.

**Returns**

- (`Any`) — New float64 finalized applied-transfer array matching particle masses
- (`Any`) — [kg].

**Raises**

- `ValueError` — If physical inputs are invalid, buffers have incompatible
metadata, or inventory finalization sidecars overlap protected or
peer storage.

> **Notes**
>
> Preflight validation finishes before this helper resolves fallback
> state, allocates output, or launches a mutating kernel. Validation
> failures leave caller-owned state unchanged; this does not promise
> rollback after a post-launch device-runtime failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1981-L2124)

## particula.gpu.kernels.condensation._mapped_condensation_mass_transfer_kernel

*function*

```python
def _mapped_condensation_mass_transfer_kernel(masses: Any, concentration: Any, density: Any, gas_concentration: Any, vapor_pressure: Any, molar_mass: Any, surface_tension: Any, kappas: Any, molar_mass_reference: Any, effective_surface_tension: Any, activity_enabled: wp.int32, activity_mode: wp.int32, surface_tension_mode: wp.int32, water_species_index: wp.int32, mass_accommodation: Any, diffusion_coefficient_vapor: Any, latent_heat: Any, latent_heat_enabled: wp.int32, dynamic_viscosity: Any, mean_free_path: Any, gas_constant: Any, boltzmann_constant: Any, temperature: Any, time_step: Any, mass_transfer: Any, particle_to_gas: Any) -> None
```

Compute particle-ordered proposals from full-gas mapped sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L885-L946)

## particula.gpu.kernels.condensation._prepare_condensation_step_gpu

*function*

```python
def _prepare_condensation_step_gpu(particles: Any, gas: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float, surface_tension: Any | None = None, mass_accommodation: Any | None = None, diffusion_coefficient_vapor: Any | None = None, mass_transfer: Any | None = None, *, environment: Any | None = None, thermodynamics: ThermodynamicsConfig | Any | None = None, activity_surface: CondensationActivitySurfaceConfig | Any | None = None, scratch_buffers: CondensationScratchBuffers | Any | None = None, latent_heat: Any | None = None, energy_transfer: Any | None = None, thermal_work: Any | None = None, species_binding: CondensationSpeciesBinding | None = None) -> _PreparedCondensationCall
```

Validate and bind a fixed four-substep condensation device call.

**Parameters**

- `particles` (`Any`) — GPU-resident particle data.
- `gas` (`Any`) — GPU-resident gas data. ``gas.partitioning`` must be an active-device binary ``wp.int32`` mask shaped ``(n_boxes, n_gas)``.
- `temperature` (`float | Any | None`) — Direct gas temperature as either a scalar or a Warp array with shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
- `pressure` (`float | Any | None`) — Direct gas pressure as either a scalar or a Warp array with shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
- `time_step` (`float`) — Total condensation time step [s], divided into four equal substeps.
- `surface_tension` (`Any | None`) (default: `None`) — Optional per-species surface tension [N/m].
- `mass_accommodation` (`Any | None`) (default: `None`) — Optional per-species accommodation coefficient.
- `diffusion_coefficient_vapor` (`Any | None`) (default: `None`) — Optional per-species vapor diffusion coefficient [m^2/s].
- `mass_transfer` (`Any | None`) (default: `None`) — Optional preallocated whole-call accumulator for the inventory-limited applied transfer [kg], with shape ``(n_boxes, n_particles, n_species)``. It is not raw proposal workspace.
- `environment` (`Any | None`) (default: `None`) — Optional ``WarpEnvironmentData`` with ``(n_boxes,)`` temperature and pressure arrays on the same device as ``particles`` and ``gas``. This mode is supported when both direct inputs are ``None``.
- `thermodynamics` (`ThermodynamicsConfig | Any | None`) (default: `None`) — Required caller-owned, device-local thermodynamic sidecar with model arrays matching the ordered gas species.
- `activity_surface` (`CondensationActivitySurfaceConfig | Any | None`) (default: `None`) — Optional frozen caller-owned sidecar with device-local ``wp.float64`` per-species arrays. It enables ideal or kappa activity only for ``water_species_index``; non-water species retain unit activity. Static tension uses the current condensing-species index, while weighted tension uses one particle-wide value. ``None`` uses unit activity and material-indexed static tension.
- `scratch_buffers` (`CondensationScratchBuffers | Any | None`) (default: `None`) — Optional frozen caller-owned buffers. Supplied transfer fields have shape ``(n_boxes, n_particles, n_species)``; supplied property fields have shape ``(n_boxes,)``. Every supplied field must be active-device ``wp.float64``. Fields may be omitted independently and use step-local fallback allocation. A supplied total transfer buffer is returned by identity. Work retains the final partitioning-gated raw proposal; total accumulates inventory-limited applied transfers. Inventory finalization limiting sidecars have shape ``(n_boxes, n_gas)``, validate before launch, and are reused by all four inventory finalization finalization cycles. validation is performed before allocation or mutation. Arrays must remain alive and unmodified until launched work completes and may be reused only after a successful completion.
- `latent_heat` (`Any | None`) (default: `None`) — Optional caller-owned per-species latent heat [J/kg] as an active-device ``wp.float64`` array shaped ``(n_species,)``. When supplied, it is consumed in every fixed substep; zero entries use the exact isothermal rate path.
- `energy_transfer` (`Any | None`) (default: `None`) — Optional caller-owned, active-device ``wp.float64`` write-only energy output [J] shaped ``(n_boxes, n_gas)``. It requires valid ``latent_heat``. After successful preflight, the same buffer is cleared and overwritten with signed whole-call finalized applied transfer times per-species ``latent_heat``; callers may reuse it on a later successful call. Its existing contents, including stale NaN/Inf, are deliberately not validated because it is output-only diagnostic storage. It is not returned as a third tuple item.
- `thermal_work` (`Any | None`) (default: `None`) — Optional caller-owned per-species thermal-work sidecar [J/kg] as an active-device ``wp.float64`` array shaped ``(n_species,)``. It is validated but deferred and unused thermal correction state; this step does not allocate, initialize, modify, or consume it.
- `species_binding` (`CondensationSpeciesBinding | None`) (default: `None`) — Explicit native gas/material map pinned to these exact carriers. Native transport/thermal arrays and reductions use full gas width; activity/surface arrays and transfers use material width. Native energy weights applied transfers by counts (density times unit volume for PMF). Untagged calls reject; this binding is required.

**Returns**

- (`_PreparedCondensationCall`) — Private prepared call retaining validated inputs, resolved device
- (`_PreparedCondensationCall`) — buffers, and the fixed substep duration. It does not enqueue the
- (`_PreparedCondensationCall`) — condensation sequence.

**Raises**

- `ValueError` — If species counts, array lengths, or devices mismatch.
- `ValueError` — If ``gas.partitioning`` is not an active-device binary
 ``wp.int32`` array shaped ``(n_boxes, n_gas)``.
- `ValueError` — If direct ``temperature`` or ``pressure`` inputs are mixed
with ``environment``.
- `ValueError` — If direct inputs are missing when ``environment`` is
omitted.
- `ValueError` — If environment arrays do not match ``(n_boxes,)`` or the
caller device.
- `ValueError` — If direct non-scalar, non-Warp-array temperature or
pressure inputs are provided.
- `ValueError` — If ``thermodynamics`` is absent or does not match the gas
species, active device, or required fixed schema.
- `ValueError` — If ``activity_surface`` violates its frozen-sidecar,
selector, device-array, domain, or molar-mass ordering contract.
- `ValueError` — If ``scratch_buffers`` is not the exact frozen sidecar type,
a supplied field lacks its stable active-device ``wp.float64``
metadata, or ``mass_transfer`` overlaps a supplied scratch transfer
field.
- `ValueError` — If ``latent_heat`` or ``thermal_work`` is not a finite,
non-negative active-device ``wp.float64`` array shaped
``(n_species,)``. Latent heat above ``1e9`` J/kg is also rejected
before launch to prevent non-isothermal arithmetic overflow.
- `ValueError` — If ``energy_transfer`` is supplied without valid
``latent_heat`` or lacks active-device ``wp.float64``
``(n_boxes, n_gas)`` metadata.

> **Notes**
>
> Accepted environment sources are scalar direct inputs, direct
> ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
> inputs, or keyword-only ``environment=...`` execution. NumPy arrays,
> Python lists, and other non-Warp direct array-likes are rejected.
> 
> Particle masses and gas concentration are updated in-place on the GPU.
> Callers requiring rollback must retain or restore pre-call snapshots of
> both fields and of caller-owned output buffers.
> 
> Exactly four equal substeps run for every valid timestep. The returned
> transfer buffer accumulates the inventory-limited applied transfer from
> all
> four substeps. A supplied scratch work buffer holds only the final
> gated raw proposal. Each proposal is zeroed before application for
> disabled ``gas.partitioning`` species and zero-concentration particle
> slots. Each finalized transfer is also subtracted from
> ``gas.concentration`` weighted by particle concentration, so the next
> substep proposal reads coupled gas inventory. Vapor-pressure refresh
> itself does not read gas concentration.
> 
> ``environment`` remains keyword-only so existing positional scalar
> callers stay source-compatible.
> 
> Aggregate configuration and optional-buffer validation completes before
> allocation, launch, refresh, or mutation. Thus a validation failure
> leaves caller-owned particle, gas, environment, sidecar, and supplied
> output buffers unchanged and is retryable with corrected inputs.
> In particular, partitioning metadata and its status-only binary
> validation complete before environment normalization, fallback
> allocation, output clearing, vapor-pressure refresh, or physical
> mutation.
> Each substep overwrites ``gas.vapor_pressure`` from the normalized
> current temperature, prepares box-level properties, calculates and
> gates a raw proposal from current particle mass, then finalizes,
> applies, couples, and accumulates it. Setup performs all host-visible
> validation; the prepared enqueue uses only frozen device references and
> does not allocate, read back, synchronize, or validate dynamically
> generated intermediate values. Earlier completed substeps are not
> rolled back after a device writer launches.
> Float32 temperature arrays are
> cast device-side to float64 for refresh.
> 
> A supplied work and/or total scratch transfer field conflicts with
> ``mass_transfer`` because they overlap its total-transfer output role; a
> property-only sidecar can be used with ``mass_transfer``. The resolved
> total buffer is cleared once and returned by identity when supplied.
> Work is separate from total so it can retain the final gated raw
> proposal.
> With no scratch transfer fields, ``mass_transfer`` remains the
> returned total by identity.
> 
> Thermodynamic metadata and production calculations stay device-resident.
> CUDA numeric validation of each supplied thermal sidecar reads back one
> device validation flag before any caller-owned state is mutated.
> 
> Supplied latent sidecars are caller-owned. Latent heat is consumed per
> fixed substep without mutation; omitting it or supplying zero for a
> species preserves that species' isothermal arithmetic. Thermal work is
> validated but unused. Invalid metadata or values leave physical state
> and all caller-owned work state unchanged. To make overflow detectable
> before launch, finite latent heat is limited to ``1e9`` J/kg.
> 
> Energy transfer is caller-owned, allocation-stable diagnostic storage.
> It is cleared once after successful preflight and records each signed
> finalized substep transfer multiplied by latent heat immediately after
> that substep commits. It therefore remains consistent with committed
> work if a later fresh proposal fails. It is deliberately not
> content-validated because the step
> only writes it. This write-only contract permits stale finite values and
> NaN/Inf before a call.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L2218-L2929)

## particula.gpu.kernels.condensation._prepare_environment_properties_kernel

*function*

```python
def _prepare_environment_properties_kernel(temperature: Any, pressure: Any, dynamic_viscosity: Any, mean_free_path: Any) -> None
```

Precompute box-level gas properties for one condensation substep.

**Parameters**

- `temperature` (`Any`) — Per-box gas temperatures ``(n_boxes,)`` [K].
- `pressure` (`Any`) — Per-box gas pressures ``(n_boxes,)`` [Pa].
- `dynamic_viscosity` (`Any`) — Output dynamic viscosity array ``(n_boxes,)`` [Pa·s].
- `mean_free_path` (`Any`) — Output mean free path array ``(n_boxes,)`` [m].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1003-L1036)

## particula.gpu.kernels.condensation._read_float64_array

*function*

```python
def _read_float64_array(array: Any) -> np.ndarray
```

Return a Warp array's values as a float64 NumPy array for validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L304-L306)

## particula.gpu.kernels.condensation._reduce_inventory_candidates_kernel

*function*

```python
def _reduce_inventory_candidates_kernel(candidate: Any, concentration: Any, demand: Any, release: Any) -> None
```

Reduce concentration-weighted demand and release in index order.

Each ``(box, species)`` lane traverses particles in ascending index order,
avoiding atomic accumulation. Positive candidates contribute to demand;
negative candidates contribute their magnitude to release.

**Parameters**

- `candidate` (`Any`) — Bounded, already gated transfer proposal [kg].
- `concentration` (`Any`) — Particle number concentrations ``(n_boxes, n_particles)`` [1/m^3].
- `demand` (`Any`) — Output positive transfer demand ``(n_boxes, n_species)`` [kg/m^3].
- `release` (`Any`) — Output evaporation release ``(n_boxes, n_species)`` [kg/m^3].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1210-L1243)

## particula.gpu.kernels.condensation._scale_inventory_uptake_kernel

*function*

```python
def _scale_inventory_uptake_kernel(gas_concentration: Any, demand: Any, release: Any, scale: Any) -> None
```

Compute each box-species uptake fraction from available inventory.

Available inventory is gas concentration plus the bounded evaporation
release. The output is one when demand is covered (including zero demand)
and otherwise the available-to-demand ratio clamped to ``[0, 1]``.

**Parameters**

- `gas_concentration` (`Any`) — Gas concentrations ``(n_boxes, n_species)`` [kg/m^3], read without mutation.
- `demand` (`Any`) — Positive transfer demand ``(n_boxes, n_species)`` [kg/m^3].
- `release` (`Any`) — Evaporation release ``(n_boxes, n_species)`` [kg/m^3].
- `scale` (`Any`) — Output positive-transfer fractions ``(n_boxes, n_species)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1246-L1279)

## particula.gpu.kernels.condensation._validate_device_arrays

*function*

```python
def _validate_device_arrays(particles: Any, gas: Any, device: Any) -> None
```

Validate particle and gas arrays share the same Warp device.

**Parameters**

- `particles` (`Any`) — Particle data with GPU-backed arrays.
- `gas` (`Any`) — Gas data with GPU-backed arrays.
- `device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If any particle or gas array is on a different device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1960-L1978)

## particula.gpu.kernels.condensation._validate_device_match

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

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1944-L1957)

## particula.gpu.kernels.condensation._validate_energy_transfer_buffer

*function*

```python
def _validate_energy_transfer_buffer(energy_transfer: Any, expected_shape: tuple[int, int], expected_device: Any) -> None
```

Validate metadata for caller-owned write-only energy output storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1435-L1455)

## particula.gpu.kernels.condensation._validate_energy_transfer_ownership

*function*

```python
def _validate_energy_transfer_ownership(energy_transfer: Any, mutable_arrays: tuple[Any, ...]) -> None
```

Reject energy output storage overlapping condensation state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1511-L1530)

## particula.gpu.kernels.condensation._validate_finite_gas_delta

*function*

```python
def _validate_finite_gas_delta(gas_delta: Any, device: Any) -> None
```

Reject non-finite finalized-transfer reductions before gas mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1705-L1719)

## particula.gpu.kernels.condensation._validate_float64_species_array

*function*

```python
def _validate_float64_species_array(name: str, array: Any, n_species: int, expected_device: Any, *, nonnegative: bool = False, maximum: float | None = None) -> None
```

Validate metadata for a caller-owned float64 per-species Warp array.

Numeric-domain validation runs before the step can clear or mutate any
caller-owned output. CUDA values are checked by a device-side preflight
kernel; only its one-element validation result is read back.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1339-L1382)

## particula.gpu.kernels.condensation._validate_gas_arrays

*function*

```python
def _validate_gas_arrays(gas: Any, n_boxes: int, n_species: int) -> None
```

Validate gas array shapes and lengths.

**Parameters**

- `gas` (`Any`) — Gas data containing molar mass, concentration, and vapor pressure.
- `n_boxes` (`int`) — Expected number of spatial boxes.
- `n_species` (`int`) — Expected number of gas species.

**Raises**

- `ValueError` — If gas array shapes do not match expectations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1861-L1887)

## particula.gpu.kernels.condensation._validate_gas_delta_values_kernel

*function*

```python
def _validate_gas_delta_values_kernel(gas_delta: Any, invalid: Any) -> None
```

Record a non-finite finalized-transfer gas reduction.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1591-L1597)

## particula.gpu.kernels.condensation._validate_int32_selector

*function*

```python
def _validate_int32_selector(name: str, value: Any, allowed: tuple[int, ...]) -> int
```

Validate a non-boolean integer selector before Warp int32 conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L369-L384)

## particula.gpu.kernels.condensation._validate_mass_transfer_buffer

*function*

```python
def _validate_mass_transfer_buffer(mass_transfer: Any, expected_shape: tuple[int, int, int], expected_device: str) -> None
```

Validate mass transfer buffer shape and device.

**Parameters**

- `mass_transfer` (`Any`) — Mass transfer array allocated on the GPU.
- `expected_shape` (`tuple[int, int, int]`) — Expected ``(n_boxes, n_particles, n_species)`` shape.
- `expected_device` (`str`) — Expected Warp device name.

**Raises**

- `ValueError` — If the shape or device does not match expectations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1404-L1432)

## particula.gpu.kernels.condensation._validate_mass_transfer_ownership

*function*

```python
def _validate_mass_transfer_ownership(mass_transfer: Any, state_arrays: tuple[Any, ...]) -> None
```

Reject a total-transfer output that aliases condensation state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1533-L1554)

## particula.gpu.kernels.condensation._validate_native_ownership

*function*

```python
def _validate_native_ownership(particles: Any, gas: Any, binding: CondensationSpeciesBinding, thermodynamics: Any, activity_surface: Any, surface_tension: Any, mass_accommodation: Any, diffusion: Any, latent_heat: Any, thermal_work: Any, mass_transfer: Any, energy_transfer: Any, scratch: Any, temperature: Any, pressure: Any, environment: Any) -> None
```

Reject overlaps involving any native writer before normalization.

Primary arrays, outputs and supplied scratch must not overwrite any peer
or configuration array. Read-only configuration may share storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L2127-L2215)

## particula.gpu.kernels.condensation._validate_no_overlap

*function*

```python
def _validate_no_overlap(read_only_arrays: tuple[Any | None, ...], writable_arrays: tuple[Any | None, ...]) -> None
```

Reject thermal sidecars that alias writable scratch property storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1489-L1508)

## particula.gpu.kernels.condensation._validate_p2_commit_values

*function*

```python
def _validate_p2_commit_values(particles: Any, finalized: Any, total_mass_transfer: Any, gas: Any, gas_delta: Any, device: Any) -> None
```

Validate every prospective inventory finalization state value before its
first commit.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1722-L1766)

## particula.gpu.kernels.condensation._validate_p2_commit_values_kernel

*function*

```python
def _validate_p2_commit_values_kernel(masses: Any, finalized: Any, total_mass_transfer: Any, gas_concentration: Any, gas_delta: Any, invalid: Any) -> None
```

Record invalid prospective inventory finalization values before mutating
public state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1600-L1635)

## particula.gpu.kernels.condensation._validate_p2_primary_state

*function*

```python
def _validate_p2_primary_state(particles: Any, gas: Any, dimensions: tuple[int, int, int], device: Any) -> None
```

Validate finite, non-negative primary inventory finalization state before
launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1638-L1681)

## particula.gpu.kernels.condensation._validate_p2_primary_values_kernel

*function*

```python
def _validate_p2_primary_values_kernel(masses: Any, concentration: Any, gas_concentration: Any, invalid: Any) -> None
```

Record invalid primary inventory finalization state without changing
caller data.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1557-L1578)

## particula.gpu.kernels.condensation._validate_p2_proposal_finiteness

*function*

```python
def _validate_p2_proposal_finiteness(proposal: Any, dimensions: tuple[int, int, int], device: Any) -> None
```

Validate a freshly written proposal gating proposal immediately before
inventory finalization mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1684-L1702)

## particula.gpu.kernels.condensation._validate_p2_proposal_values_kernel

*function*

```python
def _validate_p2_proposal_values_kernel(proposal: Any, invalid: Any) -> None
```

Record a non-finite freshly generated proposal gating proposal.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1581-L1588)

## particula.gpu.kernels.condensation._validate_p2_sidecar_ownership

*function*

```python
def _validate_p2_sidecar_ownership(demand: Any | None, release: Any | None, scale: Any | None, protected_arrays: tuple[tuple[str, Any | None], ...]) -> None
```

Reject inventory finalization reduction sidecars overlapping protected or
peer storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1769-L1796)

## particula.gpu.kernels.condensation._validate_particle_arrays

*function*

```python
def _validate_particle_arrays(particles: Any, n_boxes: int, n_particles: int, n_species: int) -> None
```

Validate particle array shapes and lengths.

**Parameters**

- `particles` (`Any`) — Particle data containing density and concentration arrays.
- `n_boxes` (`int`) — Expected number of spatial boxes.
- `n_particles` (`int`) — Expected number of particles per box.
- `n_species` (`int`) — Expected number of particle species.

**Raises**

- `ValueError` — If particle array shapes do not match expectations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1836-L1858)

## particula.gpu.kernels.condensation._validate_partitioning_metadata

*function*

```python
def _validate_partitioning_metadata(partitioning: Any, n_boxes: int, n_species: int, device: Any, caller_name: str) -> None
```

Validate an active-device per-box partitioning mask's metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1902-L1918)

## particula.gpu.kernels.condensation._validate_partitioning_values

*function*

```python
def _validate_partitioning_values(partitioning: Any, n_boxes: int, n_species: int, device: Any) -> None
```

Validate the binary values after all caller sidecars validate.

The one-element status buffer is private disposable preflight state. Its
readback is necessary to make arbitrary device-resident non-binary values
observable before any caller-owned state is changed.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1921-L1941)

## particula.gpu.kernels.condensation._validate_partitioning_values_kernel

*function*

```python
def _validate_partitioning_values_kernel(partitioning: Any, invalid: Any) -> None
```

Record non-binary entries in a per-box partitioning mask.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1890-L1899)

## particula.gpu.kernels.condensation._validate_scratch_ownership

*function*

```python
def _validate_scratch_ownership(scratch_buffers: CondensationScratchBuffers, protected_arrays: tuple[tuple[str, Any | None], ...]) -> None
```

Reject all supplied scratch buffers overlapping peers or public state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1799-L1833)

## particula.gpu.kernels.condensation._validate_species_array

*function*

```python
def _validate_species_array(name: str, array: Any, n_species: int, expected_device: Any) -> None
```

Validate per-species array length and device.

**Parameters**

- `name` (`str`) — Array name for error messages.
- `array` (`Any`) — Array-like object with ``shape`` attribute.
- `n_species` (`int`) — Expected length.
- `expected_device` (`Any`) — Expected Warp device.

**Raises**

- `ValueError` — If the array length or device mismatches expectations.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1310-L1336)

## particula.gpu.kernels.condensation._validate_species_values_kernel

*function*

```python
def _validate_species_values_kernel(values: Any, require_nonnegative: wp.int32, maximum: wp.float64, invalid: Any) -> None
```

Record whether a species sidecar has an invalid numeric value.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1385-L1401)

## particula.gpu.kernels.condensation._warp_array_memory_range

*function*

```python
def _warp_array_memory_range(array: Any) -> tuple[int, int]
```

Return a contiguous Warp array's byte range.

Ownership checks reject strided views because a single address range cannot
distinguish their gaps from overlapping storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1458-L1486)

## particula.gpu.kernels.condensation.apply_mass_transfer_kernel

*function*

```python
def apply_mass_transfer_kernel(masses: Any, mass_transfer: Any) -> None
```

Run the shared device operation through the direct kernel API.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1065-L1075)

## particula.gpu.kernels.condensation.apply_mass_transfer_kernel_body

*function*

```python
def apply_mass_transfer_kernel_body(masses: Any, mass_transfer: Any, box_idx: wp.int32, particle_idx: wp.int32) -> None
```

Apply condensation mass transfer and clamp masses to non-negative.

**Parameters**

- `masses` (`Any`) — Particle masses array ``(n_boxes, n_particles, n_species)``.
- `mass_transfer` (`Any`) — Mass transfer array matching ``masses`` shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1039-L1062)

## particula.gpu.kernels.condensation.apply_mass_transfer_kernel_guarded

*function*

```python
def apply_mass_transfer_kernel_guarded(masses: Any, mass_transfer: Any, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L1078-L1091)

## particula.gpu.kernels.condensation.condensation_mass_transfer_kernel

*function*

```python
def condensation_mass_transfer_kernel(masses: Any, concentration: Any, density: Any, gas_concentration: Any, vapor_pressure: Any, molar_mass: Any, surface_tension: Any, kappas: Any, molar_mass_reference: Any, effective_surface_tension: Any, activity_enabled: wp.int32, activity_mode: wp.int32, surface_tension_mode: wp.int32, water_species_index: wp.int32, mass_accommodation: Any, diffusion_coefficient_vapor: Any, latent_heat: Any, latent_heat_enabled: wp.int32, dynamic_viscosity: Any, mean_free_path: Any, gas_constant: Any, boltzmann_constant: Any, temperature: Any, time_step: Any, mass_transfer: Any) -> None
```

Compute a raw condensation mass-transfer proposal per particle species.

Water activity is applied only when enabled and the current species equals
``water_species_index``. In configured weighted-tension mode, the supplied
per-particle tension is shared by all condensing species. For each species,
the activity- and Kelvin-adjusted surface vapor pressure is used both for
the gas-to-surface pressure difference and, when enabled, the latent-heat
correction. Omitted latent heat and exactly zero per-species entries use
the original isothermal rate path.

**Parameters**

- `masses` (`Any`) — Particle masses array ``(n_boxes, n_particles, n_species)``.
- `concentration` (`Any`) — Particle number concentration array.
- `density` (`Any`) — Per-species particle density array.
- `gas_concentration` (`Any`) — Gas concentrations array.
- `vapor_pressure` (`Any`) — Gas-phase vapor pressure array.
- `molar_mass` (`Any`) — Gas-phase molar mass array.
- `surface_tension` (`Any`) — Per-species surface tension array.
- `kappas` (`Any`) — Per-species kappa activity parameters.
- `molar_mass_reference` (`Any`) — Per-species activity reference molar masses.
- `effective_surface_tension` (`Any`) — Per-particle weighted tension buffer.
- `activity_enabled` (`wp.int32`) — Flag enabling configured water activity.
- `activity_mode` (`wp.int32`) — Selector for ideal or kappa water activity.
- `surface_tension_mode` (`wp.int32`) — Selector for static or weighted tension.
- `water_species_index` (`wp.int32`) — Index of the sole activity-adjusted species.
- `mass_accommodation` (`Any`) — Per-species mass accommodation coefficients.
- `diffusion_coefficient_vapor` (`Any`) — Per-species vapor diffusion coefficients.
- `latent_heat` (`Any`) — Per-species latent heat [J/kg], unread when disabled.
- `latent_heat_enabled` (`wp.int32`) — Flag selecting latent-heat rate correction.
- `dynamic_viscosity` (`Any`) — Per-box gas dynamic viscosity [Pa·s].
- `mean_free_path` (`Any`) — Per-box gas mean free path [m].
- `gas_constant` (`Any`) — Universal gas constant [J/(mol·K)].
- `boltzmann_constant` (`Any`) — Boltzmann constant [J/K].
- `temperature` (`Any`) — Per-box gas temperature [K].
- `time_step` (`Any`) — Condensation substep duration [s].
- `mass_transfer` (`Any`) — Output raw mass-transfer proposal array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L786-L882)

## particula.gpu.kernels.condensation.condensation_step_gpu

*function*

```python
def condensation_step_gpu(particles: Any, gas: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float, surface_tension: Any | None = None, mass_accommodation: Any | None = None, diffusion_coefficient_vapor: Any | None = None, mass_transfer: Any | None = None, *, environment: Any | None = None, thermodynamics: ThermodynamicsConfig | Any | None = None, activity_surface: CondensationActivitySurfaceConfig | Any | None = None, scratch_buffers: CondensationScratchBuffers | Any | None = None, latent_heat: Any | None = None, energy_transfer: Any | None = None, thermal_work: Any | None = None, species_binding: CondensationSpeciesBinding | None = None) -> tuple[Any, Any]
```

Validate a direct call, then enqueue its prepared device sequence.

Preparation performs the complete read-only public preflight and resolves
all fallback storage. Only after preparation succeeds does this function
enqueue the four equal condensation substeps. The return values and
caller-owned buffer identities are unchanged from the direct condensation
API.

**Parameters**

- `particles` (`Any`) — GPU-resident particle data container.
- `gas` (`Any`) — GPU-resident gas data container.
- `temperature` (`float | Any | None`) — Direct scalar or per-box Warp temperature [K], or ``None`` when ``environment`` supplies it.
- `pressure` (`float | Any | None`) — Direct scalar or per-box Warp pressure [Pa], or ``None`` when ``environment`` supplies it.
- `time_step` (`float`) — Total condensation duration [s].
- `surface_tension` (`Any | None`) (default: `None`) — Optional per-species surface tension [N/m].
- `mass_accommodation` (`Any | None`) (default: `None`) — Optional per-species accommodation coefficients.
- `diffusion_coefficient_vapor` (`Any | None`) (default: `None`) — Optional vapor diffusion coefficients [m²/s].
- `mass_transfer` (`Any | None`) (default: `None`) — Optional caller-owned total-transfer output buffer.
- `environment` (`Any | None`) (default: `None`) — Optional device-resident environment container.
- `thermodynamics` (`ThermodynamicsConfig | Any | None`) (default: `None`) — Required thermodynamic vapor-pressure configuration.
- `activity_surface` (`CondensationActivitySurfaceConfig | Any | None`) (default: `None`) — Optional activity and surface-tension sidecar.
- `scratch_buffers` (`CondensationScratchBuffers | Any | None`) (default: `None`) — Optional reusable condensation scratch sidecar.
- `latent_heat` (`Any | None`) (default: `None`) — Optional per-species latent heat [J/kg].
- `energy_transfer` (`Any | None`) (default: `None`) — Optional caller-owned energy output [J].
- `thermal_work` (`Any | None`) (default: `None`) — Optional validated but unused full-gas thermal sidecar.
- `species_binding` (`CondensationSpeciesBinding | None`) (default: `None`) — Required explicit binding for discrete/resolved state; prepare with ``prepare_condensation_species_map`` from ``particula.gpu.species_mapping``. Material surface/activity arrays use particle order. Thermodynamics, accommodation, diffusion, heat and energy use full gas order. Transfer outputs remain particle ordered kg per particle; native energy is extensive J per gas lane.

**Returns**

- (`Any`) — A tuple ``(particles, total_mass_transfer)``. The particle container is
- (`Any`) — returned by identity, and supplied total-transfer storage is also
- (`tuple[Any, Any]`) — returned by identity.

**Raises**

- `ValueError` — If preflight rejects the containers, inputs, sidecars, or
output ownership, or if a post-launch runtime validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L3186-L3270)

## particula.gpu.kernels.condensation.validate_condensation_activity_surface_config

*function*

```python
def validate_condensation_activity_surface_config(activity_surface: CondensationActivitySurfaceConfig | Any, n_species: int, device: Any, gas_molar_mass: Any, caller_name: str, *, species_binding: CondensationSpeciesBinding | None = None) -> CondensationActivitySurfaceConfig
```

Validate and return the identical activity/surface sidecar.

**Parameters**

- `activity_surface` (`CondensationActivitySurfaceConfig | Any`) — Candidate frozen sidecar to validate.
- `n_species` (`int`) — Number of particle material species.
- `device` (`Any`) — Active Warp device for all sidecar arrays.
- `gas_molar_mass` (`Any`) — Ordered caller-owned gas molar-mass array.
- `caller_name` (`str`) — Entry-point name used in validation errors.
- `species_binding` (`CondensationSpeciesBinding | None`) (default: `None`) — Native full-gas mapping; declared pairs, including skipped pairs, must match their material reference molar masses.

**Returns**

- (`CondensationActivitySurfaceConfig`) — The original validated ``CondensationActivitySurfaceConfig`` object.

**Raises**

- `ValueError` — If the sidecar identity, selectors, arrays, domains, or
ordered molar-mass reference violates the contract.

> **Notes**
>
> This read-only preflight runs before allocation and launch. A malformed
> sidecar therefore leaves particle, gas, environment, and supplied
> output buffers unchanged; it never falls back to legacy physics.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L387-L490)

## particula.gpu.kernels.condensation.validate_condensation_scratch_buffers

*function*

```python
def validate_condensation_scratch_buffers(candidate: CondensationScratchBuffers | Any, dimensions: tuple[int, int, int], device: Any, caller_name: str, gas_species_count: int | None = None) -> CondensationScratchBuffers
```

Validate scratch metadata without allocation, reads, or mutation.

This atomic metadata-only gate requires the exact
``CondensationScratchBuffers`` type and validates every supplied field's
stable shape, ``wp.float64`` dtype, and active device. It neither reads nor
writes supplied arrays, allocates fallback buffers, launches Warp work, or
synchronizes.

**Parameters**

- `candidate` (`CondensationScratchBuffers | Any`) — Scratch-buffer sidecar to validate.
- `dimensions` (`tuple[int, int, int]`) — Expected ``(n_boxes, n_particles, n_species)`` dimensions.
- `device` (`Any`) — Active Warp device required for supplied fields.
- `caller_name` (`str`) — Entry-point name included in validation errors.
- `gas_species_count` (`int | None`) (default: `None`) — Full gas width for reduction buffers. Omitting it validates an explicitly equal-width scratch layout.

**Returns**

- (`CondensationScratchBuffers`) — The original validated ``CondensationScratchBuffers`` object.

**Raises**

- `ValueError` — If ``candidate`` is not the exact sidecar type or a
supplied field has incompatible shape, dtype, or device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/condensation.py#L309-L366)
