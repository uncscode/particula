# particula.gpu.kernels.wall_loss

*module*

Apply bounded direct GPU wall loss to fixed particle slots.

This concrete direct-kernel boundary supports particle-resolved neutral and
charged wall loss. In charged mode, only nonzero charge composes the private
image-charge enhancement
image-charge and field drift field-drift coefficient helpers; zero-charge slots
retain
the exact neutral coefficient and random-number path. A charged rectangular
field is caller-owned same-device ``wp.float64`` storage with shape ``(3,)``.
Only the charged rectangular removal kernel reads it, and it is never mutated.
The boundary retains frozen read-only preflight and sequential per-box random
number generation for eligible fixed slots.
An omitted RNG sidecar is private and seeded for each successful positive-time
call. A supplied ``uint32`` Warp sidecar remains caller-owned, advances in
place only for eligible slots, and resets only when explicitly requested.

Private prepared calls retain validated arrays and frozen selection after setup.
Their enqueue dispatches only retained device work, without validation,
allocation, readback, transfer, synchronization, lookup, or RNG reset.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1-L1)

## particula.gpu.kernels.wall_loss._BOLTZMANN_CONSTANT

*attribute* · *module attribute*

```python
_BOLTZMANN_CONSTANT = wp.constant(wp.float64(BOLTZMANN_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L76-L76)

## particula.gpu.kernels.wall_loss._ELECTRIC_PERMITTIVITY

*attribute* · *module attribute*

```python
_ELECTRIC_PERMITTIVITY = wp.constant(wp.float64(ELECTRIC_PERMITTIVITY))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L83-L83)

## particula.gpu.kernels.wall_loss._ELEMENTARY_CHARGE_VALUE

*attribute* · *module attribute*

```python
_ELEMENTARY_CHARGE_VALUE = wp.constant(wp.float64(ELEMENTARY_CHARGE_VALUE))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L82-L82)

## particula.gpu.kernels.wall_loss._GAS_CONSTANT

*attribute* · *module attribute*

```python
_GAS_CONSTANT = wp.constant(wp.float64(GAS_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L77-L77)

## particula.gpu.kernels.wall_loss._MOLECULAR_WEIGHT_AIR

*attribute* · *module attribute*

```python
_MOLECULAR_WEIGHT_AIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L78-L78)

## particula.gpu.kernels.wall_loss._REF_TEMPERATURE_STP

*attribute* · *module attribute*

```python
_REF_TEMPERATURE_STP = wp.constant(wp.float64(REF_TEMPERATURE_STP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L80-L80)

## particula.gpu.kernels.wall_loss._REF_VISCOSITY_AIR_STP

*attribute* · *module attribute*

```python
_REF_VISCOSITY_AIR_STP = wp.constant(wp.float64(REF_VISCOSITY_AIR_STP))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L79-L79)

## particula.gpu.kernels.wall_loss._SUTHERLAND_CONSTANT

*attribute* · *module attribute*

```python
_SUTHERLAND_CONSTANT = wp.constant(wp.float64(SUTHERLAND_CONSTANT))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L81-L81)

## particula.gpu.kernels.wall_loss.NeutralWallLossConfig

*class* · *dataclass*

```python
class NeutralWallLossConfig
```

Define immutable direct wall-loss geometry and charged inputs.

This concrete-module-only configuration is accepted by the direct validation
boundary, which retains frozen field drift preflight. ``geometry`` selects
spherical
or rectangular SI dimensions, while ``distribution_type`` must remain
``"particle_resolved"``. ``mode`` is ``"neutral"`` or ``"charged"``.
Charged execution composes private image-charge enhancement image and field
drift drift helpers only for
nonzero charges. Import this concrete configuration
from ``particula.gpu.kernels.wall_loss``. A charged rectangular electric
field is caller-owned device storage and is never replaced or mutated.

**Attributes**

- `geometry` (`str`) — Exact chamber geometry, ``"spherical"`` or ``"rectangular"``.
- `wall_eddy_diffusivity` (`float`) — Positive wall eddy diffusivity [m^2 s^-1].
- `chamber_radius` (`float | None`) — Positive spherical chamber radius [m]; required only for spherical geometry.
- `chamber_dimensions` (`tuple[float, float, float] | None`) — Tuple of three positive rectangular chamber dimensions [m]; required only for rectangular geometry.
- `distribution_type` (`str`) — Required ``"particle_resolved"`` representation selector.
- `mode` (`str`) — ``"neutral"`` or ``"charged"`` execution mode. Zero-charge charged slots retain the neutral coefficient and RNG path.
- `wall_potential` (`float`) — Finite signed wall potential [V].
- `wall_electric_field` (`Any`) — Finite signed spherical scalar field [V m^-1], or caller-owned same-device charged rectangular ``wp.float64`` vector with shape ``(3,)`` [V m^-1]. A rectangular vector is invalid in neutral mode and is neither replaced nor mutated in charged mode.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L86-L126)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.chamber_dimensions

*attribute* · *class attribute* · *instance attribute*

```python
chamber_dimensions: tuple[float, float, float] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L122-L122)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.chamber_radius

*attribute* · *class attribute* · *instance attribute*

```python
chamber_radius: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L121-L121)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.distribution_type

*attribute* · *class attribute* · *instance attribute*

```python
distribution_type: str = 'particle_resolved'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L123-L123)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.geometry

*attribute* · *instance attribute*

```python
geometry: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L119-L119)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.mode

*attribute* · *class attribute* · *instance attribute*

```python
mode: str = 'neutral'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L124-L124)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L120-L120)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.wall_electric_field

*attribute* · *class attribute* · *instance attribute*

```python
wall_electric_field: Any = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L126-L126)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.wall_potential

*attribute* · *class attribute* · *instance attribute*

```python
wall_potential: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L125-L125)

### particula.gpu.kernels.wall_loss.NeutralWallLossConfig.__init__

*method*

```python
def __init__(geometry: str, wall_eddy_diffusivity: float, chamber_radius: float | None = None, chamber_dimensions: tuple[float, float, float] | None = None, distribution_type: str = 'particle_resolved', mode: str = 'neutral', wall_potential: float = 0.0, wall_electric_field: Any = 0.0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L0-L0)

## particula.gpu.kernels.wall_loss._PreparedWallLossCall

*class* · *dataclass*

```python
class _PreparedWallLossCall
```

Privately freeze all setup state required by one wall-loss enqueue.

Launch arguments and device retain exact references established during
preparation, so later container-attribute rebinding cannot redirect a
writer. Validation, normalization, RNG setup, and allocation precede
enqueue; enqueue never rereads container attributes, validates payloads,
allocates, transfers, synchronizes, looks up resources, or resets RNG state.
The no-op path is write-free; rollback is not promised after a writer
launches.

**Attributes**

- `particles` (`Any`) — Caller-owned particle container returned by the operation.
- `launch_kernel` (`Any | None`) — Selected native kernel, or ``None`` for a no-op.
- `launch_dim` (`Any`) — Frozen launch dimensions for the selected kernel.
- `launch_inputs` (`tuple[Any, ...]`) — Frozen kernel arguments, including caller-owned fields.
- `device` (`Any`) — Frozen Warp device used for the selected launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1313-L1337)

### particula.gpu.kernels.wall_loss._PreparedWallLossCall.device

*attribute* · *instance attribute*

```python
device: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1337-L1337)

### particula.gpu.kernels.wall_loss._PreparedWallLossCall.launch_dim

*attribute* · *instance attribute*

```python
launch_dim: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1335-L1335)

### particula.gpu.kernels.wall_loss._PreparedWallLossCall.launch_inputs

*attribute* · *instance attribute*

```python
launch_inputs: tuple[Any, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1336-L1336)

### particula.gpu.kernels.wall_loss._PreparedWallLossCall.launch_kernel

*attribute* · *instance attribute*

```python
launch_kernel: Any | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1334-L1334)

### particula.gpu.kernels.wall_loss._PreparedWallLossCall.particles

*attribute* · *instance attribute*

```python
particles: Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1333-L1333)

### particula.gpu.kernels.wall_loss._PreparedWallLossCall.__init__

*method*

```python
def __init__(particles: Any, launch_kernel: Any | None, launch_dim: Any, launch_inputs: tuple[Any, ...], device: Any) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L0-L0)

## particula.gpu.kernels.wall_loss._cast_float32_to_float64

*function*

```python
def _cast_float32_to_float64(source: wp.array(dtype=wp.float32), destination: wp.array(dtype=wp.float64)) -> None
```

Copy a private execution environment buffer into float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L483-L491)

## particula.gpu.kernels.wall_loss._charged_rectangular_wall_loss_remove

*function*

```python
def _charged_rectangular_wall_loss_remove(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.array(dtype=wp.float64), n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32) -> None
```

Run charged rectangular wall loss for the direct all-box boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1171-L1214)

## particula.gpu.kernels.wall_loss._charged_rectangular_wall_loss_remove_guarded

*function*

```python
def _charged_rectangular_wall_loss_remove_guarded(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.array(dtype=wp.float64), n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1217-L1263)

## particula.gpu.kernels.wall_loss._charged_rectangular_wall_loss_remove_impl

*function*

```python
def _charged_rectangular_wall_loss_remove_impl(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.array(dtype=wp.float64), n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, box: wp.int32) -> None
```

Select and clear charged rectangular wall-loss slots in one launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L763-L896)

## particula.gpu.kernels.wall_loss._charged_rectangular_wall_loss_remove_selected

*function*

```python
def _charged_rectangular_wall_loss_remove_selected(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.array(dtype=wp.float64), n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, selected_boxes: wp.array(dtype=wp.int32)) -> None
```

Run charged rectangular wall loss for selected logical-box lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1266-L1310)

## particula.gpu.kernels.wall_loss._charged_spherical_wall_loss_remove

*function*

```python
def _charged_spherical_wall_loss_remove(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.float64, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32) -> None
```

Run charged spherical wall loss for the direct all-box boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1041-L1080)

## particula.gpu.kernels.wall_loss._charged_spherical_wall_loss_remove_guarded

*function*

```python
def _charged_spherical_wall_loss_remove_guarded(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.float64, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1083-L1125)

## particula.gpu.kernels.wall_loss._charged_spherical_wall_loss_remove_impl

*function*

```python
def _charged_spherical_wall_loss_remove_impl(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.float64, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, box: wp.int32) -> None
```

Select and clear charged spherical wall-loss slots in one launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L635-L760)

## particula.gpu.kernels.wall_loss._charged_spherical_wall_loss_remove_selected

*function*

```python
def _charged_spherical_wall_loss_remove_selected(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, wall_potential: wp.float64, wall_electric_field: wp.float64, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, selected_boxes: wp.array(dtype=wp.int32)) -> None
```

Run charged spherical wall loss for selected logical-box lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1128-L1168)

## particula.gpu.kernels.wall_loss._enqueue_prepared_wall_loss_call

*function*

```python
def _enqueue_prepared_wall_loss_call(prepared: _PreparedWallLossCall) -> Any
```

Issue only the frozen wall-loss launch, without setup or validation.

This private enqueue boundary performs no validation, allocation,
normalization, readback, transfer, synchronization, lookup, rebinding, or
host RNG initialization. The frozen kernel reset flag is still honored
on every launch. Persistent replay requires supplied initialized state and
``initialize_rng=False`` during preparation. The selected no-op is
write-free; rollback is not promised after a writer launches.

**Parameters**

- `prepared` (`_PreparedWallLossCall`) — Validated call record produced by ``_prepare_wall_loss_step_gpu``.

**Returns**

- (`Any`) — The pinned particle container.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1551-L1575)

## particula.gpu.kernels.wall_loss._get_required_field

*function*

```python
def _get_required_field(particles: Any, field: str) -> Any
```

Return one required particle field with a stable schema error.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L299-L304)

## particula.gpu.kernels.wall_loss._initialize_rng_states

*function*

```python
def _initialize_rng_states(seed: Any, rng_states: Any) -> None
```

Initialize or reset one execution RNG state per particle box.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L517-L522)

## particula.gpu.kernels.wall_loss._normalize_execution_environment_array

*function*

```python
def _normalize_execution_environment_array(values: Any, n_boxes: int) -> Any
```

Return a float64 execution buffer without replacing caller input.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L494-L505)

## particula.gpu.kernels.wall_loss._prepare_selected_wall_loss_call

*function*

```python
def _prepare_selected_wall_loss_call(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float | Any, *, config: NeutralWallLossConfig, rng_seed: int, rng_states: Any, selected_boxes: Any, environment: Any | None = None) -> _PreparedWallLossCall
```

Prepare a frozen wall-loss launch for selected logical box lanes.

**Parameters**

- `particles` (`Any`) — Caller-owned fixed-shape Warp particle data.
- `temperature` (`float | Any | None`) — Scalar or per-box temperature [K].
- `pressure` (`float | Any | None`) — Scalar or per-box pressure [Pa].
- `time_step` (`float | Any`) — Finite nonnegative duration [s].
- `config` (`NeutralWallLossConfig`) — Concrete geometry and wall-loss configuration.
- `rng_seed` (`int`) — Unsigned 32-bit seed forwarded to the device kernel.
- `rng_states` (`Any`) — Caller-owned per-box RNG state.
- `selected_boxes` (`Any`) — Nonempty same-device int32 array of logical box lanes.
- `environment` (`Any | None`) (default: `None`) — Optional explicit environment source.

**Returns**

- (`_PreparedWallLossCall`) — A frozen call record containing the selected launch and its inputs.

**Raises**

- `TypeError` — If a scalar configuration or RNG argument has an invalid
type.
- `ValueError` — If selected lanes, particle, environment, configuration,
or RNG inputs violate the direct-kernel contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1628-L1784)

## particula.gpu.kernels.wall_loss._prepare_wall_loss_step_gpu

*function*

```python
def _prepare_wall_loss_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float | Any, *, config: NeutralWallLossConfig, rng_seed: int = 0, rng_states: Any | None = None, initialize_rng: bool = False, environment: Any | None = None) -> _PreparedWallLossCall
```

Privately prepare direct neutral or charged wall loss for fixed slots.

The configuration supports particle-resolved neutral and charged wall loss.
Charged nonzero slots compose private image-charge enhancement image
enhancement and field drift signed
Field drift with their neutral coefficient; zero-charge charged slots use
the exact neutral coefficient and RNG path. SI inputs are wall eddy
diffusivity [m^2 s^-1],
geometry dimensions [m], potential [V], electric field [V m^-1],
temperature [K], pressure [Pa], and ``time_step`` [s]. A charged rectangular
electric-field vector is caller-owned read-only storage; only charged
rectangular execution reads its three lanes.
The matching enqueue helper evaluates neutral coefficients for usable slots
in positive-time calls after frozen field drift preflight. It applies
survival
probability ``exp(-k * time_step)`` and clears every mass lane,
concentration, and charge for removed slots. Zero time is a post-preflight
write-free no-op.

Native tagged particles must be resolved raw counts, with positive box
volumes and representable material geometry. Unequal positive counts are
supported: each surviving representative retains its entire count.
Native writers must not overlap each other or protected input storage.
Explicit particle-resolved metadata is required; untagged/PMF/PDF reject.

Omitted ``rng_states`` uses a private per-call sidecar initialized from
``rng_seed`` after successful positive-time preflight. A supplied
same-device ``uint32`` sidecar with shape ``(n_boxes,)`` remains
caller-owned, advances in place only for eligible slots, and resets only
when ``initialize_rng=True``. Eligible slots have positive concentration,
usable positive mass and derived transport properties, and a finite positive
wall-loss coefficient. An infinite positive coefficient causes deterministic
removal without consuming RNG state. Zero time and pre-launch validation
failures leave supplied state unchanged.
Rollback is not promised after a mutation kernel has launched.

**Parameters**

- `particles` (`Any`) — Caller-owned fixed-shape ``WarpParticleData``-like object.
- `temperature` (`float | Any | None`) — Scalar or per-box Warp temperature [K].
- `pressure` (`float | Any | None`) — Scalar or per-box Warp pressure [Pa].
- `time_step` (`float | Any`) — Finite nonnegative duration [s].
- `config` (`NeutralWallLossConfig`) — Exact concrete geometry, representation, and charged configuration. Rectangular fields remain caller-owned storage.
- `rng_seed` (`int`) (default: `0`) — Unsigned 32-bit seed for private state or an explicit reset.
- `rng_states` (`Any | None`) (default: `None`) — Optional caller-owned same-device ``uint32`` Warp array with shape ``(n_boxes,)``. The array is mutated in place only by eligible positive-time work.
- `initialize_rng` (`bool`) (default: `False`) — Whether to reset supplied ``rng_states`` from ``rng_seed`` before a successful positive-time call.
- `environment` (`Any | None`) (default: `None`) — Optional explicit Warp environment source. Cannot be combined with direct ``temperature`` or ``pressure`` inputs.

**Returns**

- (`_PreparedWallLossCall`) — A private frozen call record for the matching enqueue helper.

**Raises**

- `TypeError` — If the configuration, scalar charged input, time step, or
RNG metadata uses an unsupported type.
- `ValueError` — If configuration, particle, time-step, environment, or RNG
values, shapes, dtypes, or devices violate the direct contract,
including an invalid charged field or combining ``environment``
with direct environment inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1340-L1548)

## particula.gpu.kernels.wall_loss._require_finite_real

*function*

```python
def _require_finite_real(value: Any, name: str) -> float
```

Validate an exact scalar finite signed configuration value.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L213-L225)

## particula.gpu.kernels.wall_loss._require_positive_real

*function*

```python
def _require_positive_real(value: Any, name: str) -> float
```

Validate an exact scalar positive finite physical configuration value.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L198-L210)

## particula.gpu.kernels.wall_loss._scan_finite_1d

*function*

```python
def _scan_finite_1d(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record non-finite entries in one-dimensional signed metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L162-L170)

## particula.gpu.kernels.wall_loss._scan_finite_2d

*function*

```python
def _scan_finite_2d(values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record non-finite entries in two-dimensional metadata.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L151-L159)

## particula.gpu.kernels.wall_loss._scan_nonnegative_finite_1d

*function*

```python
def _scan_nonnegative_finite_1d(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record invalid entries in one-dimensional nonnegative data.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L129-L137)

## particula.gpu.kernels.wall_loss._scan_nonnegative_finite_2d

*function*

```python
def _scan_nonnegative_finite_2d(values: wp.array2d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record invalid entries in two-dimensional nonnegative data.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L140-L148)

## particula.gpu.kernels.wall_loss._scan_nonnegative_finite_3d

*function*

```python
def _scan_nonnegative_finite_3d(values: wp.array3d(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record invalid entries in three-dimensional nonnegative data.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L173-L184)

## particula.gpu.kernels.wall_loss._scan_positive_finite_1d

*function*

```python
def _scan_positive_finite_1d(values: wp.array(dtype=wp.float64), invalid: wp.array(dtype=wp.int32)) -> None
```

Record invalid entries in one-dimensional positive data.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L187-L195)

## particula.gpu.kernels.wall_loss._should_remove_for_survival_draw

*function*

```python
def _should_remove_for_survival_draw(draw: Any, survival_probability: wp.float64) -> bool
```

Return whether one eligible slot's survival draw selects removal.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L508-L514)

## particula.gpu.kernels.wall_loss._validate_array_schema

*function*

```python
def _validate_array_schema(values: Any, name: str, rank: int, shape: tuple[int, ...] | None = None, device: Any | None = None) -> Any
```

Validate exact Warp metadata without reading or replacing values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L307-L325)

## particula.gpu.kernels.wall_loss._validate_charged_field_form

*function*

```python
def _validate_charged_field_form(config: NeutralWallLossConfig) -> None
```

Validate scalar-versus-vector field form without reading Warp fields.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L261-L271)

## particula.gpu.kernels.wall_loss._validate_charged_rectangular_field

*function*

```python
def _validate_charged_rectangular_field(config: NeutralWallLossConfig, device: Any) -> None
```

Validate and scan the caller-owned charged rectangular field.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L401-L414)

## particula.gpu.kernels.wall_loss._validate_config

*function*

```python
def _validate_config(config: Any) -> NeutralWallLossConfig
```

Validate configuration without reading Warp fields or particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L274-L296)

## particula.gpu.kernels.wall_loss._validate_geometry

*function*

```python
def _validate_geometry(config: NeutralWallLossConfig) -> None
```

Validate geometry and its scalar payload without particle access.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L228-L258)

## particula.gpu.kernels.wall_loss._validate_particle_schema

*function*

```python
def _validate_particle_schema(particles: Any) -> tuple[int, Any]
```

Validate fixed particle schema and return its box count and device.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L363-L398)

## particula.gpu.kernels.wall_loss._validate_particle_values

*function*

```python
def _validate_particle_values(particles: Any) -> None
```

Validate particle values after configuration-dependent schema checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L417-L426)

## particula.gpu.kernels.wall_loss._validate_particles

*function*

```python
def _validate_particles(particles: Any) -> tuple[int, Any]
```

Validate fixed particle metadata and values in the declared order.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L429-L433)

## particula.gpu.kernels.wall_loss._validate_rng

*function*

```python
def _validate_rng(rng_seed: Any, rng_states: Any, initialize_rng: Any, n_boxes: int, device: Any) -> None
```

Validate deferred RNG metadata without scanning or consuming state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L455-L480)

## particula.gpu.kernels.wall_loss._validate_time_step

*function*

```python
def _validate_time_step(time_step: Any) -> float
```

Validate a finite nonnegative time step in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L436-L452)

## particula.gpu.kernels.wall_loss._validate_values

*function*

```python
def _validate_values(values: Any, name: str, positive: bool = False, finite_only: bool = False) -> None
```

Run a device-resident finite-domain scan without host materialization.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L328-L360)

## particula.gpu.kernels.wall_loss._wall_loss_remove

*function*

```python
def _wall_loss_remove(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, geometry_mode: wp.int32, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32) -> None
```

Run neutral wall loss for the direct all-box boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L899-L942)

## particula.gpu.kernels.wall_loss._wall_loss_remove_guarded

*function*

```python
def _wall_loss_remove_guarded(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, geometry_mode: wp.int32, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, failure: wp.array(dtype=wp.int32)) -> None
```

Skip this resident writer after a latched simulation failure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L945-L991)

## particula.gpu.kernels.wall_loss._wall_loss_remove_impl

*function*

```python
def _wall_loss_remove_impl(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, geometry_mode: wp.int32, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, box: wp.int32) -> None
```

Select and clear neutral wall-loss slots in one device launch.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L525-L632)

## particula.gpu.kernels.wall_loss._wall_loss_remove_selected

*function*

```python
def _wall_loss_remove_selected(masses: wp.array3d(dtype=wp.float64), concentration: wp.array2d(dtype=wp.float64), charge: wp.array2d(dtype=wp.float64), density: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64), time_step: wp.float64, wall_eddy_diffusivity: wp.float64, chamber_radius: wp.float64, chamber_length: wp.float64, chamber_width: wp.float64, chamber_height: wp.float64, geometry_mode: wp.int32, n_particles: wp.int32, n_species: wp.int32, rng_states: wp.array(dtype=wp.uint32), rng_seed: wp.int32, initialize_rng: wp.int32, selected_boxes: wp.array(dtype=wp.int32)) -> None
```

Run neutral wall loss once for the selected logical-box lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L994-L1038)

## particula.gpu.kernels.wall_loss.wall_loss_selected_boxes_step_gpu

*function*

```python
def wall_loss_selected_boxes_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float | Any, *, config: NeutralWallLossConfig, rng_seed: int, rng_states: Any, selected_boxes: Any, environment: Any | None = None) -> Any
```

Privately dispatch one validated wall-loss writer over selected boxes.

This resident-only seam is deliberately not re-exported.  Its index buffer
is prevalidated by the caller and maps one writer thread to each selected
logical box, avoiding repeated direct-boundary preflight and launches.

**Parameters**

- `particles` (`Any`) — Caller-owned fixed-shape Warp particle data.
- `temperature` (`float | Any | None`) — Scalar or per-box temperature [K].
- `pressure` (`float | Any | None`) — Scalar or per-box pressure [Pa].
- `time_step` (`float | Any`) — Finite nonnegative duration [s].
- `config` (`NeutralWallLossConfig`) — Concrete geometry and wall-loss configuration.
- `rng_seed` (`int`) — Unsigned 32-bit seed forwarded to the device kernel.
- `rng_states` (`Any`) — Caller-owned per-box RNG state.
- `selected_boxes` (`Any`) — Nonempty same-device int32 array of logical box lanes.
- `environment` (`Any | None`) (default: `None`) — Optional explicit environment source.

**Returns**

- (`Any`) — The identical caller-owned particle container.

**Raises**

- `TypeError` — If a scalar configuration or RNG argument has an invalid
type.
- `ValueError` — If selected lanes, particle, environment, configuration,
or RNG inputs violate the direct-kernel contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1787-L1969)

## particula.gpu.kernels.wall_loss.wall_loss_step_gpu

*function*

```python
def wall_loss_step_gpu(particles: Any, temperature: float | Any | None, pressure: float | Any | None, time_step: float | Any, *, config: NeutralWallLossConfig, rng_seed: int = 0, rng_states: Any | None = None, initialize_rng: bool = False, environment: Any | None = None) -> Any
```

Prepare and enqueue one direct neutral or charged wall-loss call.

**Parameters**

- `particles` (`Any`) — Caller-owned fixed-shape Warp particle data.
- `temperature` (`float | Any | None`) — Scalar or per-box temperature [K].
- `pressure` (`float | Any | None`) — Scalar or per-box pressure [Pa].
- `time_step` (`float | Any`) — Finite nonnegative duration [s].
- `config` (`NeutralWallLossConfig`) — Concrete geometry and wall-loss configuration.
- `rng_seed` (`int`) (default: `0`) — Unsigned 32-bit seed for RNG initialization.
- `rng_states` (`Any | None`) (default: `None`) — Optional caller-owned per-box RNG state.
- `initialize_rng` (`bool`) (default: `False`) — Whether to reset supplied RNG state before execution.
- `environment` (`Any | None`) (default: `None`) — Optional explicit environment source.

**Returns**

- (`Any`) — The identical caller-owned particle container after the asynchronous
- (`Any`) — wall-loss launch, or immediately for a valid no-op.

**Raises**

- `TypeError` — If a scalar configuration or RNG argument has an invalid
type.
- `ValueError` — If particle, environment, configuration, or RNG inputs do
not satisfy the direct-kernel contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/wall_loss.py#L1578-L1625)
