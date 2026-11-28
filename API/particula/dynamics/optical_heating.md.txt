# particula.dynamics.optical_heating

*module*

Optically thin heating and one-way water evaporation of wet spheres.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L1-L1)

## particula.dynamics.optical_heating._HEAT_CHANNELS

*attribute* · *module attribute*

```python
_HEAT_CHANNELS = ('absorbed', 'conduction', 'radiation', 'latent', 'sensible')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L16-L16)

## particula.dynamics.optical_heating._MAX_K

*attribute* · *module attribute*

```python
_MAX_K = 373.15
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L18-L18)

## particula.dynamics.optical_heating._MIN_K

*attribute* · *module attribute*

```python
_MIN_K = 273.15
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L17-L17)

## particula.dynamics.optical_heating._WATER

*attribute* · *module attribute*

```python
_WATER = WaterBuckStrategy()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L19-L19)

## particula.dynamics.optical_heating.HeatingPhysics

*class* · *dataclass*

```python
class HeatingPhysics
```

Prescribed continuum transport and liquid-solution properties.

Conductivity is W/(m K), water diffusivity is m²/s, specific heat
capacities are J/(kg K), latent heat is J/kg, and radiation_temperature
is K. Kappa and grey infrared emissivity are dimensionless.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L22-L58)

### particula.dynamics.optical_heating.HeatingPhysics.conductivity

*attribute* · *class attribute* · *instance attribute*

```python
conductivity: float = 0.026
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L32-L32)

### particula.dynamics.optical_heating.HeatingPhysics.diffusivity

*attribute* · *class attribute* · *instance attribute*

```python
diffusivity: float = 2.5e-05
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L33-L33)

### particula.dynamics.optical_heating.HeatingPhysics.emissivity

*attribute* · *class attribute* · *instance attribute*

```python
emissivity: float = 0.95
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L37-L37)

### particula.dynamics.optical_heating.HeatingPhysics.kappa

*attribute* · *class attribute* · *instance attribute*

```python
kappa: float = 0.3
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L31-L31)

### particula.dynamics.optical_heating.HeatingPhysics.latent_heat

*attribute* · *class attribute* · *instance attribute*

```python
latent_heat: float = 2440000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L36-L36)

### particula.dynamics.optical_heating.HeatingPhysics.radiation_temperature

*attribute* · *class attribute* · *instance attribute*

```python
radiation_temperature: float = 298.15
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L38-L38)

### particula.dynamics.optical_heating.HeatingPhysics.solute_heat_capacity

*attribute* · *class attribute* · *instance attribute*

```python
solute_heat_capacity: float = 1500.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L35-L35)

### particula.dynamics.optical_heating.HeatingPhysics.water_heat_capacity

*attribute* · *class attribute* · *instance attribute*

```python
water_heat_capacity: float = 4180.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L34-L34)

### particula.dynamics.optical_heating.HeatingPhysics.__init__

*method*

```python
def __init__(kappa: float = 0.3, conductivity: float = 0.026, diffusivity: float = 2.5e-05, water_heat_capacity: float = 4180.0, solute_heat_capacity: float = 1500.0, latent_heat: float = 2440000.0, emissivity: float = 0.95, radiation_temperature: float = 298.15) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L0-L0)

### particula.dynamics.optical_heating.HeatingPhysics.__post_init__

*method*

```python
def __post_init__()
```

Validate physical constants before evaluating a temperature root.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L40-L58)

## particula.dynamics.optical_heating.OpticalHeating

*class*

```python
class OpticalHeating(BoxProcess)
```

Bases: `particula.box_process.BoxProcess`

Heat a particle-resolved water/solute population with irradiance.

The irradiances are band-integrated W/m², one value per configured
wavelength. Absorption is recomputed from live wet sizes and composition
at the start of the process. Use attach_fields() before preparing the
sequence. Each box exchanges heat with its fixed-temperature gas and a
grey radiation bath. Evaporated water enters the named gas species.

The liquid, spherical continuum model neglects curvature in the vapor
balance. Its transport coefficients are prescribed. Complete optical
heat exchange and mass loss are implicit in particle temperature.
Geometry, irradiance, heat capacity and surrounding gas are evaluated
at the beginning of each cycle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L130-L334)

### particula.dynamics.optical_heating.OpticalHeating._namespace

*attribute* · *class attribute* · *instance attribute*

```python
_namespace = 'optical_heating'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L146-L146)

### particula.dynamics.optical_heating.OpticalHeating.energy_fields

*attribute* · *instance attribute*

```python
energy_fields = tuple(({name}.{key}_energy for key in _HEAT_CHANNELS))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L185-L187)

### particula.dynamics.optical_heating.OpticalHeating.gas_water

*attribute* · *instance attribute*

```python
gas_water = gas_water
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L181-L181)

### particula.dynamics.optical_heating.OpticalHeating.irradiance

*attribute* · *instance attribute*

```python
irradiance = tuple((float(value) for value in flux))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L180-L180)

### particula.dynamics.optical_heating.OpticalHeating.optics

*attribute* · *instance attribute*

```python
optics = optics
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L161-L161)

### particula.dynamics.optical_heating.OpticalHeating.physics

*attribute* · *instance attribute*

```python
physics = HeatingPhysics() if physics is None else physics
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L162-L162)

### particula.dynamics.optical_heating.OpticalHeating.power_field

*attribute* · *instance attribute*

```python
power_field = {name}.absorbed_power
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L184-L184)

### particula.dynamics.optical_heating.OpticalHeating.temperature_field

*attribute* · *instance attribute*

```python
temperature_field = {name}.temperature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L183-L183)

### particula.dynamics.optical_heating.OpticalHeating.water_material

*attribute* · *instance attribute*

```python
water_material = water_material
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L182-L182)

### particula.dynamics.optical_heating.OpticalHeating.extension_id

*attribute* · *instance attribute*

```python
extension_id = extension_id or name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L100-L100)

### particula.dynamics.optical_heating.OpticalHeating.name

*attribute* · *instance attribute*

```python
name = name
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L95-L95)

### particula.dynamics.optical_heating.OpticalHeating.parameters

*attribute* · *instance attribute*

```python
parameters = parameters
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L105-L105)

### particula.dynamics.optical_heating.OpticalHeating.reads

*attribute* · *instance attribute*

```python
reads = tuple(reads)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L96-L96)

### particula.dynamics.optical_heating.OpticalHeating.representations

*attribute* · *instance attribute*

```python
representations = tuple(representations)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L99-L99)

### particula.dynamics.optical_heating.OpticalHeating.stochastic

*attribute* · *instance attribute*

```python
stochastic = stochastic
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L98-L98)

### particula.dynamics.optical_heating.OpticalHeating.version

*attribute* · *instance attribute*

```python
version = version
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L106-L106)

### particula.dynamics.optical_heating.OpticalHeating.writes

*attribute* · *instance attribute*

```python
writes = tuple(writes)
```

Inherited from: `particula.box_process.BoxProcess`

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L97-L97)

### particula.dynamics.optical_heating.OpticalHeating.__init__

*method*

```python
def __init__(optics, irradiance, *, gas_water, water_material=0, physics=None, name='optical_heating')
```

Bind optical forcing, liquid material index and gas species name.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L148-L211)

### particula.dynamics.optical_heating.OpticalHeating._specs

*method*

```python
def _specs()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L223-L230)

### particula.dynamics.optical_heating.OpticalHeating._validate_representation

*method*

```python
def _validate_representation(aerosol)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L232-L236)

### particula.dynamics.optical_heating.OpticalHeating.advance

*method*

```python
def advance(aerosol, *, time, time_step, context)
```

Advance one box and add represented water and energy inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L287-L334)

### particula.dynamics.optical_heating.OpticalHeating.attach_fields

*method*

```python
def attach_fields(aerosol)
```

Register diagnostic and thermal state on the initial aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L238-L245)

### particula.dynamics.optical_heating.OpticalHeating.configuration

*method*

```python
def configuration()
```

Return the parameters retained in exact process identity.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L213-L221)

### particula.dynamics.optical_heating.OpticalHeating.validate

*method*

```python
def validate(aerosol)
```

Require resolved slots, two materials and mapped vapor.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L247-L285)

### particula.dynamics.optical_heating.OpticalHeating.execute

*method*

```python
def execute(aerosol, time_step, sub_steps=1, *, context=None, time=0.0)
```

Inherited from: `particula.box_process.BoxProcess`

Publish a successful process substep atomically across boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/box_process.py#L115-L186)

### particula.dynamics.optical_heating.OpticalHeating.__or__

*method*

```python
def __or__(other: RunnableABC) -> RunnableSequence
```

Inherited from: `particula.runnable.RunnableABC`

Chain this Runnable with another using the '|' operator.

This method enables an easy way to sequence processes, so they
can be executed in a defined order.

**Parameters**

- `other` (`RunnableABC`) — Another Runnable to append after this one.

**Returns**

- (`RunnableSequence`) — A sequence containing both processes.

**Examples**

```py title="Chaining two processes"
combined_process = process1 | process2
final_aerosol = combined_process.execute(aerosol, time_step=1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L272-L293)

### particula.dynamics.optical_heating.OpticalHeating._advance_box

*method*

```python
def _advance_box(aerosol, time_step)
```

Inherited from: `particula.runnable.RunnableABC`

Dispatch one staged interval through the native extension seam.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L171-L173)

### particula.dynamics.optical_heating.OpticalHeating._box_worker

*method*

```python
def _box_worker(box, boxes, transaction)
```

Inherited from: `particula.runnable.RunnableABC`

Create isolated hook state for a single box.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L159-L161)

### particula.dynamics.optical_heating.OpticalHeating._execute_box

*method*

```python
def _execute_box(aerosol, time_step, sub_steps=1)
```

Inherited from: `particula.runnable.RunnableABC`

Execute the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L175-L177)

### particula.dynamics.optical_heating.OpticalHeating._is_stochastic

*method*

```python
def _is_stochastic(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Report whether this process requires independent box streams.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L155-L157)

### particula.dynamics.optical_heating.OpticalHeating._prepare_commit

*method*

```python
def _prepare_commit(workers, uses_context)
```

Inherited from: `particula.runnable.RunnableABC`

Prepare a non-failing diagnostics publisher before numeric writes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L163-L165)

### particula.dynamics.optical_heating.OpticalHeating._rate_box

*method*

```python
def _rate_box(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate the single-box numerical building block.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L167-L169)

### particula.dynamics.optical_heating.OpticalHeating._validate_batch

*method*

```python
def _validate_batch(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Validate process-wide controls before staging boxes.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L148-L149)

### particula.dynamics.optical_heating.OpticalHeating._writer_fields

*method*

```python
def _writer_fields(aerosol)
```

Inherited from: `particula.runnable.RunnableABC`

Declare numeric destinations for transactional publication.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L151-L153)

### particula.dynamics.optical_heating.OpticalHeating.prepare

*method*

```python
def prepare(aerosol: Aerosol, *, time_step: float, backend: Backend | None = None, device: str = 'cuda:0', root_seed: int = 0, logical_box_ids=None, process_registry=None, coagulation_work_config=None, coagulation_subcycling=None, time: float = 0.0) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Upload native state once and prepare supported resident GPU physics.

**Parameters**

- `aerosol` (`Aerosol`) — Initial native particle, gas and environment state.
- `time_step` (`float`) — Fixed duration of each complete process cycle [s].
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default; CPU execution uses execute().
- `device` (`str`) (default: `'cuda:0'`) — Explicit Warp device, default CUDA device zero.
- `root_seed` (`int`) (default: `0`) — Root seed for persistent independent process/box streams.
- `logical_box_ids` (default: `None`) — Optional stable identities in physical box order.
- `process_registry` (default: `None`) — Exact-type custom GPU adapter registrations.
- `coagulation_work_config` (default: `None`) — Optional coagulation resource budgets.
- `coagulation_subcycling` (default: `None`) — Optional bounded adaptive policy.
- `time` (`float`) (default: `0.0`) — Absolute initial simulation time [s].

**Returns**

- (`PreparedSimulation`) — An owned PreparedSimulation. Use its context manager or close().
- (`PreparedSimulation`) — snapshot() returns detached CPU state for analysis and plotting.

**Raises**

- `ValueError` — If process physics or state cannot be represented.
- `RuntimeError` — If the requested runtime/device is unavailable.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L179-L230)

### particula.dynamics.optical_heating.OpticalHeating.rate

*method*

```python
def rate(aerosol: Aerosol) -> Any
```

Inherited from: `particula.runnable.RunnableABC`

Evaluate detached box hooks and retain a leading box axis.

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L72-L87)

### particula.dynamics.optical_heating.OpticalHeating.resume

*method*

```python
def resume(checkpoint: SimulationCheckpoint, *, backend: Backend | None = None, device: str | None = None, process_registry=None, retry_subcycling=None) -> PreparedSimulation
```

Inherited from: `particula.runnable.RunnableABC`

Resume exact saved state using the same process configuration.

**Parameters**

- `checkpoint` (`SimulationCheckpoint`) — SimulationCheckpoint from a prepared owner.
- `backend` (`Backend | None`) (default: `None`) — Backend.WARP by default.
- `device` (`str | None`) (default: `None`) — Exact checkpoint device by default; migration rejects.
- `process_registry` (default: `None`) — Compatible caller-supplied GPU implementations.
- `retry_subcycling` (default: `None`) — Explicitly relax a saved adaptive coagulation policy while retaining the outer timestep and random streams.

**Returns**

- (`PreparedSimulation`) — A new owner preserving state, RNG words, timestep and counters.
- (`PreparedSimulation`) — Capture is explicit again; native graph handles are never restored.

**Raises**

- `ValueError` — If configuration, order, schema or device differs, or
current population cannot be admitted (e.g. unequal Brownian
counts after a later process injected new particles).

[View source](https://github.com/uncscode/particula/blob/main/particula/runnable.py#L232-L270)

## particula.dynamics.optical_heating.heating_step

*function*

```python
def heating_step(radius, water, solute, activity, absorbed, old_temperature, ambient, vapor, water_molar_mass, dt, physics)
```

Solve one frozen-radius energy balance with conservative mass loss.

Returns new temperature [K], water transferred to gas [kg per particle],
and one energy amount [J per particle] for each heating ledger.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/optical_heating.py#L61-L127)
