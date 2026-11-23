# particula.dynamics.parcel

*module*

Closed, warm-cloud parcel with one soluble, size-resolved aerosol mode.

Mixing ratios use kg per kg dry air. Hydrostatic balance includes suspended
liquid loading. The dilute first law uses constant dry-air heat capacity and
latent heat, and neglects condensate volume, dry aerosol loading, sensible
heat of water, fallout and entrainment. Diffusion and thermal resistances act
in series (continuum growth with unit ventilation). See Rothenberg & Wang
(2016), doi:10.1175/JAS-D-15-0223.1, and Petters & Kreidenweis (2007),
doi:10.5194/acp-7-1961-2007. The concrete runner operates on parcel counts
and physical volume. Its binned number densities are counts divided by volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L1-L1)

## particula.dynamics.parcel.CP

*attribute* · *module attribute*

```python
CP = 1005.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L40-L40)

## particula.dynamics.parcel.FloatArray

*attribute* · *module attribute*

```python
FloatArray = NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L45-L45)

## particula.dynamics.parcel.LATENT

*attribute* · *module attribute*

```python
LATENT = 2500000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L41-L41)

## particula.dynamics.parcel.RD

*attribute* · *module attribute*

```python
RD = GAS_CONSTANT / MOLECULAR_WEIGHT_AIR
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L38-L38)

## particula.dynamics.parcel.RHO_W

*attribute* · *module attribute*

```python
RHO_W = 1000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L42-L42)

## particula.dynamics.parcel.RV

*attribute* · *module attribute*

```python
RV = GAS_CONSTANT / 0.01801528
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L39-L39)

## particula.dynamics.parcel.SIGMA

*attribute* · *module attribute*

```python
SIGMA = 0.072
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L43-L43)

## particula.dynamics.parcel.WATER_FACTOR

*attribute* · *module attribute*

```python
WATER_FACTOR = 4 * np.pi / 3 * RHO_W
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L44-L44)

## particula.dynamics.parcel.ParcelInitialState

*class* · *dataclass*

```python
class ParcelInitialState
```

Initial physical state before equilibration at subsaturated humidity.

**Attributes**

- `dry_radius` (`FloatArray`) — Positive dry radius of each class (m).
- `number` (`FloatArray`) — Nonnegative represented particle count per class in the parcel.
- `kappa` (`float`) — Positive dimensionless hygroscopicity shared by the classes.
- `pressure` (`float`) — Initial pressure (Pa).
- `temperature` (`float`) — Initial temperature (K).
- `vapor` (`float`) — Initial vapor mixing ratio (kg/kg dry air).
- `dry_air_mass` (`float`) — Fixed dry air inventory (kg).
- `height` (`float`) — Initial altitude (m).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L48-L70)

### particula.dynamics.parcel.ParcelInitialState.dry_air_mass

*attribute* · *class attribute* · *instance attribute*

```python
dry_air_mass: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L69-L69)

### particula.dynamics.parcel.ParcelInitialState.dry_radius

*attribute* · *instance attribute*

```python
dry_radius: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L63-L63)

### particula.dynamics.parcel.ParcelInitialState.height

*attribute* · *class attribute* · *instance attribute*

```python
height: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L70-L70)

### particula.dynamics.parcel.ParcelInitialState.kappa

*attribute* · *instance attribute*

```python
kappa: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L65-L65)

### particula.dynamics.parcel.ParcelInitialState.number

*attribute* · *instance attribute*

```python
number: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L64-L64)

### particula.dynamics.parcel.ParcelInitialState.pressure

*attribute* · *instance attribute*

```python
pressure: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L66-L66)

### particula.dynamics.parcel.ParcelInitialState.temperature

*attribute* · *instance attribute*

```python
temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L67-L67)

### particula.dynamics.parcel.ParcelInitialState.vapor

*attribute* · *instance attribute*

```python
vapor: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L68-L68)

### particula.dynamics.parcel.ParcelInitialState.__init__

*method*

```python
def __init__(dry_radius: FloatArray, number: FloatArray, kappa: float, pressure: float, temperature: float, vapor: float, dry_air_mass: float = 1.0, height: float = 0.0) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L0-L0)

## particula.dynamics.parcel.ParcelRK23

*class* · *dataclass*

```python
class ParcelRK23(_ParcelAdaptiveControls)
```

Bases: `particula.dynamics.parcel._ParcelAdaptiveControls`

Adaptive Warp controls for the embedded Bogacki–Shampine 3(2) pair.

rtol controls pressure, each class's liquid volume, vapor and temperature.
The relative scales include floors of 1e-8 in normalized pressure and
liquid/dry class volume, and 1e-12 kg/kg dry air for vapor.
min_step is the smallest permitted retry in seconds. max_trials bounds
all evaluated trials, including unused speculative Warp trials. A linearized
droplet relaxation limit also protects sensitivities near equilibrium.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L95-L105)

### particula.dynamics.parcel.ParcelRK23.max_trials

*attribute* · *class attribute* · *instance attribute*

```python
max_trials: int = 100000
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L79-L79)

### particula.dynamics.parcel.ParcelRK23.min_step

*attribute* · *class attribute* · *instance attribute*

```python
min_step: float = 1e-09
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L78-L78)

### particula.dynamics.parcel.ParcelRK23.rtol

*attribute* · *class attribute* · *instance attribute*

```python
rtol: float = 1e-05
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L77-L77)

### particula.dynamics.parcel.ParcelRK23.__init__

*method*

```python
def __init__(rtol: float = 1e-05, min_step: float = 1e-09, max_trials: int = 100000) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L0-L0)

### particula.dynamics.parcel.ParcelRK23.__post_init__

*method*

```python
def __post_init__()
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

Reject invalid controller settings before device allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L81-L92)

## particula.dynamics.parcel.ParcelROS3

*class* · *dataclass*

```python
class ParcelROS3(_ParcelAdaptiveControls)
```

Bases: `particula.dynamics.parcel._ParcelAdaptiveControls`

Adaptive third-order Rosenbrock solver with a second-order estimator.

Uses the coupled parcel Jacobian and explicit time derivative at each
trial. rtol, min_step (s) and max_trials use the ParcelRK23 scales.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L108-L114)

### particula.dynamics.parcel.ParcelROS3.max_trials

*attribute* · *class attribute* · *instance attribute*

```python
max_trials: int = 100000
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L79-L79)

### particula.dynamics.parcel.ParcelROS3.min_step

*attribute* · *class attribute* · *instance attribute*

```python
min_step: float = 1e-09
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L78-L78)

### particula.dynamics.parcel.ParcelROS3.rtol

*attribute* · *class attribute* · *instance attribute*

```python
rtol: float = 1e-05
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L77-L77)

### particula.dynamics.parcel.ParcelROS3.__init__

*method*

```python
def __init__(rtol: float = 1e-05, min_step: float = 1e-09, max_trials: int = 100000) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L0-L0)

### particula.dynamics.parcel.ParcelROS3.__post_init__

*method*

```python
def __post_init__()
```

Inherited from: `particula.dynamics.parcel._ParcelAdaptiveControls`

Reject invalid controller settings before device allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L81-L92)

## particula.dynamics.parcel.ParcelStepStatistics

*class* · *dataclass*

```python
class ParcelStepStatistics
```

Controller counts and smallest/largest accepted timestep (s).

evaluated_trials includes speculative candidates discarded after the
controller changes step size. rejected_steps counts controller rejections.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L117-L129)

### particula.dynamics.parcel.ParcelStepStatistics.accepted_steps

*attribute* · *instance attribute*

```python
accepted_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L125-L125)

### particula.dynamics.parcel.ParcelStepStatistics.evaluated_trials

*attribute* · *instance attribute*

```python
evaluated_trials: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L129-L129)

### particula.dynamics.parcel.ParcelStepStatistics.maximum_step

*attribute* · *instance attribute*

```python
maximum_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L128-L128)

### particula.dynamics.parcel.ParcelStepStatistics.minimum_step

*attribute* · *instance attribute*

```python
minimum_step: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L127-L127)

### particula.dynamics.parcel.ParcelStepStatistics.rejected_steps

*attribute* · *instance attribute*

```python
rejected_steps: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L126-L126)

### particula.dynamics.parcel.ParcelStepStatistics.__init__

*method*

```python
def __init__(accepted_steps: int, rejected_steps: int, minimum_step: float, maximum_step: float, evaluated_trials: int) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L0-L0)

## particula.dynamics.parcel.ParcelTrajectory

*class* · *dataclass*

```python
class ParcelTrajectory
```

Float64 samples, including the initial state, on the requested cadence.

time (s), height (m), pressure (Pa), temperature (K), volume (m³),
vapor and liquid (kg/kg dry air), and supersaturation (fraction above
saturation) each have one entry per time. wet_radius (m) has axes time
and class. updraft_gradient is the terminal liquid mixing ratio derivative
per m/s at fixed elapsed time, or None when unrequested. statistics
contains work counts and step extrema for adaptive Warp execution.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L132-L194)

### particula.dynamics.parcel.ParcelTrajectory.height

*attribute* · *instance attribute*

```python
height: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L145-L145)

### particula.dynamics.parcel.ParcelTrajectory.initial

*attribute* · *instance attribute*

```python
initial: ParcelInitialState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L154-L154)

### particula.dynamics.parcel.ParcelTrajectory.liquid

*attribute* · *instance attribute*

```python
liquid: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L150-L150)

### particula.dynamics.parcel.ParcelTrajectory.pressure

*attribute* · *instance attribute*

```python
pressure: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L146-L146)

### particula.dynamics.parcel.ParcelTrajectory.statistics

*attribute* · *class attribute* · *instance attribute*

```python
statistics: ParcelStepStatistics | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L155-L155)

### particula.dynamics.parcel.ParcelTrajectory.supersaturation

*attribute* · *instance attribute*

```python
supersaturation: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L151-L151)

### particula.dynamics.parcel.ParcelTrajectory.temperature

*attribute* · *instance attribute*

```python
temperature: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L147-L147)

### particula.dynamics.parcel.ParcelTrajectory.time

*attribute* · *instance attribute*

```python
time: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L144-L144)

### particula.dynamics.parcel.ParcelTrajectory.updraft_gradient

*attribute* · *instance attribute*

```python
updraft_gradient: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L153-L153)

### particula.dynamics.parcel.ParcelTrajectory.vapor

*attribute* · *instance attribute*

```python
vapor: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L149-L149)

### particula.dynamics.parcel.ParcelTrajectory.volume

*attribute* · *instance attribute*

```python
volume: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L148-L148)

### particula.dynamics.parcel.ParcelTrajectory.wet_radius

*attribute* · *instance attribute*

```python
wet_radius: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L152-L152)

### particula.dynamics.parcel.ParcelTrajectory.__init__

*method*

```python
def __init__(time: FloatArray, height: FloatArray, pressure: FloatArray, temperature: FloatArray, volume: FloatArray, vapor: FloatArray, liquid: FloatArray, supersaturation: FloatArray, wet_radius: FloatArray, updraft_gradient: float | None, initial: ParcelInitialState, statistics: ParcelStepStatistics | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L0-L0)

### particula.dynamics.parcel.ParcelTrajectory.aerosol_at

*method*

```python
def aerosol_at(sample: int, *, dry_density: float) -> Aerosol
```

Return an independent native one-box snapshot at a time index.

The discrete particle lanes are dry material and water, respectively.
Supply dry material density (kg/m³) to convert the fixed dry volumes
to mass. Gas holds water vapor in kg/m³. The native discrete volume
is 1 m³. Multiply its density inventories by the physical parcel
volume at this sample to recover extensive quantities.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L157-L194)

## particula.dynamics.parcel._ParcelAdaptiveControls

*class* · *dataclass*

```python
class _ParcelAdaptiveControls
```

Shared local error tolerance and adaptive trial limits.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L73-L92)

### particula.dynamics.parcel._ParcelAdaptiveControls.max_trials

*attribute* · *class attribute* · *instance attribute*

```python
max_trials: int = 100000
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L79-L79)

### particula.dynamics.parcel._ParcelAdaptiveControls.min_step

*attribute* · *class attribute* · *instance attribute*

```python
min_step: float = 1e-09
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L78-L78)

### particula.dynamics.parcel._ParcelAdaptiveControls.rtol

*attribute* · *class attribute* · *instance attribute*

```python
rtol: float = 1e-05
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L77-L77)

### particula.dynamics.parcel._ParcelAdaptiveControls.__init__

*method*

```python
def __init__(rtol: float = 1e-05, min_step: float = 1e-09, max_trials: int = 100000) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L0-L0)

### particula.dynamics.parcel._ParcelAdaptiveControls.__post_init__

*method*

```python
def __post_init__()
```

Reject invalid controller settings before device allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L81-L92)

## particula.dynamics.parcel._cpu_reference

*function*

```python
def _cpu_reference(initial, wet, time, updraft, phase_exchange)
```

Integrate pressure, temperature, vapor and squared radii.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L267-L333)

## particula.dynamics.parcel._equilibrium

*function*

```python
def _equilibrium(radius, dry_radius, temperature, kappa)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L202-L207)

## particula.dynamics.parcel._initial_radii

*function*

```python
def _initial_radii(initial)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L210-L264)

## particula.dynamics.parcel._parcel_output_times

*function*

```python
def _parcel_output_times(steps, timestep, updraft)
```

Validate common parcel forcing and construct the output time grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L336-L348)

## particula.dynamics.parcel._run_multivapor

*function*

```python
def _run_multivapor(initial, time, updraft, device, adaptive, phase_exchange, gradient)
```

Dispatch the explicit wet-state parcel after validating its controls.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L351-L373)

## particula.dynamics.parcel._saturation_pressure

*function*

```python
def _saturation_pressure(temperature)
```

Liquid-water saturation pressure (Pa), constant-L Clausius-Clapeyron.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L197-L199)

## particula.dynamics.parcel.run_parcel

*function*

```python
def run_parcel(initial: ParcelInitialState | MultiVaporParcelInitialState, *, updraft: float, timestep: float, steps: int, device: str = 'cpu', gradient: bool = False, phase_exchange: bool = True, adaptive: ParcelRK23 | ParcelROS3 | None = None) -> ParcelTrajectory | MultiVaporParcelTrajectory
```

Advance a single parcel under constant signed vertical velocity.

**Parameters**

- `initial` (`ParcelInitialState | MultiVaporParcelInitialState`) — A subsaturated κ parcel or explicit multi-vapor wet classes. Multi-vapor input specifies each initial material and gas inventory.
- `updraft` (`float`) — Vertical velocity in m/s, positive upward.
- `timestep` (`float`) — Output spacing (s). Also the fixed Warp step by default.
- `steps` (`int`) — Positive number of intervals.
- `device` (`str`) (default: `'cpu'`) — 'cpu' selects the independent SciPy Radau reference. 'warp:cpu' or a CUDA device selects Warp execution.
- `gradient` (`bool`) (default: `False`) — Use Warp reverse mode for terminal liquid versus updraft.
- `phase_exchange` (`bool`) (default: `True`) — Enable signed condensation and latent feedback. A passive run retains the initial equilibrium water on each class.
- `adaptive` (`ParcelRK23 | ParcelROS3 | None`) (default: `None`) — Optional Warp RK23 or ROS3 controls. Internal steps end at the requested output times. The gradient follows the accepted discrete schedule.

**Returns**

- (`ParcelTrajectory | MultiVaporParcelTrajectory`) — State histories and the requested Warp derivative.

**Raises**

- `ValueError` — Invalid state, backend, integration interval or trajectory.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L376-L488)
