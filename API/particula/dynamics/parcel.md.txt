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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L35-L35)

## particula.dynamics.parcel.FloatArray

*attribute* · *module attribute*

```python
FloatArray = NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L40-L40)

## particula.dynamics.parcel.LATENT

*attribute* · *module attribute*

```python
LATENT = 2500000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L36-L36)

## particula.dynamics.parcel.RD

*attribute* · *module attribute*

```python
RD = GAS_CONSTANT / MOLECULAR_WEIGHT_AIR
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L33-L33)

## particula.dynamics.parcel.RHO_W

*attribute* · *module attribute*

```python
RHO_W = 1000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L37-L37)

## particula.dynamics.parcel.RV

*attribute* · *module attribute*

```python
RV = GAS_CONSTANT / 0.01801528
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L34-L34)

## particula.dynamics.parcel.SIGMA

*attribute* · *module attribute*

```python
SIGMA = 0.072
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L38-L38)

## particula.dynamics.parcel.WATER_FACTOR

*attribute* · *module attribute*

```python
WATER_FACTOR = 4 * np.pi / 3 * RHO_W
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L39-L39)

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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L43-L65)

### particula.dynamics.parcel.ParcelInitialState.dry_air_mass

*attribute* · *class attribute* · *instance attribute*

```python
dry_air_mass: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L64-L64)

### particula.dynamics.parcel.ParcelInitialState.dry_radius

*attribute* · *instance attribute*

```python
dry_radius: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L58-L58)

### particula.dynamics.parcel.ParcelInitialState.height

*attribute* · *class attribute* · *instance attribute*

```python
height: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L65-L65)

### particula.dynamics.parcel.ParcelInitialState.kappa

*attribute* · *instance attribute*

```python
kappa: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L60-L60)

### particula.dynamics.parcel.ParcelInitialState.number

*attribute* · *instance attribute*

```python
number: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L59-L59)

### particula.dynamics.parcel.ParcelInitialState.pressure

*attribute* · *instance attribute*

```python
pressure: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L61-L61)

### particula.dynamics.parcel.ParcelInitialState.temperature

*attribute* · *instance attribute*

```python
temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L62-L62)

### particula.dynamics.parcel.ParcelInitialState.vapor

*attribute* · *instance attribute*

```python
vapor: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L63-L63)

### particula.dynamics.parcel.ParcelInitialState.__init__

*method*

```python
def __init__(dry_radius: FloatArray, number: FloatArray, kappa: float, pressure: float, temperature: float, vapor: float, dry_air_mass: float = 1.0, height: float = 0.0) -> None
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
per m/s at fixed elapsed time, or None when unrequested.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L68-L128)

### particula.dynamics.parcel.ParcelTrajectory.height

*attribute* · *instance attribute*

```python
height: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L80-L80)

### particula.dynamics.parcel.ParcelTrajectory.initial

*attribute* · *instance attribute*

```python
initial: ParcelInitialState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L89-L89)

### particula.dynamics.parcel.ParcelTrajectory.liquid

*attribute* · *instance attribute*

```python
liquid: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L85-L85)

### particula.dynamics.parcel.ParcelTrajectory.pressure

*attribute* · *instance attribute*

```python
pressure: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L81-L81)

### particula.dynamics.parcel.ParcelTrajectory.supersaturation

*attribute* · *instance attribute*

```python
supersaturation: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L86-L86)

### particula.dynamics.parcel.ParcelTrajectory.temperature

*attribute* · *instance attribute*

```python
temperature: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L82-L82)

### particula.dynamics.parcel.ParcelTrajectory.time

*attribute* · *instance attribute*

```python
time: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L79-L79)

### particula.dynamics.parcel.ParcelTrajectory.updraft_gradient

*attribute* · *instance attribute*

```python
updraft_gradient: float | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L88-L88)

### particula.dynamics.parcel.ParcelTrajectory.vapor

*attribute* · *instance attribute*

```python
vapor: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L84-L84)

### particula.dynamics.parcel.ParcelTrajectory.volume

*attribute* · *instance attribute*

```python
volume: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L83-L83)

### particula.dynamics.parcel.ParcelTrajectory.wet_radius

*attribute* · *instance attribute*

```python
wet_radius: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L87-L87)

### particula.dynamics.parcel.ParcelTrajectory.__init__

*method*

```python
def __init__(time: FloatArray, height: FloatArray, pressure: FloatArray, temperature: FloatArray, volume: FloatArray, vapor: FloatArray, liquid: FloatArray, supersaturation: FloatArray, wet_radius: FloatArray, updraft_gradient: float | None, initial: ParcelInitialState) -> None
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

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L91-L128)

## particula.dynamics.parcel._cpu_reference

*function*

```python
def _cpu_reference(initial, wet, time, updraft, phase_exchange)
```

Integrate pressure, temperature, vapor and squared radii.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L201-L267)

## particula.dynamics.parcel._equilibrium

*function*

```python
def _equilibrium(radius, dry_radius, temperature, kappa)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L136-L141)

## particula.dynamics.parcel._initial_radii

*function*

```python
def _initial_radii(initial)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L144-L198)

## particula.dynamics.parcel._saturation_pressure

*function*

```python
def _saturation_pressure(temperature)
```

Liquid-water saturation pressure (Pa), constant-L Clausius-Clapeyron.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L131-L133)

## particula.dynamics.parcel.run_parcel

*function*

```python
def run_parcel(initial: ParcelInitialState, *, updraft: float, timestep: float, steps: int, device: str = 'cpu', gradient: bool = False, phase_exchange: bool = True) -> ParcelTrajectory
```

Advance a single parcel under constant signed vertical velocity.

**Parameters**

- `initial` (`ParcelInitialState`) — Subsaturated state including any equilibrium aerosol water.
- `updraft` (`float`) — Vertical velocity in m/s, positive upward.
- `timestep` (`float`) — Output spacing in seconds and Warp integration step.
- `steps` (`int`) — Positive number of intervals.
- `device` (`str`) (default: `'cpu'`) — 'cpu' selects the independent SciPy Radau reference. 'warp:cpu' or a CUDA device selects fixed-step Warp execution.
- `gradient` (`bool`) (default: `False`) — Use Warp reverse mode for terminal liquid versus updraft.
- `phase_exchange` (`bool`) (default: `True`) — Enable signed condensation and latent feedback. A passive run retains the initial equilibrium water on each class.

**Returns**

- (`ParcelTrajectory`) — State histories and the requested Warp derivative.

**Raises**

- `ValueError` — Invalid state, backend, integration interval or trajectory.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/parcel.py#L270-L360)
