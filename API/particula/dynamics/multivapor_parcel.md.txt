# particula.dynamics.multivapor_parcel

*module*

Closed parcel with analytical particle masses and finite vapor species.

Mixing ratios refer to one kg of dry air. The dilute first law neglects
condensate volume and sensible heat, dry aerosol loading and entrainment.
The continuum diffusion law uses unit accommodation, unit ventilation and
a single linearized conductive heat balance at the surface of each droplet.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L1-L1)

## particula.dynamics.multivapor_parcel.CP

*attribute* · *module attribute*

```python
CP = 1005.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L28-L28)

## particula.dynamics.multivapor_parcel.FloatArray

*attribute* · *module attribute*

```python
FloatArray = NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L26-L26)

## particula.dynamics.multivapor_parcel.RD

*attribute* · *module attribute*

```python
RD = GAS_CONSTANT / MOLECULAR_WEIGHT_AIR
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L27-L27)

## particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState

*class* · *dataclass*

```python
class MultiVaporParcelInitialState
```

An explicitly wet class population and finite vapor reservoirs.

masses has shape (class, material) in kg/particle. number gives represented
counts per class. vapor holds kg/kg dry air per gas in vapors order.
density is kg/m³ per material. pressure, temperature, dry_air_mass,
height and surface_tension use Pa, K, kg, m and N/m respectively.
A vapor named 'water' identifies the supersaturation diagnostic.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L75-L147)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.activity_strategy

*attribute* · *instance attribute*

```python
activity_strategy: ActivityStrategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L92-L92)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.density

*attribute* · *instance attribute*

```python
density: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L90-L90)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.dry_air_mass

*attribute* · *class attribute* · *instance attribute*

```python
dry_air_mass: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L95-L95)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.height

*attribute* · *class attribute* · *instance attribute*

```python
height: float = 0.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L96-L96)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.masses

*attribute* · *instance attribute*

```python
masses: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L86-L86)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.material_names

*attribute* · *instance attribute*

```python
material_names: tuple[str, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L89-L89)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.number

*attribute* · *instance attribute*

```python
number: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L87-L87)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.pressure

*attribute* · *instance attribute*

```python
pressure: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L93-L93)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.surface_tension

*attribute* · *class attribute* · *instance attribute*

```python
surface_tension: float = 0.072
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L97-L97)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.temperature

*attribute* · *instance attribute*

```python
temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L94-L94)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.vapor

*attribute* · *instance attribute*

```python
vapor: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L88-L88)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.vapors

*attribute* · *instance attribute*

```python
vapors: tuple[ParcelVapor, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L91-L91)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.__init__

*method*

```python
def __init__(masses: FloatArray, number: FloatArray, vapor: FloatArray, material_names: tuple[str, ...], density: FloatArray, vapors: tuple[ParcelVapor, ...], activity_strategy: ActivityStrategy, pressure: float, temperature: float, dry_air_mass: float = 1.0, height: float = 0.0, surface_tension: float = 0.072) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L0-L0)

### particula.dynamics.multivapor_parcel.MultiVaporParcelInitialState.__post_init__

*method*

```python
def __post_init__()
```

Freeze caller-owned input arrays and verify extensive inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L99-L147)

## particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory

*class* · *dataclass*

```python
class MultiVaporParcelTrajectory
```

Histories at common output times in SI and kg per kg of dry air.

masses is (time, class, material) in kg/particle, wet_radius is (time,
class) in m. vapor and liquid are (time, vapor) mixing ratios in declared
gas order. volume is physical parcel m³, supersaturation is RH minus one.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L150-L169)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.height

*attribute* · *instance attribute*

```python
height: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L160-L160)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.initial

*attribute* · *instance attribute*

```python
initial: MultiVaporParcelInitialState
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L169-L169)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.liquid

*attribute* · *instance attribute*

```python
liquid: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L165-L165)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.masses

*attribute* · *instance attribute*

```python
masses: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L166-L166)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.pressure

*attribute* · *instance attribute*

```python
pressure: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L161-L161)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.supersaturation

*attribute* · *instance attribute*

```python
supersaturation: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L168-L168)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.temperature

*attribute* · *instance attribute*

```python
temperature: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L162-L162)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.time

*attribute* · *instance attribute*

```python
time: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L159-L159)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.vapor

*attribute* · *instance attribute*

```python
vapor: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L164-L164)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.volume

*attribute* · *instance attribute*

```python
volume: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L163-L163)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.wet_radius

*attribute* · *instance attribute*

```python
wet_radius: FloatArray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L167-L167)

### particula.dynamics.multivapor_parcel.MultiVaporParcelTrajectory.__init__

*method*

```python
def __init__(time: FloatArray, height: FloatArray, pressure: FloatArray, temperature: FloatArray, volume: FloatArray, vapor: FloatArray, liquid: FloatArray, masses: FloatArray, wet_radius: FloatArray, supersaturation: FloatArray, initial: MultiVaporParcelInitialState) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L0-L0)

## particula.dynamics.multivapor_parcel.ParcelTendencies

*class*

```python
class ParcelTendencies
```

Evaluate the coupled instantaneous mass, pressure and heat budgets.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L241-L391)

### particula.dynamics.multivapor_parcel.ParcelTendencies.diffusivity

*attribute* · *instance attribute*

```python
diffusivity = np.array([v.diffusivity for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L253-L253)

### particula.dynamics.multivapor_parcel.ParcelTendencies.enabled

*attribute* · *instance attribute*

```python
enabled = np.array([phase_exchange and v.enabled for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L256-L258)

### particula.dynamics.multivapor_parcel.ParcelTendencies.gas_r

*attribute* · *instance attribute*

```python
gas_r = GAS_CONSTANT / np.array([v.molar_mass for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L249-L251)

### particula.dynamics.multivapor_parcel.ParcelTendencies.initial

*attribute* · *instance attribute*

```python
initial = initial
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L246-L246)

### particula.dynamics.multivapor_parcel.ParcelTendencies.latent

*attribute* · *instance attribute*

```python
latent = np.array([v.latent_heat for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L252-L252)

### particula.dynamics.multivapor_parcel.ParcelTendencies.loading

*attribute* · *instance attribute*

```python
loading = 1 + initial.vapor.sum() + initial.number @ initial.masses[:, self.targets].sum(axis=1) / initial.dry_air_mass
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L259-L265)

### particula.dynamics.multivapor_parcel.ParcelTendencies.p0

*attribute* · *instance attribute*

```python
p0 = np.array([v.reference_pressure for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L254-L254)

### particula.dynamics.multivapor_parcel.ParcelTendencies.t0

*attribute* · *instance attribute*

```python
t0 = np.array([v.reference_temperature for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L255-L255)

### particula.dynamics.multivapor_parcel.ParcelTendencies.targets

*attribute* · *instance attribute*

```python
targets = np.array([v.material_index for v in initial.vapors])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L248-L248)

### particula.dynamics.multivapor_parcel.ParcelTendencies.updraft

*attribute* · *instance attribute*

```python
updraft = updraft
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L247-L247)

### particula.dynamics.multivapor_parcel.ParcelTendencies.__init__

*method*

```python
def __init__(initial, updraft, phase_exchange=True)
```

Bind property vectors and the conserved hydrostatic loading.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L244-L265)

### particula.dynamics.multivapor_parcel.ParcelTendencies.rates

*method*

```python
def rates(pressure, temperature, vapor, masses)
```

Return dp (Pa/s), dT (K/s), dq (kg/kg/s), and J (kg/particle/s).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L267-L391)

## particula.dynamics.multivapor_parcel.ParcelVapor

*class* · *dataclass*

```python
class ParcelVapor
```

Pure species properties for dilute continuum transfer to one liquid.

name and material_index bind the gas to an analytical particle material.
molar_mass is kg/mol, reference_pressure is Pa at reference_temperature
(K), latent_heat is J/kg and diffusivity is m²/s at 273.15 K/101325 Pa.
Unit accommodation applies to this continuum model. With enabled=False
the vapor remains part of the gas law and its condensed material still
contributes to AIOMFAC activities and wet radius.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L31-L72)

### particula.dynamics.multivapor_parcel.ParcelVapor.accommodation

*attribute* · *class attribute* · *instance attribute*

```python
accommodation: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L50-L50)

### particula.dynamics.multivapor_parcel.ParcelVapor.diffusivity

*attribute* · *instance attribute*

```python
diffusivity: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L49-L49)

### particula.dynamics.multivapor_parcel.ParcelVapor.enabled

*attribute* · *class attribute* · *instance attribute*

```python
enabled: bool = True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L51-L51)

### particula.dynamics.multivapor_parcel.ParcelVapor.latent_heat

*attribute* · *instance attribute*

```python
latent_heat: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L48-L48)

### particula.dynamics.multivapor_parcel.ParcelVapor.material_index

*attribute* · *instance attribute*

```python
material_index: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L44-L44)

### particula.dynamics.multivapor_parcel.ParcelVapor.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L45-L45)

### particula.dynamics.multivapor_parcel.ParcelVapor.name

*attribute* · *instance attribute*

```python
name: str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L43-L43)

### particula.dynamics.multivapor_parcel.ParcelVapor.reference_pressure

*attribute* · *instance attribute*

```python
reference_pressure: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L46-L46)

### particula.dynamics.multivapor_parcel.ParcelVapor.reference_temperature

*attribute* · *instance attribute*

```python
reference_temperature: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L47-L47)

### particula.dynamics.multivapor_parcel.ParcelVapor.__init__

*method*

```python
def __init__(name: str, material_index: int, molar_mass: float, reference_pressure: float, reference_temperature: float, latent_heat: float, diffusivity: float, accommodation: float = 1.0, enabled: bool = True) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L0-L0)

### particula.dynamics.multivapor_parcel.ParcelVapor.__post_init__

*method*

```python
def __post_init__()
```

Reject unsupported transport properties before any integration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L53-L72)

## particula.dynamics.multivapor_parcel._TrialDomainError

*class*

```python
class _TrialDomainError(ValueError)
```

Bases: `ValueError`

A numerical trial left the physical domain before acceptance.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L209-L210)

## particula.dynamics.multivapor_parcel._activity_intervals

*function*

```python
def _activity_intervals(provider, masses, temperature, targets)
```

Return activities and optional error estimates for occupied classes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L231-L238)

## particula.dynamics.multivapor_parcel._uncertain_fluxes

*function*

```python
def _uncertain_fluxes(rates, driving, slope, sink, offset, errors, latent, enabled)
```

Bound the shared surface temperature before each species flux.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L213-L228)

## particula.dynamics.multivapor_parcel.cpu_multivapor_reference

*function*

```python
def cpu_multivapor_reference(initial, time, updraft, phase_exchange=True, rtol=2e-10)
```

Integrate coupled inventories independently using scaled SciPy Radau.

An initial class mass scale per lane and independent vapor variables keep
the embedded error estimate sensitive to both trace and bulk species.
Failed physical Radau trials retry from the accepted state at a smaller
step. Provider errors on the initial state are reported immediately.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L394-L476)

## particula.dynamics.multivapor_parcel.multivapor_trajectory

*function*

```python
def multivapor_trajectory(initial, time, pressure, temperature, vapor, masses, updraft)
```

Construct the shared observable record from CPU or device outputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/multivapor_parcel.py#L172-L206)
