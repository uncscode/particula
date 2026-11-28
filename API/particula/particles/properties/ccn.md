# particula.particles.properties.ccn

*module*

Equilibrium CCN spectra of dry material in discrete or resolved particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L1-L1)

## particula.particles.properties.ccn.CCNConfig

*class* · *dataclass*

```python
class CCNConfig
```

Define counter settings and the dry material model.

saturation_ratios are dimensionless and above one. Supply either kappa
(one value or one per box) or material_kappa in particle material order,
with zero in the water lane. Supply water_index explicitly, using None
for an all-dry material list. kelvin_diameter is A in metres, scalar or
per box. If omitted, use the live box temperature (K), surface tension
(N/m), water density (kg/m³) and water molar mass (kg/mol).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L27-L109)

### particula.particles.properties.ccn.CCNConfig.kappa

*attribute* · *class attribute* · *instance attribute*

```python
kappa: float | tuple[float, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L41-L41)

### particula.particles.properties.ccn.CCNConfig.kelvin_diameter

*attribute* · *class attribute* · *instance attribute*

```python
kelvin_diameter: float | tuple[float, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L43-L43)

### particula.particles.properties.ccn.CCNConfig.material_kappa

*attribute* · *class attribute* · *instance attribute*

```python
material_kappa: tuple[float, ...] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L42-L42)

### particula.particles.properties.ccn.CCNConfig.saturation_ratios

*attribute* · *instance attribute*

```python
saturation_ratios: tuple[float, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L39-L39)

### particula.particles.properties.ccn.CCNConfig.surface_tension

*attribute* · *class attribute* · *instance attribute*

```python
surface_tension: float = 0.072
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L44-L44)

### particula.particles.properties.ccn.CCNConfig.water_density

*attribute* · *class attribute* · *instance attribute*

```python
water_density: float = 997.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L45-L45)

### particula.particles.properties.ccn.CCNConfig.water_index

*attribute* · *instance attribute*

```python
water_index: int | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L40-L40)

### particula.particles.properties.ccn.CCNConfig.water_molar_mass

*attribute* · *class attribute* · *instance attribute*

```python
water_molar_mass: float = 0.01801528
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L46-L46)

### particula.particles.properties.ccn.CCNConfig.kelvin_temperature_product

*property*

```python
kelvin_temperature_product
```

Return A times T in m K for constant water properties.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L77-L81)

### particula.particles.properties.ccn.CCNConfig.__init__

*method*

```python
def __init__(saturation_ratios: tuple[float, ...], water_index: int | None, kappa: float | tuple[float, ...] | None = None, material_kappa: tuple[float, ...] | None = None, kelvin_diameter: float | tuple[float, ...] | None = None, surface_tension: float = 0.072, water_density: float = 997.0, water_molar_mass: float = 0.01801528) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L0-L0)

### particula.particles.properties.ccn.CCNConfig.__post_init__

*method*

```python
def __post_init__()
```

Freeze input coordinates and reject conflicting material models.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L48-L74)

### particula.particles.properties.ccn.CCNConfig.bind

*method*

```python
def bind(boxes, materials, distribution)
```

Validate population and material axes and expand per-box inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L83-L109)

## particula.particles.properties.ccn.CCNResult

*class* · *dataclass*

```python
class CCNResult
```

Number density (box, setting) and critical dry diameters [m].

Common κ yields diameters (box, setting). Mixed material yields individual
diameters (box, slot, setting), with NaN for slots without a dry seed.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L112-L121)

### particula.particles.properties.ccn.CCNResult.critical_dry_diameter

*attribute* · *instance attribute*

```python
critical_dry_diameter: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L121-L121)

### particula.particles.properties.ccn.CCNResult.equilibrium_ccn_number_concentration

*attribute* · *instance attribute*

```python
equilibrium_ccn_number_concentration: np.ndarray
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L120-L120)

### particula.particles.properties.ccn.CCNResult.__init__

*method*

```python
def __init__(equilibrium_ccn_number_concentration: np.ndarray, critical_dry_diameter: np.ndarray) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L0-L0)

## particula.particles.properties.ccn._evaluate

*function*

```python
def _evaluate(masses, population, volume, density, temperature, distribution, config, fixed=None)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L124-L176)

## particula.particles.properties.ccn._sample_state

*function*

```python
def _sample_state(state, config, fixed)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L184-L196)

## particula.particles.properties.ccn._values

*function*

```python
def _values(value, name, lower=0.0, strict=False)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L16-L24)

## particula.particles.properties.ccn.evaluate_ccn

*function*

```python
def evaluate_ccn(state, config: CCNConfig) -> CCNResult
```

Evaluate CCN number [m⁻³] and dry thresholds [m] on a snapshot.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L179-L181)

## particula.particles.properties.ccn.evaluate_ccn_history

*function*

```python
def evaluate_ccn_history(history, config: CCNConfig) -> CCNResult
```

Evaluate a saved native history on a new saturation grid.

Requires slot masses, populations, sampled volume and material densities.
Temperature is needed for a temperature-derived Kelvin length. Legacy
discrete concentration histories use their per-volume convention.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/ccn.py#L199-L238)
