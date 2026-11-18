# particula.activity.water_activity

*module*

Calculate the activity coefficients at fixed water activity.

This module contains functions to calculates the activity coefficients
at a fixed water activity or the water activity cross point.

Gorkowski, K., Preston, T. C., Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L1-L1)

## particula.activity.water_activity._PhaseProperties

*attribute* · *module attribute*

```python
_PhaseProperties = tuple[float | NDArray[np.float64], ...] | NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L99-L99)

## particula.activity.water_activity.FixedWaterActivityResult

*class*

```python
class FixedWaterActivityResult(NamedTuple)
```

Bases: `NamedTuple`

Properties of the water-rich and optional organic-rich liquid branches.

Each phase contains water and organic activities, water and organic mass
fractions, then water and organic activity coefficients, in that order.
The last axis of a sweep follows the requested water activities.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L102-L112)

### particula.activity.water_activity.FixedWaterActivityResult.alpha

*attribute* · *instance attribute*

```python
alpha: _PhaseProperties
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L110-L110)

### particula.activity.water_activity.FixedWaterActivityResult.beta

*attribute* · *instance attribute*

```python
beta: _PhaseProperties | None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L111-L111)

### particula.activity.water_activity.FixedWaterActivityResult.q_alpha

*attribute* · *instance attribute*

```python
q_alpha: float | NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L112-L112)

## particula.activity.water_activity.biphasic_water_activity_point

*function*

```python
def biphasic_water_activity_point(oxygen2carbon: Union[float, NDArray[np.float64]], hydrogen2carbon: Union[float, NDArray[np.float64]], molar_mass_ratio: Union[float, NDArray[np.float64]], functional_group: Optional[Union[list[str], str]] = None) -> np.ndarray
```

Computes the biphasic to single phase water activity (RH*100).

**Parameters**

- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — The oxygen to carbon ratio.
- `hydrogen2carbon` (`Union[float, NDArray[np.float64]]`) — The hydrogen to carbon ratio.
- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — The molar mass ratio of water to organic matter.
- `functional_group` (`Optional[Union[list[str], str]]`) (default: `None`) — Optional functional group(s) of the organic compound, if applicable.

**Returns**

- (`np.ndarray`) — The RH cross point array.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L29-L96)

## particula.activity.water_activity.fixed_water_activity

*function*

```python
def fixed_water_activity(water_activity: Union[float, NDArray[np.float64]], molar_mass_ratio: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]]) -> FixedWaterActivityResult
```

Evaluate binary BAT liquid branches at prescribed water activities.

**Parameters**

- `water_activity` (`Union[float, NDArray[np.float64]]`) — Dimensionless target activity, scalar or 1-D sweep.
- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Water molar mass divided by organic molar mass.
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Organic atomic O:C ratio.
- `density` (`Union[float, NDArray[np.float64]]`) — Organic liquid density in kg/m³.

**Returns**

- (`FixedWaterActivityResult`) — Phase properties and the model's alpha-phase allocation fraction.
- (`FixedWaterActivityResult`) — The result supports tuple unpacking as (alpha, beta, q_alpha).
- (`FixedWaterActivityResult`) — A homogeneous solution has beta equal to None and q_alpha equal to 1.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L115-L253)
