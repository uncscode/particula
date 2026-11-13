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
def fixed_water_activity(water_activity: Union[float, NDArray[np.float64]], molar_mass_ratio: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]]) -> Tuple[Union[float, NDArray[np.float64]], Optional[Union[float, NDArray[np.float64]]], Union[float, NDArray[np.float64]]]
```

Activity coefficients in organic-water mixtures.

This function assumes a fixed water activity value (e.g., RH = 75%
corresponds to 0.75 water activity in equilibrium).
It calculates the activity coefficients for different phases and
determines phase separations if they occur.

**Parameters**

- `water_activity` (`Union[float, NDArray[np.float64]]`) — An array of water activity values.
- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Array of molar mass ratios of the components.
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Array of oxygen-to-carbon ratios.
- `density` (`Union[float, NDArray[np.float64]]`) — Array of densities of the mixture, in kg/m^3.

**Returns**

- (`Tuple[Union[float, NDArray[np.float64]], Optional[Union[float, NDArray[np.float64]]], Union[float, NDArray[np.float64]]]`) — A tuple containing the activity coefficients for alpha and beta phases, and the q_alpha (phase separation) value. If no phase separation occurs, the beta phase values are None.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/water_activity.py#L99-L244)
