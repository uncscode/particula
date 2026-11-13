# particula.activity.bat_blending

*module*

Blending weights for the BAT model.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py#L1-L1)

## particula.activity.bat_blending._calculate_blending_weights

*function*

```python
def _calculate_blending_weights(oxygen2carbon: float, oxygen2carbon_ml: float) -> NDArray[np.float64]
```

Helper to compute blending weights for a single oxygen value.

**Parameters**

- `oxygen2carbon` (`float`) — The oxygen-to-carbon ratio for the sample point.
- `oxygen2carbon_ml` (`float`) — The single-phase oxygen-to-carbon threshold.

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Weights for the low, mid, and high regimes.

**Examples**

```pycon
>>> from particula.activity.bat_blending import (
...     _calculate_blending_weights,
... )
>>> _calculate_blending_weights(
...     oxygen2carbon=0.35,
...     oxygen2carbon_ml=0.225,
... )
array([0.00000000e+00, 9.11531178e-05, 9.99908847e-01])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py#L65-L115)

## particula.activity.bat_blending.bat_blending_weights

*function*

```python
def bat_blending_weights(molar_mass_ratio: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]
```

Estimate BAT blending weights for oxygen-rich regimes.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Ratio of water to organic matter. Must be positive.
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon ratio. Must be nonnegative.

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Blending weights for the low, mid, and high oxygen regions. Scalar inputs return shape (3,), array inputs return (n, 3) where n is the length of ``oxygen2carbon``.

**Examples**

```pycon
>>> from particula.activity.bat_blending import bat_blending_weights
>>> bat_blending_weights(molar_mass_ratio=0.5, oxygen2carbon=0.35)
array([0.00000000e+00, 9.11531178e-05, 9.99908847e-01])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py#L19-L62)
