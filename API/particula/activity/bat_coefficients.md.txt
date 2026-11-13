# particula.activity.bat_coefficients

*module*

Fit coefficients for the Binary Activity Coefficient.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L1-L1)

## particula.activity.bat_coefficients.FloatArray

*attribute* · *module attribute*

```python
FloatArray = Union[float, NDArray[np.float64]]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L15-L15)

## particula.activity.bat_coefficients.G19_FIT_HIGH

*attribute* · *module attribute*

```python
G19_FIT_HIGH = FitValues(a1=[5.92155, -2.528295, -3.883017, -7.898128], a2=[-100.0, -100.0, 1.353916, -11.60145], s=[-0.07868187, 3.65086])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L36-L40)

## particula.activity.bat_coefficients.G19_FIT_LOW

*attribute* · *module attribute*

```python
G19_FIT_LOW = FitValues(a1=[7.089476, -7.71186, -38.85941, -100.0], a2=[-0.6226781, -100.0, 3.081244e-09, 61.88812], s=[-5.988895, 6.940689])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L26-L30)

## particula.activity.bat_coefficients.G19_FIT_MID

*attribute* · *module attribute*

```python
G19_FIT_MID = FitValues(a1=[5.872214, -4.535007, -5.129327, -28.09232], a2=[-0.9740486, -100.0, 2.109751, -23.67683], s=[-1.219164, 4.742729])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L31-L35)

## particula.activity.bat_coefficients.INTERPOLATE_WATER_FIT

*attribute* · *module attribute*

```python
INTERPOLATE_WATER_FIT = 500
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L41-L41)

## particula.activity.bat_coefficients.LOWEST_ORGANIC_MOLE_FRACTION

*attribute* · *module attribute*

```python
LOWEST_ORGANIC_MOLE_FRACTION = 1e-12
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L42-L42)

## particula.activity.bat_coefficients.FitValues

*class*

```python
class FitValues(NamedTuple)
```

Bases: `NamedTuple`

Named tuple for the fit values for the activity model.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L18-L23)

### particula.activity.bat_coefficients.FitValues.a1

*attribute* · *instance attribute*

```python
a1: List[float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L21-L21)

### particula.activity.bat_coefficients.FitValues.a2

*attribute* · *instance attribute*

```python
a2: List[float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L22-L22)

### particula.activity.bat_coefficients.FitValues.s

*attribute* · *instance attribute*

```python
s: List[float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L23-L23)

## particula.activity.bat_coefficients.coefficients_c

*function*

```python
def coefficients_c(molar_mass_ratio: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], fit_values: List[float]) -> FloatArray
```

Compute fitted BAT coefficients from published parameters.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Water-to-organic molar mass ratio.
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon ratio.
- `fit_values` (`List[float]`) — Fit parameters (a1,a2,s) from ``FitValues``.

**Returns**

- (`FloatArray`) — Computed coefficient array matching ``fit_values`` shape.
- (`FloatArray`) — Scalar inputs return scalar; array inputs return ``np.ndarray``.

**Examples**

```pycon
>>> from particula.activity.bat_coefficients import (
...     coefficients_c,
...     G19_FIT_MID,
... )
>>> coefficients_c(
...     molar_mass_ratio=0.5,
...     oxygen2carbon=1.2,
...     fit_values=G19_FIT_MID.a1,
... )
0.025427174197017652
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L45-L82)
