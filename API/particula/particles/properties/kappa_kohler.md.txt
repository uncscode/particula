# particula.particles.properties.kappa_kohler

*module*

Equilibrium activation of an insoluble or constant-κ dry particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kappa_kohler.py#L1-L1)

## particula.particles.properties.kappa_kohler._log_activity_penalty

*function*

```python
def _log_activity_penalty(t, log_kappa)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kappa_kohler.py#L7-L10)

## particula.particles.properties.kappa_kohler._peak

*function*

```python
def _peak(left, right, log_kappa, log_s)
```

Find a maximum on a bracket where the slope changes from + to −.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kappa_kohler.py#L13-L25)

## particula.particles.properties.kappa_kohler._threshold_scale

*function*

```python
def _threshold_scale(kappa, log_s)
```

Return critical diameter per unit Kelvin length for positive κ.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kappa_kohler.py#L28-L48)

## particula.particles.properties.kappa_kohler.get_kappa_kohler_critical_dry_diameter

*function*

```python
def get_kappa_kohler_critical_dry_diameter(kappa: ArrayLike, saturation_ratio: ArrayLike, kelvin_diameter: ArrayLike) -> float | NDArray[np.float64]
```

Invert the global maximum of the equilibrium κ-Köhler curve.

**Parameters**

- `kappa` (`ArrayLike`) — Dimensionless nonnegative hygroscopicity, constant during growth.
- `saturation_ratio` (`ArrayLike`) — Water vapor pressure ratio over flat water (> 1).
- `kelvin_diameter` (`ArrayLike`) — Positive diameter-basis Kelvin length in m, twice the Kelvin radius. Surface tension stays constant during growth.

**Returns**

- (`float | NDArray[np.float64]`) — Minimum dry diameter in m, as a float or a float64 array with the
- (`float | NDArray[np.float64]`) — NumPy broadcast shape. At zero κ the result is A / log(S).

**Raises**

- `ValueError` — For nonfinite, incompatible or out-of-domain inputs.
- `FloatingPointError` — If the result cannot be represented in float64.

> **References**
>
> Petters and Kreidenweis (2007), Eq. 6, doi:10.5194/acp-7-1961-2007.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kappa_kohler.py#L51-L98)
