# particula.dynamics.ions.attachment

*module*

LF2013 air-ion attachment at 298.15 K and 101325 Pa.

Coefficients describe conductive particles and are in m^3/s. Polynomial fits
are valid only over the per-charge radius intervals in the published tables.
They omit fluxes below 1e-15 m^3/s; omitted values are not extrapolated.

Reference: Lopez-Yglesias and Flagan (2013), Aerosol Science and Technology,
47, 688–704, https://doi.org/10.1080/02786826.2013.783684.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/attachment.py#L1-L1)

## particula.dynamics.ions.attachment.ATTACHMENT_CUTOFF

*attribute* · *module attribute*

```python
ATTACHMENT_CUTOFF = 1e-15
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/attachment.py#L18-L18)

## particula.dynamics.ions.attachment._tables

*function* · *cached*

```python
def _tables()
```

Load immutable published coefficients, preserving decimal precision.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/attachment.py#L21-L33)

## particula.dynamics.ions.attachment.attachment_coefficient

*function*

```python
def attachment_coefficient(radius, charge, polarity, *, omitted='reject')
```

Return LF2013 air-ion attachment rates in m^3/s.

**Parameters**

- `radius` — Aerosol radius in m, scalar or broadcastable array.
- `charge` — Integer aerosol charge in elementary charges, from -100 to 100.
- `polarity` — Free ion polarity, exactly -1 or +1.
- `omitted` (default: `'reject'`) — ``reject`` (default), ``lower`` or ``upper``. Bounds apply only to repulsive rates omitted below the table's lower radius at radii >= 0.2 nm. Lower returns zero, upper returns 1e-15 m^3/s. These are uncertainty bounds, not fitted rate estimates.

**Returns**

- Broadcast NumPy array of finite positive rates (or zero lower bounds).

**Raises**

- `ValueError` — If input is outside the supported fit/bound domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/attachment.py#L36-L99)
