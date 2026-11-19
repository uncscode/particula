# particula.particles.properties.bipolar_charge

*module*

Equilibrium charge fractions of submicron particles in bipolar air ions.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/bipolar_charge.py#L1-L1)

## particula.particles.properties.bipolar_charge._COEFFICIENTS

*attribute* · *module attribute*

```python
_COEFFICIENTS = np.array([[-26.3328, 35.9044, -21.4608, 7.0867, -1.3088, 0.1051], [-2.3197, 0.6175, 0.6201, -0.1105, -0.126, 0.0297], [-0.0003, -0.1014, 0.3073, -0.3372, 0.1023, -0.0105], [-2.3484, 0.6044, 0.48, 0.0013, -0.1553, 0.032], [-44.4756, 79.3772, -62.89, 26.4492, -5.748, 0.5049]])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/bipolar_charge.py#L7-L15)

## particula.particles.properties.bipolar_charge.get_bipolar_charge_probability

*function*

```python
def get_bipolar_charge_probability(particle_diameter: ArrayLike, charge: ArrayLike) -> np.float64 | NDArray[np.float64]
```

Estimate the fraction carrying a signed charge in bipolar equilibrium.

Evaluate the Wiedensohler polynomial in log10(diameter / nm). Each fit
is independent. A sum over selected states may differ from one. Sampling
a population requires choosing the charge tails and normalizing it.

**Parameters**

- `particle_diameter` (`ArrayLike`) — Mobility diameter in metres, 1–1000 nm for neutral or singly charged particles and 20–1000 nm for ±2.
- `charge` (`ArrayLike`) — Signed number of elementary charges, one of -2, -1, 0, 1, 2. Diameter and charge follow NumPy broadcasting rules.

**Returns**

- (`np.float64 | NDArray[np.float64]`) — Dimensionless fitted fraction for each diameter and charge pair.

**Raises**

- `ValueError` — A diameter or charge lies outside its fitted domain.

> **References**
>
> Wiedensohler (1988), https://doi.org/10.1016/0021-8502(88)90278-9.
> Wiedensohler et al. (2012), Table A7,
> https://doi.org/10.5194/amt-5-657-2012.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/bipolar_charge.py#L18-L58)
