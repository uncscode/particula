# particula.dynamics.ions.mobility_attachment

*module*

Coulomb transport approximation with independently specified ion mobility.

Low-field, constant collision-cross-section scaling gives Z proportional to
sqrt(T)/P (Mason--Schamp with ideal-gas number density). This is not LF2013:
image forces, hydration and three-body trapping are omitted. The transition
fit is equation (5) of Gopalakrishnan and Hogan (2012),
https://doi.org/10.1103/PhysRevE.85.026410. For the mobility relation see the
primary interlaboratory study https://doi.org/10.1021/acs.analchem.7b01729.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/mobility_attachment.py#L1-L1)

## particula.dynamics.ions.mobility_attachment.mobility_attachment

*function*

```python
def mobility_attachment(radius, mass, charge, temperature, pressure, ion_mass, ion_mobility, ion_sign, ion_radius=2.5e-10)
```

Return broadcast ion-aerosol attachment coefficients [m^3/s].

**Parameters**

- `radius` — Aerosol volume-equivalent radius [m].
- `mass` — Aerosol total mass [kg].
- `charge` — Integer aerosol charge [elementary charges].
- `temperature` — Gas temperature [K].
- `pressure` — Air pressure [Pa].
- `ion_mass` — Singly charged ion mass [kg].
- `ion_mobility` — Mobility at 298.15 K, 101325 Pa [m^2/(V s)].
- `ion_sign` — +1 or -1.
- `ion_radius` (default: `2.5e-10`) — Ion contact radius [m], independent of mobility.

Coulomb potential ratios are clipped to [-200, 200], and kinetic
enhancement below 1e-80 is treated as zero, matching device guards.

**Returns**

- Broadcast nonnegative attachment coefficient array [m^3/s].

**Raises**

- `ValueError` — For invalid inputs or unrepresentable rates.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/ions/mobility_attachment.py#L27-L124)
