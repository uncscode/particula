# particula.particles.properties.convert_mass_concentration

*module*

Convert mass concentrations into mole, volume, and mass fractions.

This module provides helpers for transforming component-wise mass
concentrations into normalized composition fractions for one-dimensional and
two-dimensional arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L1-L1)

## particula.particles.properties.convert_mass_concentration._normalize_rows

*function*

```python
def _normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float64]
```

Normalize 2D rows while leaving zero-total rows as zeros.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L14-L19)

## particula.particles.properties.convert_mass_concentration.get_mass_fraction_from_mass

*function*

```python
def get_mass_fraction_from_mass(mass_concentrations: NDArray[np.float64]) -> NDArray[np.float64]
```

Convert mass concentrations to mass fractions.

The mass fraction of each component is computed by normalizing each
composition vector by its total mass concentration. One-dimensional inputs
return a single mass fraction vector, while two-dimensional inputs are
normalized row by row.

**Parameters**

- `mass_concentrations` (`NDArray[np.float64]`) — Component mass concentrations in kg/m^3.

**Returns**

- (`NDArray[np.float64]`) — Mass fractions with the same shape as ``mass_concentrations``. If a
- (`NDArray[np.float64]`) — total mass sum is zero, the corresponding output vector is all zeros.

**Raises**

- `ValueError` — If ``mass_concentrations`` is not one- or two-dimensional.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L126-L168)

## particula.particles.properties.convert_mass_concentration.get_mole_fraction_from_mass

*function*

```python
def get_mole_fraction_from_mass(mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]) -> NDArray[np.float64]
```

Convert mass concentrations to mole fractions.

The mole fraction of each component is computed from the component moles,
``mass_concentrations / molar_masses``, normalized by the total moles in
each composition vector. One-dimensional inputs return a single mole
fraction vector, while two-dimensional inputs are normalized row by row.

**Parameters**

- `mass_concentrations` (`NDArray[np.float64]`) — Component mass concentrations in kg/m^3.
- `molar_masses` (`NDArray[np.float64]`) — Component molar masses in kg/mol.

**Returns**

- (`NDArray[np.float64]`) — Mole fractions with the same shape as ``mass_concentrations``. If a
- (`NDArray[np.float64]`) — total mole sum is zero, the corresponding output vector is all zeros.

**Raises**

- `ValueError` — If ``mass_concentrations`` is not one- or two-dimensional.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L22-L71)

## particula.particles.properties.convert_mass_concentration.get_volume_fraction_from_mass

*function*

```python
def get_volume_fraction_from_mass(mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]) -> NDArray[np.float64]
```

Convert mass concentrations to volume fractions.

The volume fraction of each component is computed from component volumes,
``mass_concentrations / densities``, normalized by the total volume in each
composition vector. One-dimensional inputs return a single volume fraction
vector, while two-dimensional inputs are normalized row by row.

**Parameters**

- `mass_concentrations` (`NDArray[np.float64]`) — Component mass concentrations in kg/m^3.
- `densities` (`NDArray[np.float64]`) — Component material densities in kg/m^3.

**Returns**

- (`NDArray[np.float64]`) — Volume fractions with the same shape as ``mass_concentrations``. If a
- (`NDArray[np.float64]`) — total volume sum is zero, the corresponding output vector is all zeros.

**Raises**

- `ValueError` — If ``mass_concentrations`` is not one- or two-dimensional.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_mass_concentration.py#L74-L123)
