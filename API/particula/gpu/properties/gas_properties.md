# particula.gpu.properties.gas_properties

*module*

Warp GPU implementations of gas property functions.

These functions mirror the NumPy implementations in
``particula.gas.properties`` for use inside Warp kernels.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L1-L1)

## particula.gpu.properties.gas_properties._PI

*attribute* · *module attribute*

```python
_PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L9-L9)

## particula.gpu.properties.gas_properties.dynamic_viscosity_wp

*function*

```python
def dynamic_viscosity_wp(temperature: wp.float64, ref_viscosity: wp.float64, ref_temperature: wp.float64, sutherland_constant: wp.float64) -> wp.float64
```

Calculate dynamic viscosity with Sutherland's formula.

Port of
``particula.gas.properties.dynamic_viscosity.get_dynamic_viscosity``.

**Parameters**

- `temperature` (`wp.float64`) — Gas temperature [K].
- `ref_viscosity` (`wp.float64`) — Reference viscosity at ref_temperature [Pa·s].
- `ref_temperature` (`wp.float64`) — Reference temperature [K].
- `sutherland_constant` (`wp.float64`) — Sutherland constant [K].

**Returns**

- (`wp.float64`) — Dynamic viscosity [Pa·s].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L12-L38)

## particula.gpu.properties.gas_properties.molecule_mean_free_path_wp

*function*

```python
def molecule_mean_free_path_wp(molar_mass: wp.float64, temperature: wp.float64, pressure: wp.float64, dynamic_viscosity: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Calculate mean free path for a gas molecule.

Port of
``particula.gas.properties.mean_free_path.get_molecule_mean_free_path``.

**Parameters**

- `molar_mass` (`wp.float64`) — Molar mass of the gas [kg/mol].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `pressure` (`wp.float64`) — Gas pressure [Pa].
- `dynamic_viscosity` (`wp.float64`) — Dynamic viscosity [Pa·s].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Mean free path of a gas molecule [m].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L41-L67)

## particula.gpu.properties.gas_properties.partial_pressure_wp

*function*

```python
def partial_pressure_wp(concentration: wp.float64, molar_mass: wp.float64, temperature: wp.float64, gas_constant: wp.float64) -> wp.float64
```

Calculate partial pressure from concentration and temperature.

Port of ``particula.gas.properties.pressure_function.get_partial_pressure``.

**Parameters**

- `concentration` (`wp.float64`) — Gas concentration [kg/m³].
- `molar_mass` (`wp.float64`) — Molar mass [kg/mol].
- `temperature` (`wp.float64`) — Gas temperature [K].
- `gas_constant` (`wp.float64`) — Universal gas constant [J/(mol·K)].

**Returns**

- (`wp.float64`) — Partial pressure [Pa].

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/properties/gas_properties.py#L70-L90)
