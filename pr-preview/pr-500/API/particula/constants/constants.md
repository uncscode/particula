# Constants

[Particula Index](../README.md#particula-index) / [Particula](./index.md#particula) / Constants

> Auto-generated documentation for [particula.constants](https://github.com/uncscode/particula/blob/main/particula/constants.py) module.

#### Attributes

- `GAS_CONSTANT` - Gas constant in J mol^-1 K^-1 = m^2 kg mol^-1 s^-2 K^-1
  J = kg m^2 s^-2
  or (1*u.molar_gas_constant).to_base_units(): BOLTZMANN_CONSTANT * AVOGADRO_NUMBER

- `RELATIVE_PERMITTIVITY_AIR_ROOM` - Relative permittivity of air at approx.
  296.15 K and 101325 Pa and 40% RH
  See https://www.osti.gov/servlets/purl/1504063
  Previously known as the "dielectric constant"
  Often denoted as epsilon: 1.000530569

- `RELATIVE_PERMITTIVITY_AIR_STP` - At STP (273.15 K, 1 atm):
  see: https://en.wikipedia.org/wiki/Relative_permittivity: 1.00058986

- `RELATIVE_PERMITTIVITY_AIR` - select one of the two:: RELATIVE_PERMITTIVITY_AIR_ROOM

- `VACUUM_PERMITTIVITY` - Permittivity of free space in F/m
  Also known as the electric constant, permittivity of free space
  Often denoted by epsilon_0: 1 * u.vacuum_permittivity.to_base_units()

- `REF_VISCOSITY_AIR_STP` - These values are used to calculate the dynamic viscosity of air
  Here, REF temperature and viscosity are at STP:
  Standard temperature and pressure (273.15 K and 101325 Pa): 1.716e-05 * u.Pa * u.s
