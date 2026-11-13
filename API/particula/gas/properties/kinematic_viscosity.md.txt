# particula.gas.properties.kinematic_viscosity

*module*

Kinematic viscosity for fluids.

> **Long Description**
>
> The kinematic viscosity (ν) is the ratio of the dynamic viscosity (μ)
> to the density (ρ).

> **Equation**
>
> - ν = μ / ρ

> **Where**
>
> - ν : Kinematic viscosity [m²/s].
> - μ : Dynamic viscosity [Pa·s].
> - ρ : Fluid density [kg/m³].

> **References**
>
> - "Viscosity Conversion Formula," Wolfram Formula Repository,
>   https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula
> - Wikipedia contributors, "Viscosity," Wikipedia,
>   https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L1-L1)

## particula.gas.properties.kinematic_viscosity.get_kinematic_viscosity

*function*

```python
def get_kinematic_viscosity(dynamic_viscosity: float, fluid_density: float) -> float
```

Calculate the kinematic viscosity of a fluid.

The function calculates ν by dividing the dynamic viscosity (μ)
by the fluid density (ρ).

- ν = μ / ρ
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

**Parameters**

- `dynamic_viscosity` (`float`) — Dynamic viscosity of the fluid [Pa·s].
- `fluid_density` (`float`) — Density of the fluid [kg/m³].

**Returns**

- (`float`) — The kinematic viscosity [m²/s].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_kinematic_viscosity(1.8e-5, 1.2)
# Output: ~1.5e-5
```

> **References**
>
> - "Viscosity Conversion Formula," Wolfram Formula Repository.
>   https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L31-L64)

## particula.gas.properties.kinematic_viscosity.get_kinematic_viscosity_via_system_state

*function*

```python
def get_kinematic_viscosity_via_system_state(temperature: float, fluid_density: float, reference_viscosity: float = REF_VISCOSITY_AIR_STP, reference_temperature: float = REF_TEMPERATURE_STP) -> float
```

Calculate the kinematic viscosity of air by first computing its dynamic
viscosity.

This function uses get_dynamic_viscosity(...) and divides by the given
fluid_density to get the kinematic viscosity.

- ν = μ / ρ
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

> **Where**
>
> - ν is Kinematic viscosity [m²/s].
> - μ is Dynamic viscosity [Pa·s].
> - ρ is Fluid density [kg/m³].

**Parameters**

- `temperature` (`float`) — Desired air temperature [K]. Must be > 0.
- `fluid_density` (`float`) — Density of the fluid [kg/m³].
- `reference_viscosity` (`float`) (default: `REF_VISCOSITY_AIR_STP`) — Reference dynamic viscosity [Pa·s].
- `reference_temperature` (`float`) (default: `REF_TEMPERATURE_STP`) — Reference temperature [K].

**Returns**

- (`float`) — The kinematic viscosity of air [m²/s].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_kinematic_viscosity_via_system_state(300, 1.2)
# Output: ~1.5e-5
```

> **References**
>
> - "Sutherland's Formula," Wolfram Formula Repository,
>   https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L67-L117)
