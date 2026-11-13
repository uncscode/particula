# particula.gas.properties.mean_free_path

*module*

Calculating the mean free path of air.

> **Long Description**
>
> The mean free path is the average distance traveled by a molecule
> between collisions with other molecules in a medium (air). Typical
> mean free path values in air are about 65 nm at 298 K and 101325 Pa.

> **References**
>
> - "Mean Free Path," Wikipedia, The Free Encyclopedia,
>   https://en.wikipedia.org/wiki/Mean_free_path

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/mean_free_path.py#L1-L1)

## particula.gas.properties.mean_free_path.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/mean_free_path.py#L28-L28)

## particula.gas.properties.mean_free_path.get_molecule_mean_free_path

*function*

```python
def get_molecule_mean_free_path(molar_mass: Union[float, NDArray[np.float64]] = MOLECULAR_WEIGHT_AIR, temperature: float = 298.15, pressure: float = 101325, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Calculate the mean free path of a gas molecule in air.

This function calculates λ based on the input conditions. If
dynamic_viscosity is not provided, it is computed via
get_dynamic_viscosity(temperature).

- λ = (2 × μ / P) / √(8 × M / (π × R × T))
    - λ is Mean free path [m].
    - μ is Dynamic viscosity [Pa·s].
    - P is Gas pressure [Pa].
    - M is Molar mass [kg/mol].
    - R is Universal gas constant [J/(mol·K)].
    - T is Gas temperature [K].

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) (default: `MOLECULAR_WEIGHT_AIR`) — The molar mass of the gas molecule [kg/mol].
- `temperature` (`float`) (default: `298.15`) — The temperature of the gas [K].
- `pressure` (`float`) (default: `101325`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa·s]. If None, it will be calculated based on the temperature.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Mean free path of the gas molecule in meters (m).

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_molecule_mean_free_path()
# Returns mean free path at ~298K and 101325Pa, ~6.5e-8 m
```

> **References**
>
> - "Mean Free Path," Wikipedia, The Free Encyclopedia.
>   https://en.wikipedia.org/wiki/Mean_free_path

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/mean_free_path.py#L31-L88)
