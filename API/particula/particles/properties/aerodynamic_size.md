# particula.particles.properties.aerodynamic_size

*module*

Module for aerodynamic size and shape factor of a particle in a fluid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py#L1-L1)

## particula.particles.properties.aerodynamic_size.AERODYNAMIC_SHAPE_FACTOR_DICT

*attribute* · *module attribute*

```python
AERODYNAMIC_SHAPE_FACTOR_DICT = {'sphere': 1.0, 'cube': 1.08, 'cylinder_avg_aspect_2': 1.1, 'cylinder_avg_aspect_5': 1.35, 'cylinder_avg_aspect_10': 1.68, 'spheres_cluster_3': 1.15, 'spheres_cluster_4': 1.17, 'bituminous_coal': 1.08, 'quartz': 1.36, 'sand': 1.57, 'talc': 1.88}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py#L10-L22)

## particula.particles.properties.aerodynamic_size.get_aerodynamic_length

*function*

```python
def get_aerodynamic_length(physical_length: Union[float, NDArray[np.float64]], physical_slip_correction_factor: Union[float, NDArray[np.float64]], aerodynamic_slip_correction_factor: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], reference_density: float = 1000, aerodynamic_shape_factor: float = 1.0) -> Union[float, NDArray[np.float64]]
```

Calculate the aerodynamic length scale of a particle for a given shape.

The aerodynamic length (d_a) is determined by:

- d_a = d_p × √( (C_p / C_a) × (ρ / (ρ₀ × χ)) )
    - d_a is the aerodynamic size (m).
    - d_p is the physical size (m).
    - C_p is the slip correction factor for the physical size.
    - C_a is the slip correction factor for the aerodynamic size.
    - ρ is the particle density (kg/m³).
    - ρ₀ is the reference density (kg/m³).
    - χ is the shape factor (dimensionless).

**Parameters**

- `physical_length` (`Union[float, NDArray[np.float64]]`) — Physical length scale of the particle (m).
- `physical_slip_correction_factor` (`Union[float, NDArray[np.float64]]`) — Slip correction factor for the particle's physical size (dimensionless).
- `aerodynamic_slip_correction_factor` (`Union[float, NDArray[np.float64]]`) — Slip correction factor for the particle's aerodynamic size (dimensionless).
- `density` (`Union[float, NDArray[np.float64]]`) — Density of the particle in kg/m³.
- `reference_density` (`float`) (default: `1000`) — Reference density in kg/m³, typically water (1000 by default).
- `aerodynamic_shape_factor` (`float`) (default: `1.0`) — Shape factor (dimensionless, 1.0 for spheres).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Aerodynamic length scale (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_aerodynamic_length(
    physical_length=0.00005,
    physical_slip_correction_factor=1.1,
    aerodynamic_slip_correction_factor=1.0,
    density=1000,
    reference_density=1000,
    aerodynamic_shape_factor=1.0,
)
# Output: ...
```

> **References**
>
> - "Aerosol: Aerodynamic diameter," Wikipedia,
>   https://en.wikipedia.org/wiki/Aerosol#Aerodynamic_diameter
> - Hinds, W.C. (1998). Aerosol Technology: Properties, behavior, and
>   measurement of airborne particles (2nd ed.). Wiley-Interscience.
>   (pp. 51–53, Section 3.6).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py#L25-L94)

## particula.particles.properties.aerodynamic_size.get_aerodynamic_shape_factor

*function*

```python
def get_aerodynamic_shape_factor(shape_key: str) -> float
```

Retrieve the aerodynamic shape factor for a given particle shape.

The shape factor (χ) accounts for non-sphericity in aerodynamic
calculations. For spheres, χ=1.0. Larger values indicate more deviation
from spherical shape.

**Parameters**

- `shape_key` (`str`) — String representing the particle's shape (e.g. "sphere", "sand").

**Returns**

- (`float`) — The shape factor (dimensionless).

**Examples**

``` py title="Example"
shape_factor = get_aerodynamic_shape_factor("sand")
# shape_factor = 1.57
```

**Raises**

- `ValueError` — If the shape key is not found in the predefined
dictionary.

> **References**
>
> - Hinds, W.C. (1998). Aerosol Technology: Properties, behavior, and
>   measurement of airborne particles (2nd ed.). Wiley-Interscience.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_size.py#L97-L133)
