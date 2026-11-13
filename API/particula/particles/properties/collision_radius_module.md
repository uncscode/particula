# particula.particles.properties.collision_radius_module

*module*

Radius of collision module for non-spherical particles. Based on the
compiled models in Qian et al. (2022).

Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
Effects of agglomerate characteristics on their collision kernels in
the free molecular regime. Journal of Aerosol Science, 159.
https://doi.org/10.1016/j.jaerosci.2021.10586

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L1-L1)

## particula.particles.properties.collision_radius_module.get_collision_radius_mg1988

*function*

```python
def get_collision_radius_mg1988(gyration_radius: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the mg1988 model.

The collision radius (R_c) is set equal to the radius of gyration (R_g):

- R_c = R_g

**Parameters**

- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration of the particle (m).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_mg1988(1.5)
# 1.5
```

> **References**
>
> - Mulholland, G. W., Mountain, R. D., Samson, R. J., & Ernst, M. H.
> (1988). "Cluster Size Distribution for Free Molecular Agglomeration."
>   Energy and Fuels, 2(4). https://doi.org/10.1021/ef00010a014

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L18-L50)

## particula.particles.properties.collision_radius_module.get_collision_radius_mzg2002

*function*

```python
def get_collision_radius_mzg2002(gyration_radius: Union[NDArray[np.float64], float], fractal_prefactor: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the mzg2002 model.

The collision radius (R_c) is given by the empirical relation:

- R_c = 1.037 × (k₀^0.077) × R_g
    - R_c is the collision radius (m).
    - k₀ is the fractal prefactor (dimensionless).
    - R_g is the radius of gyration (m).

**Parameters**

- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration of the particle (m).
- `fractal_prefactor` (`Union[NDArray[np.float64], float]`) — Fractal prefactor of particle (dimensionless).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_mzg2002(1.5, 1.2)
# 1.577...
```

> **References**
>
> - Zurita-Gotor, M., & Rosner, D. E. (2002). "Effective diameters for
>   collisions of fractal-like aggregates: Recommendations for improved
>   aerosol coagulation frequency predictions." Journal of Colloid and
>   Interface Science, 255(1).
>   https://doi.org/10.1006/jcis.2002.8634

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L97-L137)

## particula.particles.properties.collision_radius_module.get_collision_radius_sr1992

*function*

```python
def get_collision_radius_sr1992(gyration_radius: Union[NDArray[np.float64], float], fractal_dimension: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the sr1992 model.

This model includes the fractal dimension (d_f). The collision radius
(R_c) is:

- R_c = √((d_f + 2) / 3) × R_g
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - R_g is the radius of gyration (m).

**Parameters**

- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration of the particle (m).
- `fractal_dimension` (`Union[NDArray[np.float64], float]`) — Fractal dimension of the particle (dimensionless).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_sr1992(1.5, 1.2)
# 1.8371173...
```

> **References**
>
> - Rogak, S. N., & Flagan, R. C. (1992). "Coagulation of aerosol
>   agglomerates in the transition regime." Journal of Colloid and
>   Interface Science, 151(1), 203-224.
>   https://doi.org/10.1016/0021-9797(92)90252-H

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L53-L94)

## particula.particles.properties.collision_radius_module.get_collision_radius_tt2012

*function*

```python
def get_collision_radius_tt2012(fractal_dimension: float, number_of_particles: float, gyration_radius: Union[NDArray[np.float64], float], radius_monomer: float) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the tt2012 model.

This function uses fitting parameters α₁, α₂ based on the fractal
dimension (d_f) and number of monomers (N). The collision radius
(R_c) is derived in multiple steps, ultimately returning:

- R_c = (radius_s_ii) / 2

**Parameters**

- `fractal_dimension` (`float`) — Fractal dimension of the particle (dimensionless).
- `number_of_particles` (`float`) — Number of monomers in the aggregate.
- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration of the particle (m).
- `radius_monomer` (`float`) — Radius of the monomer (m).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_tt2012(2.5, 100, 1.5, 0.1)
# 2.034...
```

> **References**
>
> - Thajudeen, T., Gopalakrishnan, R., & Hogan, C. J. (2012). "The
>   collision rate of nonspherical particles and aggregates for all
>   diffusive knudsen numbers." Aerosol Science and Technology, 46(11).
>   https://doi.org/10.1080/02786826.2012.701353

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L140-L193)

## particula.particles.properties.collision_radius_module.get_collision_radius_wq2022_rg

*function*

```python
def get_collision_radius_wq2022_rg(gyration_radius: Union[NDArray[np.float64], float], radius_monomer: float) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the wq2022_rg model.

This function uses a fitted model based on the ratio (R_g / rₘ).
The collision radius (R_c) is:

- R_c = (A × (R_g / rₘ) + B) × rₘ
    - R_c is the collision radius (m).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B are empirical coefficients from Qian et al. (2022).

**Parameters**

- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration of the particle (m).
- `radius_monomer` (`float`) — Monomer radius (m).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg(1.5, 0.1)
# 1.50...
```

> **References**
>
> - Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
>   "Effects of agglomerate characteristics on their collision kernels in
>   the free molecular regime." Journal of Aerosol Science, 159.
>   https://doi.org/10.1016/j.jaerosci.2021.105868

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L196-L240)

## particula.particles.properties.collision_radius_module.get_collision_radius_wq2022_rg_df

*function*

```python
def get_collision_radius_wq2022_rg_df(fractal_dimension: Union[NDArray[np.float64], float], gyration_radius: Union[NDArray[np.float64], float], radius_monomer: float) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the wq2022_rg_df model.

This function uses a fitted model based on fractal dimension (d_f), ratio
(R_g / rₘ), and empirical coefficients. The collision radius (R_c) is:

- R_c = (A × d_f^B × (R_g / rₘ) + C) × rₘ
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B, C are empirical coefficients from Qian et al. (2022).

**Parameters**

- `fractal_dimension` (`Union[NDArray[np.float64], float]`) — Fractal dimension of particle (dimensionless).
- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration of the particle (m).
- `radius_monomer` (`float`) — Monomer radius (m).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg_df(2.5, 1.5, 0.1)
# 1.66...
```

> **References**
>
> - Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
>   "Effects of agglomerate characteristics on their collision kernels in
>   the free molecular regime." Journal of Aerosol Science, 159.
>   https://doi.org/10.1016/j.jaerosci.2021.105868

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L243-L294)

## particula.particles.properties.collision_radius_module.get_collision_radius_wq2022_rg_df_k0

*function*

```python
def get_collision_radius_wq2022_rg_df_k0(fractal_dimension: float, fractal_prefactor: float, gyration_radius: Union[NDArray[np.float64], float], radius_monomer: float) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the wq2022_rg_df_k0 model.

This function uses a fitted expression depending on fractal dimension
(d_f), fractal prefactor (k₀), and ratio (R_g / rₘ). The collision
radius (R_c) is:

- R_c = (A × d_f^B × k₀^C × (R_g / rₘ) + D × k₀ + E) × rₘ
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - k₀ is the fractal prefactor (dimensionless).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B, C, D, E are empirical coefficients from Qian et al. (2022).

**Parameters**

- `fractal_dimension` (`float`) — Fractal dimension of particle (dimensionless).
- `fractal_prefactor` (`float`) — Fractal prefactor of particle (dimensionless).
- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration (m).
- `radius_monomer` (`float`) — Monomer radius (m).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg_df_k0(2.5, 1.2, 1.5, 0.1)
# 1.83...
```

> **References**
>
> - Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
>   "Effects of agglomerate characteristics on their collision kernels in
>   the free molecular regime." Journal of Aerosol Science, 159.
>   https://doi.org/10.1016/j.jaerosci.2021.105868

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L297-L355)

## particula.particles.properties.collision_radius_module.get_collision_radius_wq2022_rg_df_k0_a13

*function*

```python
def get_collision_radius_wq2022_rg_df_k0_a13(fractal_dimension: float, fractal_prefactor: float, shape_anisotropy: float, gyration_radius: Union[NDArray[np.float64], float], radius_monomer: float) -> Union[NDArray[np.float64], float]
```

Calculate the collision radius using the wq2022_rg_df_k0_a13 model.

This function uses a fitted expression depending on fractal dimension
(d_f), fractal prefactor (k₀), shape anisotropy (A₁₃), and ratio
(R_g / rₘ). The collision radius (R_c) is:

- R_c = (A × d_f^B × k₀^C × (R_g / rₘ) + D × k₀ + E × A₁₃ + F) × rₘ
    - R_c is the collision radius (m).
    - d_f is the fractal dimension (dimensionless).
    - k₀ is the fractal prefactor (dimensionless).
    - A₁₃ is the shape anisotropy parameter (dimensionless).
    - R_g is the radius of gyration (m).
    - rₘ is the monomer radius (m).
    - A, B, C, D, E, F are empirical coefficients from Qian et al. (2022).

**Parameters**

- `fractal_dimension` (`float`) — Fractal dimension of particle (dimensionless).
- `fractal_prefactor` (`float`) — Fractal prefactor of particle (dimensionless).
- `shape_anisotropy` (`float`) — Shape anisotropy parameter (dimensionless, A₁₃).
- `gyration_radius` (`Union[NDArray[np.float64], float]`) — Radius of gyration (m).
- `radius_monomer` (`float`) — Monomer radius (m).

**Returns**

- (`Union[NDArray[np.float64], float]`) — Collision radius of the particle (m).

**Examples**

``` py title="Example"
import particula as par
par.particles.get_collision_radius_wq2022_rg_df_k0_a13(
    2.5, 1.2, 1.82, 1.5, 0.1
)
# 1.82...
```

> **References**
>
> - Qian, W., Kronenburg, A., Hui, X., Lin, Y., & Karsch, M. (2022).
>   "Effects of agglomerate characteristics on their collision kernels in
>   the free molecular regime." Journal of Aerosol Science, 159.
>   https://doi.org/10.1016/j.jaerosci.2021.105868

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/collision_radius_module.py#L358-L423)
