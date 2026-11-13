# particula.gas.properties.taylor_microscale_module

*module*

Taylor microscale module, for both the Lagrangian Taylor microscale time and
the Taylor microscale.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L1-L1)

## particula.gas.properties.taylor_microscale_module.get_lagrangian_taylor_microscale_time

*function*

```python
def get_lagrangian_taylor_microscale_time(kolmogorov_time: Union[float, NDArray[np.float64]], re_lambda: Union[float, NDArray[np.float64]], accel_variance: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the Lagrangian Taylor microscale time.

The Lagrangian Taylor microscale time (τ_T) represents the characteristic
time for the decay of turbulent velocity correlations. It provides insight
into the memory of turbulent fluid elements. It is given by:

- τ_T = τ_k * (2 R_λ / (15^(1/2) a_o))^(1/2)
    - τ_T is Lagrangian Taylor microscale time [s]
    - τ_k (kolmogorov_time) is Kolmogorov time scale [s]
    - R_λ (re_lambda) is Taylor-microscale Reynolds number [-]
    - a_o (accel_variance) is Normalized acceleration variance in isotropic
        turbulence [-]

**Parameters**

- `kolmogorov_time` (`Union[float, NDArray[np.float64]]`) — Kolmogorov time scale [s]
- `re_lambda` (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [-]
- `accel_variance` (`Union[float, NDArray[np.float64]]`) — Normalized acceleration variance in isotropic turbulence [-]

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_lagrangian_taylor_microscale_time(0.387, 500, 0.05)
# Output: 0.3872983346207417
```

Returns:
    Lagrangian Taylor microscale time [s]

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L13-L55)

## particula.gas.properties.taylor_microscale_module.get_taylor_microscale

*function*

```python
def get_taylor_microscale(fluid_rms_velocity: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the Taylor microscale.

The Taylor microscale (λ) represents an intermediate length scale in
turbulence, linking the dissipative and energy-containing ranges of
turbulence. It characterizes the smoothness of velocity fluctuations
in turbulent flows. It is given by:

- λ = u' * (15 ν² / ε)^(1/2)
    - λ is Taylor microscale [m]
    - u' (rms_velocity) is Fluid RMS fluctuation velocity [m/s]
    - v (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s]
    - ε (turbulent_dissipation) is Turbulent kinetic energy dissipation
        rate [m²/s³]

**Parameters**

- `fluid_rms_velocity` (`Union[float, NDArray[np.float64]]`) — Fluid RMS fluctuation velocity [m/s]
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m²/s]
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Turbulent kinetic energy dissipation rate [m²/s³]

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Taylor microscale [m]

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_taylor_microscale(0.35, 1.5e-5, 0.1)
# Output: 0.00021081851067789195
```

> **References**
>
> - https://en.wikipedia.org/wiki/Taylor_microscale

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L58-L105)

## particula.gas.properties.taylor_microscale_module.get_taylor_microscale_reynolds_number

*function*

```python
def get_taylor_microscale_reynolds_number(fluid_rms_velocity: Union[float, NDArray[np.float64]], taylor_microscale: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the Taylor-microscale Reynolds number (Re_λ).

The Taylor-scale micro Reynolds number is a dimensionless quantity used in
turbulence studies to characterize the relative importance of inertial and
viscous forces at the Taylor microscale.

- Re_λ = (u' λ) / ν
    - u' (fluid_rms_velocity) is Fluid (RMS) velocity fluctuation [m/s].
    - λ (taylor_microscale) is Taylor microscale [m].
    - ν (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s].

**Parameters**

- `fluid_rms_velocity` (`Union[float, NDArray[np.float64]]`) — Fluid RMS velocity fluctuation [m/s].
- `taylor_microscale` (`Union[float, NDArray[np.float64]]`) — Taylor microscale [m].
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m²/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [dimensionless].

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_taylor_microscale_reynolds_number(0.35, 0.00021, 1.5e-5)
# Output: 500.0
```

> **References**
>
> - https://en.wikipedia.org/wiki/Taylor_microscale

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L108-L149)
