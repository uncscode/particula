# particula.equilibria.partitioning

*module*

Equilibrium calculations for the particula thermodynamic model.

This module provides utilities to solve liquid-vapor partitioning using
activity coefficients and water activity. The implementation follows
Google-style documentation, explicit type hints, and small helpers to
improve readability and testability.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L1-L1)

## particula.equilibria.partitioning.EPSILON

*attribute* · *module attribute*

```python
EPSILON = 1e-16
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L20-L20)

## particula.equilibria.partitioning.PhaseOutput

*attribute* · *module attribute*

```python
PhaseOutput = tuple[NDArray[np.float64], NDArray[np.float64], float, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L22-L22)

## particula.equilibria.partitioning.SystemOutput

*attribute* · *module attribute*

```python
SystemOutput = tuple[float, float, NDArray[np.float64], float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L23-L23)

## particula.equilibria.partitioning._calculate_alpha_phase

*function*

```python
def _calculate_alpha_phase(c_j_liquid: NDArray[np.float64], q_ab: NDArray[np.float64], mass_fraction_water_ab: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, NDArray[np.float64]]
```

Wrapper to compute alpha-phase concentrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L73-L87)

## particula.equilibria.partitioning._calculate_beta_phase

*function*

```python
def _calculate_beta_phase(c_j_liquid: NDArray[np.float64], q_ab: NDArray[np.float64], mass_fraction_water_ab: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, NDArray[np.float64]]
```

Wrapper to compute beta-phase concentrations.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L90-L104)

## particula.equilibria.partitioning._calculate_cstar

*function*

```python
def _calculate_cstar(c_star_j_dry: NDArray[np.float64], gamma_organic_phase: NDArray[np.float64], q_phase: NDArray[np.float64], c_liquid_total: float, molar_mass: NDArray[np.float64], mass_weighted_molar_mass: float) -> NDArray[np.float64]
```

Calculate phase-specific C* with zero protection.

Implements:

.. math::

    C^*_j = C^*_{j,\mathrm{dry}} \cdot \gamma_{j,\mathrm{phase}} \cdot
    q_{\mathrm{phase}} \cdot \frac{C_{\mathrm{liq}}}
    {M_j \cdot \overline{M}_{\mathrm{phase}}}

**Parameters**

- `c_star_j_dry` (`NDArray[np.float64]`) — Dry saturation concentrations [µg/m³].
- `gamma_organic_phase` (`NDArray[np.float64]`) — Activity coefficients for the phase [-].
- `q_phase` (`NDArray[np.float64]`) — Phase fraction for the target phase [-].
- `c_liquid_total` (`float`) — Total liquid concentration across phases [µg/m³].
- `molar_mass` (`NDArray[np.float64]`) — Species molar masses [g/mol].
- `mass_weighted_molar_mass` (`float`) — Mass-weighted mean molar mass for the phase [g/mol].

**Returns**

- (`NDArray[np.float64]`) — Phase-specific C* array [µg/m³]; zeros if the mass-weighted molar mass
- (`NDArray[np.float64]`) — is zero to maintain numerical stability.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L107-L147)

## particula.equilibria.partitioning._calculate_phase

*function*

```python
def _calculate_phase(c_j_liquid: NDArray[np.float64], q_phase: NDArray[np.float64], mass_fraction_water_phase: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, NDArray[np.float64]]
```

Compute phase concentrations, aqueous content, and totals.

**Parameters**

- `c_j_liquid` (`NDArray[np.float64]`) — Liquid phase concentrations per species [µg/m³].
- `q_phase` (`NDArray[np.float64]`) — Phase fraction for the target phase (q_α or q_β) [-].
- `mass_fraction_water_phase` (`NDArray[np.float64]`) — Mass fraction of water per species in the target phase [-].

**Returns**

- (`NDArray[np.float64]`) — Tuple containing:
- `c_j_phase` (`NDArray[np.float64]`) — Organic concentrations per species in the phase [µg/m³].
- `c_j_aq_phase` (`float`) — Aqueous concentrations per species in the phase [µg/m³].
- `c_phase_total` (`float`) — Total concentration (organic + water) in the phase [µg/m³].
- `c_aq_total` (`NDArray[np.float64]`) — Total aqueous concentration in the phase [µg/m³].
- `denominator` (`tuple[NDArray[np.float64], NDArray[np.float64], float, float, NDArray[np.float64]]`) — Stabilized denominator ``(1 - mass_fraction_water)`` inverse with zero-fill when the mass fraction is one.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L26-L70)

## particula.equilibria.partitioning.get_properties_for_liquid_vapor_partitioning

*function*

```python
def get_properties_for_liquid_vapor_partitioning(water_activity_desired: NDArray[np.float64] | float, molar_mass: NDArray[np.float64], oxygen2carbon: NDArray[np.float64], density: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Get activity and phase properties for liquid-vapor partitioning.

**Parameters**

- `water_activity_desired` (`NDArray[np.float64] | float`) — Target water activity [-]; scalar or array of length n for species broadcast.
- `molar_mass` (`NDArray[np.float64]`) — Species molar masses [g/mol], shape (n,).
- `oxygen2carbon` (`NDArray[np.float64]`) — Oxygen-to-carbon ratios [-], shape (n,).
- `density` (`NDArray[np.float64]`) — Species densities [kg/m³], shape (n,).

**Returns**

- (`NDArray[np.float64]`) — Tuple ``(gamma_organic_ab, mass_fraction_water_ab, q_ab)`` each shaped
- (`NDArray[np.float64]`) — (n, 2). ``gamma_organic_ab[:, 1]`` and ``mass_fraction_water_ab[:, 1]``
- (`NDArray[np.float64]`) — are zero-filled when the beta phase is absent.

**Raises**

- `ValueError` — If input lengths are inconsistent or water activity cannot
be broadcast to the species dimension.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L462-L556)

## particula.equilibria.partitioning.liquid_vapor_obj_function

*function*

```python
def liquid_vapor_obj_function(e_j_partition_guess: NDArray[np.float64], c_star_j_dry: NDArray[np.float64], concentration_organic_matter: NDArray[np.float64], gamma_organic_ab: NDArray[np.float64], mass_fraction_water_ab: NDArray[np.float64], q_ab: NDArray[np.float64], molar_mass: NDArray[np.float64], *, error_only: bool = True) -> float | tuple[PhaseOutput, PhaseOutput, SystemOutput]
```

Objective function for liquid-vapor partitioning.

**Parameters**

- `e_j_partition_guess` (`NDArray[np.float64]`) — Partition coefficient guess per species [-].
- `c_star_j_dry` (`NDArray[np.float64]`) — Dry saturation concentrations [µg/m³].
- `concentration_organic_matter` (`NDArray[np.float64]`) — Organic aerosol mass concentration [µg/m³].
- `gamma_organic_ab` (`NDArray[np.float64]`) — Activity coefficients (shape: (n, 2)) [-].
- `mass_fraction_water_ab` (`NDArray[np.float64]`) — Water mass fractions (shape: (n, 2)) [-].
- `q_ab` (`NDArray[np.float64]`) — Phase fractions for alpha/beta (shape: (n, 2)) [-].
- `molar_mass` (`NDArray[np.float64]`) — Species molar masses [g/mol].
- `error_only` (`bool`) (default: `True`) — Whether to return only the scalar error (True) or detailed phase/system outputs (False).

**Returns**

- (`float | tuple[PhaseOutput, PhaseOutput, SystemOutput]`) — If ``error_only`` is True, a scalar error. Otherwise, a tuple:
- (`float | tuple[PhaseOutput, PhaseOutput, SystemOutput]`) — ``(alpha_phase_output, beta_phase_output, system_output)`` where each
- (`float | tuple[PhaseOutput, PhaseOutput, SystemOutput]`) — phase output is ``(c_j_liquid, c_j_aq, c_liquid_total, c_aq_total)`` and
- (`float | tuple[PhaseOutput, PhaseOutput, SystemOutput]`) — ``system_output`` is ``(c_liquid_total, c_liquid_total_water,
- (`float | tuple[PhaseOutput, PhaseOutput, SystemOutput]`) — e_j_partition_new, error_out)``.

**Raises**

- `ValueError` — If input shapes are incompatible.

**Examples**

```pycon
>>> import numpy as np
>>> from particula.equilibria import partitioning
>>> guess = np.full(3, 0.5)
>>> c_star = np.array([1e-3, 1e-2, 1e-1])
>>> c_om = np.array([1.0, 2.0, 3.0])
>>> gamma = np.ones((3, 2))
>>> mf_water = np.zeros((3, 2))
>>> q = np.full((3, 2), 0.5)
>>> molar_mass = np.full(3, 200.0)
>>> partitioning.liquid_vapor_obj_function(
...     guess, c_star, c_om, gamma, mf_water, q, molar_mass
... ) > 0
True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L178-L333)

## particula.equilibria.partitioning.liquid_vapor_partitioning

*function*

```python
def liquid_vapor_partitioning(c_star_j_dry: NDArray[np.float64], concentration_organic_matter: NDArray[np.float64], molar_mass: NDArray[np.float64], gamma_organic_ab: NDArray[np.float64], mass_fraction_water_ab: NDArray[np.float64], q_ab: NDArray[np.float64], partition_coefficient_guess: NDArray[np.float64] | None = None) -> tuple[PhaseOutput, PhaseOutput, SystemOutput, OptimizeResult]
```

Thermodynamic equilibrium between liquid and vapor phases.

**Parameters**

- `c_star_j_dry` (`NDArray[np.float64]`) — Dry saturation concentrations [µg/m³], shape (n,).
- `concentration_organic_matter` (`NDArray[np.float64]`) — Organic aerosol mass concentration [µg/m³], shape (n,).
- `molar_mass` (`NDArray[np.float64]`) — Species molar masses [g/mol], shape (n,).
- `gamma_organic_ab` (`NDArray[np.float64]`) — Activity coefficients, shape (n, 2) [-].
- `mass_fraction_water_ab` (`NDArray[np.float64]`) — Water mass fractions, shape (n, 2) [-].
- `q_ab` (`NDArray[np.float64]`) — Phase fractions for alpha/beta, shape (n, 2) [-].
- `partition_coefficient_guess` (`NDArray[np.float64] | None`) (default: `None`) — Optional initial partition coefficients (bounds [0, 1]); defaults to 0.5 for each species when None.

**Returns**

- (`PhaseOutput`) — Tuple ``(alpha, beta, system, fit_result)`` where ``alpha`` and ``beta``
- (`PhaseOutput`) — are phase outputs from :func:`liquid_vapor_obj_function`, ``system``
- (`SystemOutput`) — contains aggregate system values, and ``fit_result`` is the SciPy
- (`OptimizeResult`) — optimization result.

**Raises**

- `ValueError` — If array lengths are inconsistent or guess length mismatches
the number of species.

[View source](https://github.com/uncscode/particula/blob/main/particula/equilibria/partitioning.py#L336-L459)
