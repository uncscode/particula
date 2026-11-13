# particula.activity.activity_coefficients

*module*

Compute activity coefficients for binary organic-water mixtures using BAT.

The Binary Activity Thermodynamics (BAT) model leverages AIOMFAC-derived
fits to describe thermodynamic activities of water and organics across realistic
O:C and concentration ranges. The helper optionally converts functional group
information to OH-equivalent form before blending Gibbs mixing weights and
translating them into activity coefficients.

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
> Relative-humidity-dependent organic aerosol thermodynamics via an efficient
> reduced-complexity model. Atmospheric Chemistry and Physics.
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/activity_coefficients.py#L1-L1)

## particula.activity.activity_coefficients.bat_activity_coefficients

*function*

```python
def bat_activity_coefficients(molar_mass_ratio: Union[float, NDArray[np.float64]], organic_mole_fraction: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], functional_group: Optional[Union[str, List[str]]] = None) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Calculate the activity coefficients for water-organic mixtures.

Uses the BAT (Binary Activity Thermodynamics) model to compute activity
coefficients for binary organic-water mixtures based on AIOMFAC-derived
fits. Optionally converts functional groups to OH-equivalent form to align
with the BAT parameterization.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Ratio of the molecular weight of water to the molecular weight of organic matter (dimensionless).
- `organic_mole_fraction` (`Union[float, NDArray[np.float64]]`) — Molar fraction of organic matter in the mixture. Range: [0, 1].
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon ratio in the organic compound.
- `density` (`Union[float, NDArray[np.float64]]`) — Density of the mixture, in kg/m^3.
- `functional_group` (`Optional[Union[str, List[str]]]`) (default: `None`) — Optional functional group(s) of the organic compound (e.g., "alcohol", "carboxylic_acid", "ether").

**Returns**

- (`Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]`) — Tuple of six values: activity_water: Thermodynamic activity of water (:math:`a_w`).     Range: [0, 1] for stable systems. activity_organic: Thermodynamic activity of the organic component     (:math:`a_{org}`). Range: [0, 1] for stable systems. mass_water: Mass fraction of water in the mixture. Range: [0, 1]. mass_organic: Mass fraction of organic; :math:`1 - mass_water`. gamma_water: Activity coefficient of water (:math:`\gamma_w`). gamma_organic: Activity coefficient of the organic component     (:math:`\gamma_{org}`).

**Examples**

```pycon
>>> from particula.activity import bat_activity_coefficients
>>> a_w, a_org, m_w, m_org, g_w, g_org = bat_activity_coefficients(
...     molar_mass_ratio=0.09,
...     organic_mole_fraction=0.3,
...     oxygen2carbon=0.4,
...     density=1400.0,
... )
```

> **References**
>
> Gorkowski et al. (2019), Equations 1-6 and SI S1-S2.
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/activity_coefficients.py#L29-L126)
