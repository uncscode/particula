# particula.activity.gibbs_mixing

*module*

Gibbs free energy of mixing for a binary mixture.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L1-L1)

## particula.activity.gibbs_mixing._calculate_gibbs_mix_single

*function*

```python
def _calculate_gibbs_mix_single(molar_mass_ratio: float, organic_mole_fraction: float, oxygen2carbon: float, density: float, weights: NDArray[np.float64]) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Compute Gibbs free energy for one compound using blended weights.

Uses the precomputed blending weights to select the appropriate fit
combinations (low/mid or mid/high O:C) from Gorkowski et al. (2019).

**Parameters**

- `molar_mass_ratio` (`float`) — Ratio of water to organic molar mass.
- `organic_mole_fraction` (`float`) — Mole fraction of the organic component.
- `oxygen2carbon` (`float`) — Oxygen-to-carbon atomic ratio.
- `density` (`float`) — Density of the organic compound in kg/m^3.
- `weights` (`NDArray[np.float64]`) — Three-element array of blending weights for low, mid, high O:C regions.

**Returns**

- `gibbs_mix` (`Union[float, NDArray[np.float64]]`) — Gibbs energy of mixing normalized by :math:`RT`.
- `derivative_gibbs` (`Union[float, NDArray[np.float64]]`) — Derivative of Gibbs energy with respect to the organic mole fraction, normalized by :math:`RT`.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L213-L283)

## particula.activity.gibbs_mixing.gibbs_mix_weight

*function*

```python
def gibbs_mix_weight(molar_mass_ratio: Union[float, NDArray[np.float64]], organic_mole_fraction: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], functional_group: Optional[str] = None) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Blend Gibbs mixing energies across O:C regimes for BAT activity.

The BAT model interpolates between Gorkowski et al. (2019) fits for low,
mid, and high oxygen-to-carbon ratios via :func:`bat_blending_weights`. The
weights ensure smooth transitions so downstream activity coefficient
calculations remain continuous regardless of single-compound inputs or
vectorized batches.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Ratio of water to organic molar mass (:math:`MW_{water} / MW_{organic}`). Dimensionless.
- `organic_mole_fraction` (`Union[float, NDArray[np.float64]]`) — Mole fraction of the organic component. Range: [0, 1].
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon atomic ratio of the organic compound.
- `density` (`Union[float, NDArray[np.float64]]`) — Density of the organic compound in kg/m^3.
- `functional_group` (`Optional[str]`) (default: `None`) — Optional functional group for OH-equivalent conversion (e.g., "alcohol", "carboxylic_acid", "ether").

**Returns**

- (`Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]`) — Tuple containing: gibbs_mix: Gibbs energy of mixing normalized by :math:`RT`     (dimensionless). derivative_gibbs: Derivative of Gibbs energy with respect to the     organic mole fraction, normalized by :math:`RT`.

**Examples**

```pycon
>>> from particula.activity import gibbs_mix_weight
>>> g_mix, dg_dx = gibbs_mix_weight(
...     molar_mass_ratio=0.09,
...     organic_mole_fraction=0.3,
...     oxygen2carbon=0.4,
...     density=1400.0,
... )
```

> **References**
>
> Gorkowski et al. (2019), Section 2.2 and SI S2.
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L121-L210)

## particula.activity.gibbs_mixing.gibbs_of_mixing

*function*

```python
def gibbs_of_mixing(molar_mass_ratio: Union[float, NDArray[np.float64]], organic_mole_fraction: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], fit_dict: FitValues) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Compute the Gibbs free energy of mixing for a single fit region.

This helper evaluates the Gibbs energy and its derivative for the low,
mid, or high O:C fit indicated by ``fit_dict``. The calculation follows
Gorkowski et al. (2019) with AIOMFAC-derived coefficients and normalizes
by :math:`RT` to remain dimensionless.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Ratio of water to organic molar mass (:math:`MW_{water} / MW_{organic}`). Dimensionless.
- `organic_mole_fraction` (`Union[float, NDArray[np.float64]]`) — Mole fraction of the organic component. Range: [0, 1].
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon ratio for the organic compound.
- `density` (`Union[float, NDArray[np.float64]]`) — Mixture density in kg/m^3 assuming water is the other phase.
- `fit_dict` (`FitValues`) — Fit constants defining the selected O:C region (low, mid, or high) from the BAT parameterization.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Tuple containing the Gibbs mixing energy and its derivative with
- (`Union[float, NDArray[np.float64]]`) — respect to the organic mole fraction. Both values are normalized by
- (`Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]`) — math:`RT` and share the shape of ``organic_mole_fraction``.

**Examples**

```pycon
>>> from particula.activity import gibbs_of_mixing
>>> g_mix, dg_dx = gibbs_of_mixing(
...     molar_mass_ratio=0.09,
...     organic_mole_fraction=0.3,
...     oxygen2carbon=0.4,
...     density=1400.0,
...     fit_dict=G19_FIT_LOW,
... )
```

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019), Equations 1-6 and
> Supplementary Information S1-S2. https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs_mixing.py#L29-L117)
