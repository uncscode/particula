# particula.activity

*module*

Particula activity sub-package.

Provides Binary Activity Thermodynamics (BAT) helpers for organic–water
thermodynamics. Public APIs cover BAT activity coefficients, Gibbs mixing
weights, ratio conversions, and phase-separation diagnostics derived from
Gorkowski, Preston, and Zuend (2019). The package re-exports constants and
helper routines to simplify imports and keep validation consistent.

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
> Relative-humidity-dependent organic aerosol thermodynamics via an
> efficient reduced-complexity model. Atmospheric Chemistry and Physics.
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/__init__.py#L1-L1)

## particula.activity.G19_FIT_HIGH

*attribute* · *module attribute*

```python
G19_FIT_HIGH = FitValues(a1=[5.92155, -2.528295, -3.883017, -7.898128], a2=[-100.0, -100.0, 1.353916, -11.60145], s=[-0.07868187, 3.65086])
```

Re-exported from: `particula.activity.bat_coefficients`

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L36-L40)

## particula.activity.G19_FIT_LOW

*attribute* · *module attribute*

```python
G19_FIT_LOW = FitValues(a1=[7.089476, -7.71186, -38.85941, -100.0], a2=[-0.6226781, -100.0, 3.081244e-09, 61.88812], s=[-5.988895, 6.940689])
```

Re-exported from: `particula.activity.bat_coefficients`

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L26-L30)

## particula.activity.G19_FIT_MID

*attribute* · *module attribute*

```python
G19_FIT_MID = FitValues(a1=[5.872214, -4.535007, -5.129327, -28.09232], a2=[-0.9740486, -100.0, 2.109751, -23.67683], s=[-1.219164, 4.742729])
```

Re-exported from: `particula.activity.bat_coefficients`

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L31-L35)

## particula.activity.MIN_SPREAD_IN_AW

*attribute* · *module attribute*

```python
MIN_SPREAD_IN_AW: float = 10 ** -6
```

Re-exported from: `particula.activity.phase_separation`

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L31-L31)

## particula.activity.Q_ALPHA_AT_1PHASE_AW

*attribute* · *module attribute*

```python
Q_ALPHA_AT_1PHASE_AW: float = 0.99
```

Re-exported from: `particula.activity.phase_separation`

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L32-L32)

## particula.activity.FitValues

*class*

```python
class FitValues(NamedTuple)
```

Bases: `NamedTuple`

Re-exported from: `particula.activity.bat_coefficients`

Named tuple for the fit values for the activity model.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L18-L23)

### particula.activity.FitValues.a1

*attribute* · *instance attribute*

```python
a1: List[float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L21-L21)

### particula.activity.FitValues.a2

*attribute* · *instance attribute*

```python
a2: List[float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L22-L22)

### particula.activity.FitValues.s

*attribute* · *instance attribute*

```python
s: List[float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L23-L23)

## particula.activity.bat_activity_coefficients

*function*

```python
def bat_activity_coefficients(molar_mass_ratio: Union[float, NDArray[np.float64]], organic_mole_fraction: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], functional_group: Optional[Union[str, List[str]]] = None) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Re-exported from: `particula.activity.activity_coefficients`

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

## particula.activity.bat_blending_weights

*function*

```python
def bat_blending_weights(molar_mass_ratio: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]
```

Re-exported from: `particula.activity.bat_blending`

Estimate BAT blending weights for oxygen-rich regimes.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Ratio of water to organic matter. Must be positive.
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon ratio. Must be nonnegative.

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Blending weights for the low, mid, and high oxygen regions. Scalar inputs return shape (3,), array inputs return (n, 3) where n is the length of ``oxygen2carbon``.

**Examples**

```pycon
>>> from particula.activity.bat_blending import bat_blending_weights
>>> bat_blending_weights(molar_mass_ratio=0.5, oxygen2carbon=0.35)
array([0.00000000e+00, 9.11531178e-05, 9.99908847e-01])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_blending.py#L19-L62)

## particula.activity.coefficients_c

*function*

```python
def coefficients_c(molar_mass_ratio: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], fit_values: List[float]) -> FloatArray
```

Re-exported from: `particula.activity.bat_coefficients`

Compute fitted BAT coefficients from published parameters.

**Parameters**

- `molar_mass_ratio` (`Union[float, NDArray[np.float64]]`) — Water-to-organic molar mass ratio.
- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen-to-carbon ratio.
- `fit_values` (`List[float]`) — Fit parameters (a1,a2,s) from ``FitValues``.

**Returns**

- (`FloatArray`) — Computed coefficient array matching ``fit_values`` shape.
- (`FloatArray`) — Scalar inputs return scalar; array inputs return ``np.ndarray``.

**Examples**

```pycon
>>> from particula.activity.bat_coefficients import (
...     coefficients_c,
...     G19_FIT_MID,
... )
>>> coefficients_c(
...     molar_mass_ratio=0.5,
...     oxygen2carbon=1.2,
...     fit_values=G19_FIT_MID.a1,
... )
0.025427174197017652
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/bat_coefficients.py#L45-L82)

## particula.activity.convert_to_oh_equivalent

*function*

```python
def convert_to_oh_equivalent(oxygen2carbon: FloatArray, molar_mass_ratio: FloatArray, functional_group: Optional[Union[str, Sequence[str]]] = None) -> Tuple[FloatArray, FloatArray]
```

Re-exported from: `particula.activity.convert_functional_group`

Convert oxygen and molar mass ratios for BAT functional groups.

**Parameters**

- `oxygen2carbon` (`FloatArray`) — The oxygen to carbon ratio.
- `molar_mass_ratio` (`FloatArray`) — The molar mass ratio of water to organic matter.
- `functional_group` (`Optional[Union[str, Sequence[str]]]`) (default: `None`) — Optional functional group requiring adjustments. Supported values are ``None``, ``"alcohol"``, ``"ether"``, and ``"carboxylic_acid"``.

**Returns**

- (`FloatArray`) — Tuple containing the converted oxygen2carbon and molar_mass_ratio.
- (`FloatArray`) — Scalar inputs return scalars, array inputs return ``np.ndarray``.

**Raises**

- `ValueError` — If ``functional_group`` is not one of the supported values.

**Examples**

```pycon
>>> from particula.activity.convert_functional_group import (
...     convert_to_oh_equivalent,
... )
>>> convert_to_oh_equivalent(
...     oxygen2carbon=0.3,
...     molar_mass_ratio=0.1,
...     functional_group="carboxylic_acid",
... )
(2.3, 45.1)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/convert_functional_group.py#L50-L97)

## particula.activity.find_phase_sep_index

*function*

```python
def find_phase_sep_index(activity_data: Union[float, NDArray[np.float64]]) -> Dict[str, Union[int, float]]
```

Re-exported from: `particula.activity.phase_separation`

Detect phase separation using activity monotonicity and limits.

Identifies potential phase separation by checking for activities above one
and curvature inflections in the activity curve. Physical systems should
have activity ≤ 1 and monotonic behavior; violations indicate LLPS.

**Parameters**

- `activity_data` (`Union[float, NDArray[np.float64]]`) — Activity values, scalars or arrays promoted to NumPy arrays respecting the same dtype and shape.

**Returns**

- (`Dict[str, Union[int, float]]`) — Dictionary with detection flags and indices: phase_sep_activity: 1 if any activity exceeds 1, else 0. phase_sep_curve: 1 if curvature changes sign, else 0. index_phase_sep_starts: Index where separation starts. index_phase_sep_end: Index where separation ends.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L67-L158)

## particula.activity.find_phase_separation

*function*

```python
def find_phase_separation(activity_water: Union[float, NDArray[np.float64]], activity_org: Union[float, NDArray[np.float64]]) -> Dict[str, Union[int, float]]
```

Re-exported from: `particula.activity.phase_separation`

Check for phase separation across water and organic activity curves.

Applies :func:`find_phase_sep_index` to both water and organic activities,
then combines the detected indices into a consolidated separation report.

**Parameters**

- `activity_water` (`Union[float, NDArray[np.float64]]`) — Water activity values. Scalars are promoted to NumPy arrays.
- `activity_org` (`Union[float, NDArray[np.float64]]`) — Organic activity values. Scalars are promoted to NumPy arrays.

**Returns**

- (`Dict[str, Union[int, float]]`) — Dictionary containing: phase_sep_check: 1 if any phase separation is detected, else 0. lower_seperation_index: Lower separation index across both series. upper_seperation_index: Upper separation index across both series. matching_upper_seperation_index: Index where water activity crosses     the matched separation value. lower_seperation: Water activity at lower separation index. upper_seperation: Water activity at upper separation index. matching_upper_seperation: Water activity at the matched index.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L161-L252)

## particula.activity.from_molar_mass_ratio

*function*

```python
def from_molar_mass_ratio(molar_mass_ratio: Union[float, Iterable[float], NDArray[np.float64]], other_molar_mass: float = 18.01528) -> FloatArray
```

Re-exported from: `particula.activity.ratio`

Convert molar mass ratio to organic molar mass.

**Parameters**

- `molar_mass_ratio` (`Union[float, Iterable[float], NDArray[np.float64]]`) — Molar mass ratio (MW water / MW organic). Must be positive.
- `other_molar_mass` (`float`) (default: `18.01528`) — Reference molar mass (default water, 18.01528). Must be positive.

**Returns**

- (`FloatArray`) — Organic molar mass. Float for scalar input; ndarray for list/array
- (`FloatArray`) — input.

**Raises**

- `ValueError` — If molar_mass_ratio or other_molar_mass is not positive.

**Examples**

```pycon
>>> from_molar_mass_ratio(0.5)
9.00764
>>> from_molar_mass_ratio([0.5, 1.0])
array([ 9.00764, 18.01528])
>>> from_molar_mass_ratio(np.array([0.25]))
array([4.50382])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L46-L85)

## particula.activity.gibbs_free_energy

*function*

```python
def gibbs_free_energy(organic_mole_fraction: NDArray[np.float64], gibbs_mix: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

Re-exported from: `particula.activity.gibbs`

Compute ideal and real Gibbs free energy of mixing.

**Parameters**

- `organic_mole_fraction` (`NDArray[np.float64]`) — Mole fraction of the organic component. Values should be between 0 and 1.
- `gibbs_mix` (`NDArray[np.float64]`) — Gibbs free energy of mixing (non-ideal contribution).

**Returns**

- (`Tuple[NDArray[np.float64], NDArray[np.float64]]`) — Tuple containing: gibbs_ideal: Ideal Gibbs free energy of mixing. gibbs_real: Real Gibbs free energy of mixing (ideal + mix).

**Examples**

```pycon
>>> import numpy as np
>>> from particula.activity.gibbs import gibbs_free_energy
>>> x_org = np.array([0.2, 0.5, 0.8])
>>> g_mix = np.array([0.1, 0.2, 0.1])
>>> g_ideal, g_real = gibbs_free_energy(x_org, g_mix)
>>> g_ideal.shape, g_real.shape
((3,), (3,))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/gibbs.py#L12-L41)

## particula.activity.gibbs_mix_weight

*function*

```python
def gibbs_mix_weight(molar_mass_ratio: Union[float, NDArray[np.float64]], organic_mole_fraction: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], functional_group: Optional[str] = None) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Re-exported from: `particula.activity.gibbs_mixing`

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

## particula.activity.gibbs_of_mixing

*function*

```python
def gibbs_of_mixing(molar_mass_ratio: Union[float, NDArray[np.float64]], organic_mole_fraction: Union[float, NDArray[np.float64]], oxygen2carbon: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]], fit_dict: FitValues) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]
```

Re-exported from: `particula.activity.gibbs_mixing`

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

## particula.activity.organic_water_single_phase

*function*

```python
def organic_water_single_phase(molar_mass_ratio: Union[int, float, list, np.ndarray]) -> np.ndarray
```

Re-exported from: `particula.activity.phase_separation`

Compute single-phase O:C limit for organic–water mixtures.

Converts the molar mass ratio (:math:`MW_{water} / MW_{organic}`) to the
oxygen-to-carbon threshold above which mixtures remain single phase.
Below this threshold, liquid–liquid phase separation (LLPS) is possible.

**Parameters**

- `molar_mass_ratio` (`Union[int, float, list, np.ndarray]`) — Molar mass ratio with respect to water.

**Returns**

- (`np.ndarray`) — Single-phase O:C crossover point as a NumPy array.

**Examples**

```pycon
>>> organic_water_single_phase(molar_mass_ratio=0.1)
array(...)
```

> **References**
>
> Gorkowski et al. (2019).
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L35-L63)

## particula.activity.q_alpha

*function*

```python
def q_alpha(seperation_activity: Union[float, NDArray[np.float64]], activities: Union[float, NDArray[np.float64]]) -> np.ndarray
```

Re-exported from: `particula.activity.phase_separation`

Compute :math:`q_\alpha` transition using a squeezed logistic curve.

Maps activity values to a smooth transition between phase-separated and
single-phase regimes. The sigmoid is calibrated so that
:data:`Q_ALPHA_AT_1PHASE_AW` is reached at the single-phase boundary and
the transition width is bounded below by :data:`MIN_SPREAD_IN_AW`.

**Parameters**

- `seperation_activity` (`Union[float, NDArray[np.float64]]`) — Activity at which the mixture transitions between phases. Scalars or arrays are flattened internally.
- `activities` (`Union[float, NDArray[np.float64]]`) — Activity values to evaluate. Scalars are promoted to NumPy arrays before computing the logistic curve.

**Returns**

- (`np.ndarray`) — NumPy array of :math:`q_\alpha` values with the same shape as
- (`np.ndarray`) — ``activities``.

**Examples**

```pycon
>>> q_alpha(0.8, np.array([0.7, 0.8, 0.9]))
array(...)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L255-L306)

## particula.activity.to_molar_mass_ratio

*function*

```python
def to_molar_mass_ratio(molar_mass: Union[float, Iterable[float], NDArray[np.float64]], other_molar_mass: float = 18.01528) -> FloatArray
```

Re-exported from: `particula.activity.ratio`

Convert to molar mass ratio (MW water / MW organic).

**Parameters**

- `molar_mass` (`Union[float, Iterable[float], NDArray[np.float64]]`) — Molar mass of the organic compound. Must be positive.
- `other_molar_mass` (`float`) (default: `18.01528`) — Reference molar mass (default water, 18.01528). Must be positive.

**Returns**

- (`FloatArray`) — Molar mass ratio. Float for scalar input; ndarray for list/array input.

**Raises**

- `ValueError` — If molar_mass or other_molar_mass is not positive.

**Examples**

```pycon
>>> to_molar_mass_ratio(36.03)
0.5
>>> to_molar_mass_ratio([36.03, 18.01528])
array([0.5, 1. ])
>>> to_molar_mass_ratio(np.array([72.06]))
array([0.25])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/ratio.py#L13-L43)

**Modules**

- `particula.activity.activity_coefficients`
- `particula.activity.bat_blending`
- `particula.activity.bat_coefficients`
- `particula.activity.convert_functional_group`
- `particula.activity.gibbs`
- `particula.activity.gibbs_mixing`
- `particula.activity.phase_separation`
- `particula.activity.ratio`
- `particula.activity.water_activity`
