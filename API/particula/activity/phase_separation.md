# particula.activity.phase_separation

*module*

Phase separation utilities for organic–water systems.

Implements BAT (Binary Activity Thermodynamics) phase-separation helpers used
by the activity module to identify liquid–liquid phase separation (LLPS) and
compute transfer functions.

> **Constants**
>
> MIN_SPREAD_IN_AW: Minimum allowed spread in water activity when computing
>     the transition width for :func:`q_alpha`. Prevents numerical issues
>     in the logistic transfer when activity ranges are extremely small.
>     Value: 1e-6 (dimensionless).
> 
> Q_ALPHA_AT_1PHASE_AW: Target :math:`q_\alpha` value at the single-phase
>     activity boundary. Used to calibrate the sigmoid steepness in
>     :func:`q_alpha`. Value: 0.99 (dimensionless).

> **References**
>
> Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
> Relative-humidity-dependent organic aerosol thermodynamics via an
> efficient reduced-complexity model. Atmospheric Chemistry and Physics.
> https://doi.org/10.5194/acp-19-13383-2019

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L1-L1)

## particula.activity.phase_separation.MIN_SPREAD_IN_AW

*attribute* · *module attribute*

```python
MIN_SPREAD_IN_AW: float = 10 ** -6
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L31-L31)

## particula.activity.phase_separation.Q_ALPHA_AT_1PHASE_AW

*attribute* · *module attribute*

```python
Q_ALPHA_AT_1PHASE_AW: float = 0.99
```

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L32-L32)

## particula.activity.phase_separation.find_phase_sep_index

*function*

```python
def find_phase_sep_index(activity_data: Union[float, NDArray[np.float64]]) -> Dict[str, Union[int, float]]
```

Detect phase separation using activity monotonicity and limits.

Identifies potential phase separation by checking for activities above one
and curvature inflections in the activity curve. Physical systems should
have activity ≤ 1 and monotonic behavior; violations indicate LLPS.

**Parameters**

- `activity_data` (`Union[float, NDArray[np.float64]]`) — Activity values, scalars or arrays promoted to NumPy arrays respecting the same dtype and shape.

**Returns**

- (`Dict[str, Union[int, float]]`) — Dictionary with detection flags and indices: phase_sep_activity: 1 if any activity exceeds 1, else 0. phase_sep_curve: 1 if curvature changes sign, else 0. index_phase_sep_starts: Index where separation starts. index_phase_sep_end: Index where separation ends.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L67-L158)

## particula.activity.phase_separation.find_phase_separation

*function*

```python
def find_phase_separation(activity_water: Union[float, NDArray[np.float64]], activity_org: Union[float, NDArray[np.float64]]) -> Dict[str, Union[int, float]]
```

Check for phase separation across water and organic activity curves.

Applies :func:`find_phase_sep_index` to both water and organic activities,
then combines the detected indices into a consolidated separation report.

**Parameters**

- `activity_water` (`Union[float, NDArray[np.float64]]`) — Water activity values. Scalars are promoted to NumPy arrays.
- `activity_org` (`Union[float, NDArray[np.float64]]`) — Organic activity values. Scalars are promoted to NumPy arrays.

**Returns**

- (`Dict[str, Union[int, float]]`) — Dictionary containing: phase_sep_check: 1 if any phase separation is detected, else 0. lower_seperation_index: Lower separation index across both series. upper_seperation_index: Upper separation index across both series. matching_upper_seperation_index: Index where water activity crosses     the matched separation value. lower_seperation: Water activity at lower separation index. upper_seperation: Water activity at upper separation index. matching_upper_seperation: Water activity at the matched index.

[View source](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L161-L252)

## particula.activity.phase_separation.organic_water_single_phase

*function*

```python
def organic_water_single_phase(molar_mass_ratio: Union[int, float, list, np.ndarray]) -> np.ndarray
```

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

## particula.activity.phase_separation.q_alpha

*function*

```python
def q_alpha(seperation_activity: Union[float, NDArray[np.float64]], activities: Union[float, NDArray[np.float64]]) -> np.ndarray
```

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
