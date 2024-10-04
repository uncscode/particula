# Phase Separation

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Phase Separation

> Auto-generated documentation for [particula.activity.phase_separation](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py) module.

## find_phase_sep_index

[Show source in phase_separation.py:55](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L55)

This function finds phase separation using activity>1 and
inflections in the activity curve data.
In physical systems activity can not be above one and
curve should be monotonic. Or else there will be phase separation.

#### Arguments

- `-` *activity_data* - A array of activity data.

#### Returns

- `dict` - A dictionary containing the following keys:
    - `-` *'phase_sep_activity'* - Phase separation via activity
        (1 if there is phase separation, 0 otherwise)
    - `-` *'phase_sep_curve'* - Phase separation via activity curvature
        (1 if there is phase separation, 0 otherwise)
    - `-` *'index_phase_sep_starts'* - Index where phase separation starts
    - `-` *'index_phase_sep_end'* - Index where phase separation ends

#### Signature

```python
def find_phase_sep_index(activity_data: ArrayLike) -> dict: ...
```



## find_phase_separation

[Show source in phase_separation.py:137](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L137)

This function checks for phase separation in each activity curve.

#### Arguments

- activity_water (np.array): A numpy array of water activity values.
- activity_org (np.array): A numpy array of organic activity values.

#### Returns

- `dict` - A dictionary containing the following keys:
    - `-` *'phase_sep_check'* - An integer indicating whether phase separation
            is present (1) or not (0).
    - `-` *'lower_seperation_index'* - The index of the lower separation point
            in the activity curve.
    - `-` *'upper_seperation_index'* - The index of the upper separation point in
            the activity curve.
    - `-` *'matching_upper_seperation_index'* - The index where the difference
            between activity_water_beta and match_a_w is greater than 0.
    - `-` *'lower_seperation'* - The value of water activity at the lower
            separation point.
    - `-` *'upper_seperation'* - The value of water activity at the upper
            separation point.
    - `-` *'matching_upper_seperation'* - The value of water activity at the
            matching upper separation point.

#### Signature

```python
def find_phase_separation(
    activity_water: ArrayLike, activity_org: ArrayLike
) -> dict: ...
```



## organic_water_single_phase

[Show source in phase_separation.py:23](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L23)

Convert the given molar mass ratio (MW water / MW organic) to a
and oxygen2carbon value were above is a single phase with water and below
phase separation is possible.

#### Arguments

- `-` *molar_mass_ratio* - The molar mass ratio with respect to water.

#### Returns

- The single phase cross point.

#### References

- Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
  Relative-humidity-dependent organic aerosol thermodynamics
  Via an efficient reduced-complexity model.
  Atmospheric Chemistry and Physics
  https://doi.org/10.5194/acp-19-13383-2019

#### Signature

```python
def organic_water_single_phase(
    molar_mass_ratio: Union[int, float, list, np.ndarray],
) -> np.ndarray: ...
```



## q_alpha

[Show source in phase_separation.py:233](https://github.com/uncscode/particula/blob/main/particula/activity/phase_separation.py#L233)

This function calculates the q_alpha value using a squeezed logistic
    function.

#### Arguments

- seperation_activity (np.array): A numpy array of values representing
    the separation activity.
- activities (np.array): A numpy array of activity values.

#### Returns

- `np.array` - The q_alpha value.

#### Notes

- The q_alpha value represents the transfer from
    q_alpha ~0 to q_alpha ~1.
- The function uses a sigmoid curve parameter to calculate the
    q_alpha value.

#### Signature

```python
def q_alpha(
    seperation_activity: Union[int, float, np.ndarray], activities: ArrayLike
) -> np.ndarray: ...
```
