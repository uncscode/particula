# Binary Activity

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Activity](./index.md#activity) / Binary Activity

> Auto-generated documentation for [particula.activity.binary_activity](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py) module.

#### Attributes

- `FIT_LOW` - the fit values for the activity model: {'a1': [7.089476, -7.71186, -38.85941, -100.0], 'a2': [-0.6226781, -100.0, 3.081244e-09, 61.88812], 's': [-5.988895, 6.940689]}

- `INTERPOLATE_WATER_FIT` - interpolation points, could be done smarter: 500


## activity_coefficients

[Show source in binary_activity.py:34](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L34)

Calculate the activity coefficients for water and organic matter in
organic-water mixtures.

#### Arguments

- `-` *molar_mass_ratio* - Ratio of the molecular weight of water to the
    molecular weight of organic matter.
- `-` *organic_mole_fraction* - Molar fraction of organic matter in the
    mixture.
- `-` *oxygen2carbon* - Oxygen to carbon ratio in the organic compound.
- `-` *density* - Density of the mixture.
- `-` *functional_group* - Optional functional group(s) of the organic
    compound, if applicable.

#### Returns

A tuple containing the activity of water, activity
of organic matter, mass fraction of water, and mass
fraction of organic matter, gamma_water (activity coefficient),
and gamma_organic (activity coefficient).

#### Signature

```python
def activity_coefficients(
    molar_mass_ratio: ArrayLike,
    organic_mole_fraction: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
    functional_group=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
```



## bat_blending_weights

[Show source in binary_activity.py:335](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L335)

Function to estimate the blending weights for the BAT model.

#### Arguments

- `-` *molar_mass_ratio* - The molar mass ratio of water to organic
    matter.
- `-` *oxygen2carbon* - The oxygen to carbon ratio.

#### Returns

- blending_weights : List of blending weights for the BAT model
in the low, mid, and high oxygen2carbon regions.

#### Signature

```python
def bat_blending_weights(
    molar_mass_ratio: ArrayLike, oxygen2carbon: ArrayLike
) -> np.ndarray: ...
```



## biphasic_water_activity_point

[Show source in binary_activity.py:240](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L240)

This function computes the biphasic to single phase
water activity (RH*100).

#### Arguments

- `-` *oxygen2carbon* - The oxygen to carbon ratio.
- `-` *hydrogen2carbon* - The hydrogen to carbon ratio.
- `-` *molar_mass_ratio* - The molar mass ratio of water to organic
    matter.
- `-` *functional_group* - Optional functional group(s) of the organic
    compound, if applicable.

#### Returns

- `-` *np.array* - The RH cross point array.

#### Signature

```python
def biphasic_water_activity_point(
    oxygen2carbon: ArrayLike,
    hydrogen2carbon: ArrayLike,
    molar_mass_ratio: ArrayLike,
    functional_group: Optional[Union[list[str], str]] = None,
) -> np.ndarray: ...
```



## coefficients_c

[Show source in binary_activity.py:398](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L398)

Coefficients for activity model, see Gorkowski (2019). equation S1 S2.

#### Arguments

- `-` *molar_mass_ratio* - The molar mass ratio of water to organic
    matter.
- `-` *oxygen2carbon* - The oxygen to carbon ratio.
- `-` *fit_values* - The fit values for the activity model.

#### Returns

- `-` *np.ndarray* - The coefficients for the activity model.

#### Signature

```python
def coefficients_c(
    molar_mass_ratio: ArrayLike, oxygen2carbon: ArrayLike, fit_values: ArrayLike
) -> np.ndarray: ...
```



## convert_to_oh_equivalent

[Show source in binary_activity.py:314](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L314)

just a pass through now, but will
add the oh equivalent conversion

#### Signature

```python
def convert_to_oh_equivalent(
    oxygen2carbon: ArrayLike,
    molar_mass_ratio: ArrayLike,
    functional_group: Optional[Union[list[str], str]] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## fixed_water_activity

[Show source in binary_activity.py:425](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L425)

Calculate the activity coefficients of water and organic matter in
organic-water mixtures.

This function assumes a fixed water activity value (e.g., RH = 75%
corresponds to 0.75 water activity in equilibrium).
It calculates the activity coefficients for different phases and
determines phase separations if they occur.

#### Arguments

- `water_activity` *ArrayLike* - An array of water activity values.
- `molar_mass_ratio` *ArrayLike* - Array of molar mass ratios of the components.
- `oxygen2carbon` *ArrayLike* - Array of oxygen-to-carbon ratios.
- `density` *ArrayLike* - Array of densities of the mixture.

#### Returns

- `Tuple` - A tuple containing the activity coefficients for alpha and beta
        phases, and the alpha phase mole fraction.
       If no phase separation occurs, the beta phase values are None.

#### Signature

```python
def fixed_water_activity(
    water_activity: ArrayLike,
    molar_mass_ratio: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
) -> Tuple: ...
```



## gibbs_mix_weight

[Show source in binary_activity.py:158](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L158)

Gibbs free energy of mixing, see Gorkowski (2019), with weighted
oxygen2carbon regions. Only can run one compound at a time.

#### Arguments

- `-` *molar_mass_ratio* - The molar mass ratio of water to organic
    matter.
- `-` *organic_mole_fraction* - The fraction of organic matter.
- `-` *oxygen2carbon* - The oxygen to carbon ratio.
- `-` *density* - The density of the mixture.
- `-` *functional_group* - Optional functional group(s) of the organic
    compound, if applicable.

#### Returns

- gibbs_mix : Gibbs energy of mixing (including 1/RT)
- derivative_gibbs : derivative of Gibbs energy with respect to
- mole fraction of organics (includes 1/RT)

#### Signature

```python
def gibbs_mix_weight(
    molar_mass_ratio: ArrayLike,
    organic_mole_fraction: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
    functional_group: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## gibbs_of_mixing

[Show source in binary_activity.py:98](https://github.com/uncscode/particula/blob/main/particula/activity/binary_activity.py#L98)

Calculate the Gibbs free energy of mixing for a binary mixture.

#### Arguments

- `-` *molar_mass_ratio* - The molar mass ratio of water to organic
    matter.
- `-` *organic_mole_fraction* - The fraction of organic matter.
- `-` *oxygen2carbon* - The oxygen to carbon ratio.
- `-` *density* - The density of the mixture.
- `-` *fit_dict* - A dictionary of fit values for the low oxygen2carbon region

#### Returns

- `Tuple[np.ndarray,` *np.ndarray]* - A tuple containing the Gibbs free
energy of mixing and its derivative.

#### Signature

```python
def gibbs_of_mixing(
    molar_mass_ratio: ArrayLike,
    organic_mole_fraction: ArrayLike,
    oxygen2carbon: ArrayLike,
    density: ArrayLike,
    fit_dict: dict,
) -> Tuple[np.ndarray, np.ndarray]: ...
```
