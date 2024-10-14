# Convert

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Convert

> Auto-generated documentation for [particula.util.convert](https://github.com/uncscode/particula/blob/main/particula/util/convert.py) module.

## coerce_type

[Show source in convert.py:9](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L9)

Coerces data to dtype if it is not already of that type.

#### Examples

```python
>>> coerce_type(1, float)
1.0
>>> coerce_type([1, 2, 3], np.ndarray)
array([1, 2, 3])
```

#### Arguments

- `data` - The data to be coerced.
- `dtype` - The desired data type.

#### Returns

The coerced data.

#### Raises

- `ValueError` - If the data cannot be coerced to the desired type.

#### Signature

```python
def coerce_type(data, dtype): ...
```



## convert_sizer_dn

[Show source in convert.py:440](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L440)

Converts the sizer data from dn/dlogdp to d_num.

The bin width is defined as the  difference between the upper and lower
diameter limits of each bin. This function calculates the bin widths
based on the input diameter array. Assumes a log10 scale for dp edges.

#### Arguments

- `diameter` *np.ndarray* - Array of particle diameters.
- `dn_dlogdp` *np.ndarray* - Array of number concentration of particles per
unit logarithmic diameter.
- `inverse` *bool* - If True, converts from d_num to dn/dlogdp.

#### Returns

- `np.ndarray` - Array of number concentration of particles
per unit diameter.

#### References

- `Eq` - dN/dlogD_p = dN/( log(D_{p-upper}) - log(D_{p-lower}) )
https://tsi.com/getmedia/1621329b-f410-4dce-992b-e21e1584481a/
PR-001-RevA_Aerosol-Statistics-AppNote?ext=.pdf

#### Signature

```python
def convert_sizer_dn(
    diameter: np.ndarray, dn_dlogdp: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## data_shape_check

[Show source in convert.py:543](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L543)

Check the shape of the input data and header list, and reshape the data if
necessary. The data array can be 1D or 2D. If the data array is 2D, the
time array must match the last dimensions of the data array. If the data
array is 1D, the header list must be a single entry.

#### Arguments

- `time` *np.ndarray* - 1D array of time values.
- `data` *np.ndarray* - 1D or 2D array of data values.
- `header` *list* - List of header values.

#### Returns

Reshaped data array.

#### Raises

- `ValueError` - If the length of the header list does not match the first
dimension of the data array.

#### Signature

```python
def data_shape_check(time: np.ndarray, data: np.ndarray, header: list) -> np.ndarray: ...
```



## distribution_convert_pdf_pms

[Show source in convert.py:603](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L603)

Convert between a probability density function (PDF) and a probability
mass spectrum (PMS) based on the specified direction.

#### Arguments

x_array : An array of radii corresponding to the bins of the
    distribution, shape (m).
distribution : The concentration values of the distribution
    (either PDF or PMS) at the given radii. Supports broadcasting
    across x_array (n,m).
to_PDF : Direction of conversion. If True, converts PMS to PDF.
    If False, converts PDF to PMS.

#### Returns

converted_distribution : The converted distribution array
    (either PDF or PMS).

#### Signature

```python
def distribution_convert_pdf_pms(
    x_array: np.ndarray, distribution: np.ndarray, to_pdf: bool = True
) -> np.ndarray: ...
```



## effective_refractive_index

[Show source in convert.py:402](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L402)

Calculate the effective refractive index of a mixture of two solutes, given
the refractive index of each solute and the volume of each solute. The
mixing is based on volume-weighted molar refraction.

#### Arguments

- `m_zero` - The refractive index of solute 0.
- `m_one` - The refractive index of solute 1.
- `volume_zero` - The volume of solute 0.
- `volume_one` - The volume of solute 1.

#### Returns

The effective refractive index of the mixture.

#### References

Liu, Y., &#38; Daum, P. H. (2008). Relationship of refractive
index to mass density and self-consistency mixing rules for
multicomponent mixtures like ambient aerosols.
Journal of Aerosol Science, 39(11), 974-986.
https://doi.org/10.1016/j.jaerosci.2008.06.006

#### Signature

```python
def effective_refractive_index(
    m_zero: Union[float, complex],
    m_one: Union[float, complex],
    volume_zero: float,
    volume_one: float,
) -> Union[float, complex]: ...
```



## get_values_in_dict

[Show source in convert.py:509](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L509)

Returns a list of values for keys in a dictionary.

#### Examples

```python
>>> my_dict = {'a': 1, 'b': 2, 'c': 3}
>>> get_values_in_dict(['a', 'c'], my_dict)
[1, 3]
```

#### Arguments

- `key_list` - List of keys to check in the dictionary.
- `dict_to_check` - The dictionary to check for the given keys.

#### Returns

- `List` - A list of values for keys in the dictionary.

#### Raises

- `KeyError` - If any of the keys in the `key_list` are not present in
the dictionary.

#### Signature

```python
def get_values_in_dict(
    key_list: List[str], dict_to_check: Dict[str, Any]
) -> List[Any]: ...
```



## kappa_from_volume

[Show source in convert.py:206](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L206)

Calculate the kappa parameter from the volume of solute and water,
given the water activity.

#### Arguments

- `volume_solute` - The volume of solute.
- `volume_water` - The volume of water.
- `water_activity` - The water activity.

#### Returns

--------
    The kappa parameter as a float.

#### Signature

```python
def kappa_from_volume(
    volume_solute: Union[float, np.ndarray],
    volume_water: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]: ...
```



## kappa_volume_solute

[Show source in convert.py:148](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L148)

Calculate the volume of solute in a volume of total solution,
given the kappa parameter and water activity.

#### Arguments

- `volume_total` - The volume of the total solution.
- `kappa` - The kappa parameter.
- `water_activity` - The water activity.

#### Returns

--------
    The volume of solute as a numpy array.

#### Signature

```python
def kappa_volume_solute(
    volume_total: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]: ...
```



## kappa_volume_water

[Show source in convert.py:178](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L178)

Calculate the volume of water given volume of solute, kappa parameter,
and water activity.

#### Arguments

- `volume_solute` - The volume of solute.
- `kappa` - The kappa parameter.
- `water_activity` - The water activity.

#### Returns

--------
    The volume of water as a float.

#### Signature

```python
def kappa_volume_water(
    volume_solute: Union[float, NDArray[np.float64]],
    kappa: Union[float, NDArray[np.float64]],
    water_activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## length_to_volume

[Show source in convert.py:125](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L125)

Convert radius or diameter to volume.

#### Arguments

- `length` - The length to be converted.
- `length_type` - The type of length ('radius' or 'diameter').
    Default is 'radius'.

#### Returns

--------
    The volume.

#### Signature

```python
def length_to_volume(
    length: Union[float, np.ndarray], length_type: str = "radius"
) -> Union[float, np.ndarray]: ...
```



## list_to_dict

[Show source in convert.py:486](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L486)

Converts a list of strings to a dictionary. The keys are the strings
and the values are the index of the string in the list.

#### Arguments

- `list_of_str` *list* - A non-empty list of strings.

#### Returns

- `dict` - A dictionary where the keys are the strings and the values are
    the index of the string in the list.

#### Signature

```python
def list_to_dict(list_of_str: list) -> dict: ...
```



## mass_concentration_to_mole_fraction

[Show source in convert.py:284](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L284)

Convert mass concentrations to mole fractions for N components.

#### Arguments

- `mass_concentrations` - A list or ndarray of mass concentrations
(e.g., kg/m^3).
- `molar_masses` - A list or ndarray of molecular weights (e.g., g/mol).

#### Returns

An ndarray of mole fractions.

#### Notes

The mole fraction of a component is given by the ratio of its molar
concentration to the total molar concentration of all components.

#### Signature

```python
def mass_concentration_to_mole_fraction(
    mass_concentrations: NDArray[np.float64], molar_masses: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## mass_concentration_to_volume_fraction

[Show source in convert.py:311](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L311)

Convert mass concentrations to volume fractions for N components.

#### Arguments

- `mass_concentrations` - A list or ndarray of mass concentrations
(e.g., kg/m^3).
- `densities` - A list or ndarray of densities of each component
(e.g., kg/m^3).

#### Returns

An ndarray of volume fractions.

#### Notes

The volume fraction of a component is calculated by dividing the volume
of that component (derived from mass concentration and density) by the
total volume of all components.

#### Signature

```python
def mass_concentration_to_volume_fraction(
    mass_concentrations: NDArray[np.float64], densities: NDArray[np.float64]
) -> NDArray[np.float64]: ...
```



## mass_fraction_to_volume_fraction

[Show source in convert.py:342](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L342)

Converts the mass fraction of a solute to the volume fraction in a
binary mixture.

#### Arguments

- `mass_fraction` *float* - The mass fraction of the solute in the mixture.
- `density_solute` *float* - The density of the solute.
- `density_solvent` *float* - The density of the solvent.

#### Returns

- `Tuple[float,` *float]* - A tuple containing the volume fraction of the
    solute and solvent in the mixture.

#### Examples

If `mass_fraction` is 0.5, `density_solute` is 1.5 g/cm^3, and
`density_solvent` is 2 g/cm^3, this function returns (0.5714, 0.4285),
indicating that the solute and solvent occupy 57% and 42% of the
mixture's volume, respectively.

#### Signature

```python
def mass_fraction_to_volume_fraction(
    mass_fraction: float, density_solute: float, density_solvent: float
) -> Tuple[float, float]: ...
```



## mole_fraction_to_mass_fraction

[Show source in convert.py:233](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L233)

Convert mole fraction to mass fraction.

#### Arguments

- `mole_fraction0` - The mole fraction of the first component.
- `molecular_weight0` - The molecular weight of the first component.
- `molecular_weight1` - The molecular weight of the second component.

#### Returns

A tuple containing the mass fractions of the two components as floats.

#### Signature

```python
def mole_fraction_to_mass_fraction(
    mole_fraction0: float, molecular_weight0: float, molecular_weight1: float
) -> Tuple[float, float]: ...
```



## mole_fraction_to_mass_fraction_multi

[Show source in convert.py:259](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L259)

Convert mole fractions to mass fractions for N components.
Assumes that sum(mole_fractions) == 1.

#### Arguments

- `mole_fractions` - A list of mole fractions.
- `molecular_weights` - A list of molecular weights.

#### Returns

A list of mass fractions.

#### Signature

```python
def mole_fraction_to_mass_fraction_multi(
    mole_fractions: list[float], molecular_weights: list[float]
) -> list[float]: ...
```



## radius_diameter

[Show source in convert.py:87](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L87)

Convert a radius to a diameter, or vice versa.

#### Arguments

- `value` - The value to be converted.
- `to_diameter` - If True, convert from radius to diameter.
If False, convert from diameter to radius.

#### Returns

The converted value.

#### Signature

```python
def radius_diameter(value: float, to_diameter: bool = True) -> float: ...
```



## round_arbitrary

[Show source in convert.py:37](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L37)

Rounds the input values to the nearest multiple of the base.

For values exactly halfway between rounded decimal values, "Bankers
rounding applies" rounds to the nearest even value. Thus 1.5 and 2.5
round to 2.0, -0.5 and 0.5 round to 0.0, etc.

#### Arguments

- `values` - The values to be rounded.
- `base` - The base to which the values should be rounded.
- `mode` - The rounding mode: 'round', 'floor', 'ceil'
- `nonzero_edge` - If true the zero values are replaced
by the original values.

#### Returns

- `rounded` - The rounded values.

#### Signature

```python
def round_arbitrary(
    values: Union[float, list[float], np.ndarray],
    base: Union[float, np.float64] = 1.0,
    mode: str = "round",
    nonzero_edge: bool = False,
) -> Union[float, NDArray[np.float64]]: ...
```



## volume_to_length

[Show source in convert.py:102](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L102)

Convert a volume to a radius or diameter.

#### Arguments

- `volume` - The volume to be converted.
- `length_type` - The type of length to convert to ('radius' or 'diameter')
Default is 'radius'.

#### Returns

The converted length.

#### Signature

```python
def volume_to_length(
    volume: Union[float, NDArray[np.float64]], length_type: str = "radius"
) -> Union[float, NDArray[np.float64]]: ...
```



## volume_water_from_volume_fraction

[Show source in convert.py:376](https://github.com/uncscode/particula/blob/main/particula/util/convert.py#L376)

Calculates the volume of water in a volume of solute, given the volume
fraction of water in the mixture.

#### Arguments

- `volume_solute_dry` *float* - The volume of the solute, excluding water.
- `volume_fraction_water` *float* - The volume fraction of water in the
                    mixture, expressed as a decimal between 0 and 1.

#### Returns

- `float` - The volume of water in the mixture, in the same units as
    `volume_solute_dry`.

#### Examples

If `volume_solute_dry` is 100 mL and `volume_fraction_water` is 0.8,
this function returns 400 mL, indicating that there are 400 mL of water
in the total 100 mL + 400 mL mixture.

#### Signature

```python
def volume_water_from_volume_fraction(
    volume_solute_dry: float, volume_fraction_water: float
) -> float: ...
```
