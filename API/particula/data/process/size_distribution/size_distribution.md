# Size Distribution

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Size Distribution

> Auto-generated documentation for [particula.data.process.size_distribution](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py) module.

## iterate_merge_distributions

[Show source in size_distribution.py:341](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py#L341)

Merge two sets of particle size distributions using linear weighting.

#### Arguments

- `concentration_lower` - The concentration of particles in the lower
    distribution.
- `diameters_lower` - The diameters corresponding to the lower distribution.
- `concentration_upper` - The concentration of particles in the upper
    distribution.
- `diameters_upper` - The diameters corresponding to the upper distribution.

#### Returns

Tuple:
- The merged diameter distribution.
- The merged concentration distribution.

#### Signature

```python
def iterate_merge_distributions(
    concentration_lower: np.ndarray,
    diameters_lower: np.ndarray,
    concentration_upper: np.ndarray,
    diameters_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
```



## mean_properties

[Show source in size_distribution.py:20](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py#L20)

Calculate the mean properties of the size distribution.

#### Arguments

- `sizer_dndlogdp` - Array of particle concentrations in each bin.
- `sizer_diameter` - Array of bin center diameters.
- `total_concentration` - Optional; the total concentration of particles
    in the distribution. If not provided, it will be calculated.
- `sizer_limits` - Optional; the lower and upper limits of the size
    range of interest. If not provided, the full range will be used.

#### Returns

Tuple:
- Total concentration of particles in the distribution.
- Total mass of particles in the distribution.
- Mean diameter of the distribution by number.
- Mean diameter of the distribution by volume.
- Geometric mean diameter of the distribution.
- Mode diameter of the distribution by number.
- Mode diameter of the distribution by volume.

#### Signature

```python
def mean_properties(
    sizer_dndlogdp: np.ndarray,
    sizer_diameter: np.ndarray,
    total_concentration: Optional[float] = None,
    sizer_limits: Optional[list] = None,
) -> Tuple[float, float, float, float, float, float, float]: ...
```



## merge_distributions

[Show source in size_distribution.py:261](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py#L261)

Merge two particle size distributions using linear weighting,
accounting for mobility versus aerodynamic diameters.

#### Arguments

- `concentration_lower` - The concentration of particles in the lower
    distribution.
- `diameters_lower` - The diameters corresponding to the lower distribution.
- `concentration_upper` - The concentration of particles in the upper
    distribution.
- `diameters_upper` - The diameters corresponding to the upper distribution.

#### Returns

Tuple:
- `-` *new_2d* - The merged concentration distribution.
- `-` *new_diameter* - The merged diameter distribution.

#### Notes

Add process the moblity vs aerodynamic diameters

#### Signature

```python
def merge_distributions(
    concentration_lower: np.ndarray,
    diameters_lower: np.ndarray,
    concentration_upper: np.ndarray,
    diameters_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
```



## merge_size_distribution

[Show source in size_distribution.py:387](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py#L387)

Merge two particle size distributions using linear weighting.
The concentrations should be in dN/dlogDp.

#### Arguments

- `stream_lower` - The stream with the lower size range, e.g., from an SMPS.
- `stream_upper` - The stream with the upper size range, e.g., from an
    OPS or APS.
- `lower_units` - The units of the lower distribution. Default is 'nm'.
- `upper_units` - The units of the upper distribution. Default is 'um'.

#### Returns

- `Stream` - A stream object containing the merged size distribution.

#### Signature

```python
def merge_size_distribution(
    stream_lower: Stream,
    stream_upper: Stream,
    lower_units: str = "nm",
    upper_units: str = "um",
) -> object: ...
```

#### See also

- [Stream](../stream.md#stream)



## resample_distribution

[Show source in size_distribution.py:428](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py#L428)

Resample a particle size distribution to a new set of diameters using
numpy interpolation. Extrapolated values will be set to NaN.

#### Arguments

- `stream` - The stream object containing the size distribution to resample.
- `new_diameters` - The new diameters to which the distribution will be
    resampled.
- `concentration_scale` - The concentration scale of the distribution.
    Options are 'dn/dlogdp', 'dn', 'pms'
    (which is equivalent to 'dn'), or 'pdf'. Default is 'dn/dlogdp'.
- `clone` - Whether to clone the stream before resampling. Default is False.

#### Returns

- `Stream` - The resampled stream object.

#### Signature

```python
def resample_distribution(
    stream: Stream,
    new_diameters: np.ndarray,
    concentration_scale: str = "dn/dlogdp",
    clone: bool = False,
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)



## sizer_mean_properties

[Show source in size_distribution.py:112](https://github.com/uncscode/particula/blob/main/particula/data/process/size_distribution.py#L112)

Calculate the mean properties of the size distribution and return the
updated stream.

#### Arguments

- `stream` - The stream containing the size distribution data to process.
- `sizer_limits` - A list specifying the lower and upper limits of the
    size range of interest, in the units specified by `diameter_units`.
    Default is None, which means the full range is used.
- `density` - The density of the particles in g/cm³. Default is 1.5 g/cm³.
- `diameter_units` - The units of the diameter. Default is 'nm'. The
    specified units will be converted to nanometers.

#### Returns

- `Stream` - The updated stream with the mean properties added.

#### Signature

```python
def sizer_mean_properties(
    stream: Stream,
    sizer_limits: Optional[List[float]] = None,
    density: float = 1.5,
    diameter_units: str = "nm",
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)
