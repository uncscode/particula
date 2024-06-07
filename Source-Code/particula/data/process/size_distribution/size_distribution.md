# Size Distribution

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Size Distribution

> Auto-generated documentation for [particula.data.process.size_distribution](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py) module.

## iterate_merge_distributions

[Show source in size_distribution.py:351](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py#L351)

Merge two sets of particle size distributions using linear weighting.

#### Arguments

- `concentration_lower` - The concentration of particles in the
    lower distribution.
- `diameters_lower` - The diameters corresponding to the
    lower distribution.
- `concentration_upper` - The concentration of particles in the
    upper distribution.
- `diameters_upper` - The diameters corresponding to the upper distribution.

#### Returns

A tuple containing the merged diameter distribution and the merged
    concentration distribution.

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

[Show source in size_distribution.py:20](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py#L20)

Calculates the mean properties of the size distribution.

Args
----------
sizer_dndlogdp : List[float]
    Concentration of particles in each bin.
sizer_diameter : List[float]
    Bin centers
total_concentration : Optional[float], default=None
    Total concentration of particles in the distribution.
sizer_limits : Optional[Tuple[float, float]], default=None
    The lower and upper limits of the size of interest.

Returns
-------
Tuple[float, float, float, float, float, float, float]
    Total concentration of particles in the distribution.
    Total mass of particles in the distribution.
    Mean diameter of the distribution by number.
    Mean diameter of the distribution by volume.
    Geometric mean diameter of the distribution.
    Mode diameter of the distribution by number.
    Mode diameter of the distribution by volume.

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

[Show source in size_distribution.py:271](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py#L271)

Merge two particle size distributions using linear weighting.

#### Arguments

concentration_lower:
    The concentration of particles in the lower
    distribution.
diameters_lower:
    The diameters corresponding to the lower distribution.
    - `concentration_upper` - The concentration of particles in the upper
    distribution.
diameters_upper:
    The diameters corresponding to the upper distribution.

#### Returns

- `new_2d` - The merged concentration distribution.
- `new_diameter` - The merged diameter distribution.

add in an acount for the moblity vs aerodynamic diameters

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

[Show source in size_distribution.py:397](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py#L397)

Merge two sets of particle size distributions using linear weighting.
The concentration should be in dN/dlogDp.

#### Arguments

- `stream_smaller` - The stream with lower sizes, e.g. SMPS.
- `stream_larger` - The stream with larger sizes, e.g. OPS. or APS
- `lower_units` - The units of the lower distribution. The default is 'nm'.
- `upper_units` - The units of the upper distribution. The default is 'um'.

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

[Show source in size_distribution.py:437](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py#L437)

Resample a particle size distribution to a new set of diameters.
Using np interpolation, and extrapolation is nan.

#### Arguments

- `stream` - (Stream)
    The stream to resample.
- `new_diameters` - (np.ndarry)
    The new diameters to resample to.
- `concentration_scale` - (str)
    The concentration scale of the distribution. Either, 'dn/dlogdp',
    'dn', 'pms' (which is dn), or 'pdf'.

#### Returns

- `Stream` - The resampled stream.

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

[Show source in size_distribution.py:116](https://github.com/Gorkowski/particula/blob/main/particula/data/process/size_distribution.py#L116)

Calculates the mean properties of the size distribution, and returns a
stream.

Args
----------
stream : Stream
    The stream to process.
sizer_limits : list, optional [in diameter_units]
    The lower and upper limits of the size of interest. The default is None
density : float, optional
    The density of the particles. The default is 1.5 g/cm3.
diameter_units : str
    The units of the diameter. The default is 'nm'. This will be converted
    to nm.

Returns
-------
stream : Stream
    The stream with the mean properties added.

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
