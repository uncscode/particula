# Optical Instrument

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Optical Instrument

> Auto-generated documentation for [particula.data.process.optical_instrument](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py) module.

## CapsInstrumentKeywordBuilder

[Show source in optical_instrument.py:13](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L13)

Builder class for CAPS Instrument Keywords dictionary.

#### Signature

```python
class CapsInstrumentKeywordBuilder:
    def __init__(self): ...
```

### CapsInstrumentKeywordBuilder().build

[Show source in optical_instrument.py:92](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L92)

Validate and return the keywords dictionary.

#### Returns

- `dict` - The validated keywords dictionary.

#### Signature

```python
def build(self) -> dict[str, Union[str, float, int, bool]]: ...
```

### CapsInstrumentKeywordBuilder().pre_build_check

[Show source in optical_instrument.py:77](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L77)

Check that all required parameters have been set.

#### Raises

- `ValueError` - If any required keyword has not been set.

#### Signature

```python
def pre_build_check(self): ...
```

### CapsInstrumentKeywordBuilder().set_keyword

[Show source in optical_instrument.py:38](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L38)

Set the keyword parameter for the activity calculation.

#### Arguments

- `keyword` - The keyword to set.
- `value` - The value to set the keyword to.

#### Raises

- `ValueError` - If the keyword is not recognized or the value type
    is incorrect.

#### Signature

```python
def set_keyword(self, keyword: str, value: Optional[Union[str, float, int, bool]]): ...
```

### CapsInstrumentKeywordBuilder().set_keywords

[Show source in optical_instrument.py:67](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L67)

Set multiple keywords at once.

#### Arguments

- `kwargs` - The keywords and their values to set.

#### Signature

```python
def set_keywords(self, **kwargs: Union[str, float, int, bool]): ...
```



## albedo_from_ext_scat

[Show source in optical_instrument.py:247](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L247)

Calculate the albedo from the extinction and scattering data in the stream.

This function computes the absorption as the difference between extinction
and scattering, and the single-scattering albedo as the ratio of scattering
to extinction. If the extinction values are zero or negative, the albedo is
set to `np.nan`.

#### Arguments

- `stream` - The datastream containing CAPS data.
- `extinction_key` - The key for the extinction data in the stream.
- `scattering_key` - The key for the scattering data in the stream.
- `new_absorption_key` - The key where the calculated absorption will
    be stored.
- `new_albedo_key` - The key where the calculated albedo will
    be stored.

#### Returns

- `Stream` - The updated datastream with the new absorption and albedo
values.

#### Raises

- `KeyError` - If the provided extinction or scattering keys are not found
    in the stream.

#### Signature

```python
def albedo_from_ext_scat(
    stream: Stream,
    extinction_key: str,
    scattering_key: str,
    new_absorption_key: str,
    new_albedo_key: str,
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)



## caps_processing

[Show source in optical_instrument.py:102](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L102)

Process CAPS data and SMPS data for kappa fitting, apply truncation
corrections, and add the results to the caps stream.

#### Arguments

- `stream_size_distribution` - Stream containing size distribution data.
- `stream_sizer_properties` - Stream containing sizer properties data.
- `stream_caps` - Stream containing CAPS data.
- `keywords` - Dictionary containing configuration parameters.

#### Returns

Stream with processed CAPS data, including kappa fitting results
and truncation corrections.

#### Signature

```python
def caps_processing(
    stream_size_distribution: Stream,
    stream_sizer_properties: Stream,
    stream_caps: Stream,
    keywords: dict[str, Union[str, float, int, bool]],
): ...
```

#### See also

- [Stream](../stream.md#stream)



## enhancement_ratio

[Show source in optical_instrument.py:313](https://github.com/uncscode/particula/blob/main/particula/data/process/optical_instrument.py#L313)

Calculate the enhancement ratio from two data keys in the stream.

This is the ratio between the numerator and the denominator. If the
denominator is zero, then the ratio is set to `np.nan`. This function
is useful for f(RH) calculations.

#### Arguments

- `stream` - The datastream containing the data.
- `numerator_key` - The key for the numerator data in the stream.
- `denominator_key` - The key for the denominator data in the stream.
- `new_key` - The key where the calculated enhancement ratio will
    be stored.

#### Returns

- `Stream` - The updated datastream with the new enhancement ratio values.

#### Raises

- `KeyError` - If the provided numerator or denominator keys are not found
    in the stream.

#### Signature

```python
def enhancement_ratio(
    stream: Stream, numerator_key: str, denominator_key: str, new_key: str
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)
