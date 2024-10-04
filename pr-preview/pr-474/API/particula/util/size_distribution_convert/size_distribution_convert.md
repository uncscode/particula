# Size Distribution Convert

[Particula Index](../../README.md#particula-index) / [Particula](../index.md#particula) / [Util](./index.md#util) / Size Distribution Convert

> Auto-generated documentation for [particula.util.size_distribution_convert](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py) module.

## ConversionStrategy

[Show source in size_distribution_convert.py:23](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L23)

Defines an interface for conversion strategies between particle size
distribution formats.

Subclasses must implement the convert method to perform specific
conversion logic.

#### Signature

```python
class ConversionStrategy: ...
```

### ConversionStrategy().convert

[Show source in size_distribution_convert.py:31](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L31)

Converter method common interface, for subclasses.

#### Arguments

- `diameters` *np.ndarray* - The particle diameters.
- `concentration` *np.ndarray* - The concentration values.
- `inverse` *bool* - Flag to perform the inverse conversion.

#### Returns

- `np.ndarray` - The concentration values converted.

#### Raises

- `NotImplementedError` - If the subclass does not implement this.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## DNdlogDPtoPDFConversionStrategy

[Show source in size_distribution_convert.py:83](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L83)

Implements conversion between dn/dlogdp and PDF formats through an
intermediate PMS format.

#### Signature

```python
class DNdlogDPtoPDFConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### DNdlogDPtoPDFConversionStrategy().convert

[Show source in size_distribution_convert.py:87](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L87)

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## DNdlogDPtoPMSConversionStrategy

[Show source in size_distribution_convert.py:61](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L61)

Implements conversion between dn/dlogdp and PMS formats using the
convert_sizer_dn method.

#### Signature

```python
class DNdlogDPtoPMSConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### DNdlogDPtoPMSConversionStrategy().convert

[Show source in size_distribution_convert.py:65](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L65)

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## PMStoPDFConversionStrategy

[Show source in size_distribution_convert.py:73](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L73)

Implements conversion between PMS and PDF formats.

#### Signature

```python
class PMStoPDFConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### PMStoPDFConversionStrategy().convert

[Show source in size_distribution_convert.py:76](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L76)

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## SameScaleConversionStrategy

[Show source in size_distribution_convert.py:52](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L52)

Implements conversion between the same scales, which is a no-op.

#### Signature

```python
class SameScaleConversionStrategy(ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### SameScaleConversionStrategy().convert

[Show source in size_distribution_convert.py:55](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L55)

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## SizerConverter

[Show source in size_distribution_convert.py:102](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L102)

A converter that uses a specified ConversionStrategy to convert
particle size distribution data between different formats.

#### Signature

```python
class SizerConverter:
    def __init__(self, strategy: ConversionStrategy): ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)

### SizerConverter().convert

[Show source in size_distribution_convert.py:114](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L114)

Converts particle size distribution data using the specified
strategy.

#### Arguments

- `diameters` *np.ndarray* - The particle diameters.
- `concentration` *np.ndarray* - The concentration values.
- `inverse` *bool* - Flag to perform the inverse conversion.

#### Returns

- `np.ndarray` - The converted concentration values.

#### Signature

```python
def convert(
    self, diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False
) -> np.ndarray: ...
```



## get_conversion_strategy

[Show source in size_distribution_convert.py:130](https://github.com/uncscode/particula/blob/main/particula/util/size_distribution_convert.py#L130)

Factory function to create and return an appropriate conversion
strategy based on input and output scales. Use the inverse flag in the
converter to invert the directions of the input and output scales.

#### Arguments

- `input_scale` - The scale of the input concentration values.
    Either 'dn/dlogdp' or 'pms'.
- `output_scale` - The desired scale of the output concentration
    values. Either 'pms' or 'pdf'. Use inverse flag to invert the input
    and output scales.

#### Returns

- [ConversionStrategy](#conversionstrategy) - A strategy object capable of converting between
    the specified scales.

#### Raises

- `ValueError` - If the input_scale or output_scale is not supported, or
    if the specified conversion is unsupported.

#### Examples

``` py title="Convert dn/dlogdp to PMS"
strategy = get_conversion_strategy('dn/dlogdp', 'pms')
converter = Converter(strategy)
converted_concentration = converter.convert(
    diameters, concentration, inverse=False)
```

#### Signature

```python
def get_conversion_strategy(
    input_scale: str, output_scale: str
) -> ConversionStrategy: ...
```

#### See also

- [ConversionStrategy](#conversionstrategy)
