# particula.particles.properties.convert_size_distribution

*module*

Convert between different size distribution formats.

This module defines conversion strategies as classes that implement
the ConversionStrategy interface. The SizerConverter composes a
ConversionStrategy instance to flexibly convert distribution data among
scales.

**Examples**

``` py title="Example usage"
import numpy as np
from particula.util.size_distribution_convert import (
    get_conversion_strategy, SizerConverter
)

diameters = np.array([1e-7, 1e-6, 1e-5])
concentration = np.array([1e6, 1e5, 1e4])

strategy = get_conversion_strategy("dn/dlogdp", "PMF")
converter = SizerConverter(strategy)
converted_conc = converter.convert(diameters, concentration)
print(converted_conc)
# Output: [...]
```

> **References**
>
> - W. C. Hinds, "Aerosol Technology," 2nd ed., Wiley-Interscience, 1999.

To be moved to particle.properties. -kyle

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L1-L1)

## particula.particles.properties.convert_size_distribution.ConversionStrategy

*class*

```python
class ConversionStrategy
```

Conversion strategy interface for particle size distribution data.

**Functions**

- `- convert` — Convert distribution data between input and output scales.

Defines an interface for conversion strategies between particle size
distribution formats.

All subclasses must implement the convert method to perform the actual
conversion logic.

**Examples**

``` py title="Subclass Example"
class CustomStrategy(ConversionStrategy):
    def convert(self, diameters, concentration, inverse=False):
        # Custom conversion logic here
        return concentration
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L36-L78)

### particula.particles.properties.convert_size_distribution.ConversionStrategy.convert

*method*

```python
def convert(diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False) -> np.ndarray
```

Convert distribution data from one scale to another.

**Parameters**

- `diameters` (`np.ndarray`) — Array of particle diameters.
- `concentration` (`np.ndarray`) — The distribution data corresponding to these diameters.
- `inverse` (`bool`) (default: `False`) — If True, reverse the direction of the conversion.

**Returns**

- (`np.ndarray`) — np.ndarray of converted distribution data.

**Raises**

- `NotImplementedError` — If not overridden by a subclass.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L56-L78)

## particula.particles.properties.convert_size_distribution.DNdlogDPtoPDFConversionStrategy

*class*

```python
class DNdlogDPtoPDFConversionStrategy(ConversionStrategy)
```

Bases: `particula.particles.properties.convert_size_distribution.ConversionStrategy`

Conversion strategy for converting between dn/dlogdp and PDF formats.

This strategy first converts dn/dlogdp to PMF, then to PDF, or the
reverse if inverse is True.

**Examples**

``` py title="Example Usage"
strategy = DNdlogDPtoPDFConversionStrategy()
result_pdf = strategy.convert(diameters, dn_dlogdp_data)
# result_pdf is now in PDF format
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L178-L225)

### particula.particles.properties.convert_size_distribution.DNdlogDPtoPDFConversionStrategy.convert

*method*

```python
def convert(diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False) -> np.ndarray
```

Convert between dn/dlogdp and PDF formats.

This method first converts dn/dlogdp to PMF, then to PDF, or the
reverse if inverse is True.

**Parameters**

- `diameters` (`np.ndarray`) — Array of particle diameters.
- `concentration` (`np.ndarray`) — Distribution data in dn/dlogdp or PDF format.
- `inverse` (`bool`) (default: `False`) — If True, convert from PDF to dn/dlogdp; otherwise the opposite.

**Returns**

- (`np.ndarray`) — np.ndarray of the distribution in the target format.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L192-L225)

## particula.particles.properties.convert_size_distribution.DNdlogDPtoPMFConversionStrategy

*class*

```python
class DNdlogDPtoPMFConversionStrategy(ConversionStrategy)
```

Bases: `particula.particles.properties.convert_size_distribution.ConversionStrategy`

Conversion strategy for converting between dn/dlogdp and PMF formats.

**Examples**

``` py title="Example Usage"
strategy = DNdlogDPtoPMFConversionStrategy()
result = strategy.convert(diameters, dn_dlogdp_conc)
# result is now in PMF format
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L114-L142)

### particula.particles.properties.convert_size_distribution.DNdlogDPtoPMFConversionStrategy.convert

*method*

```python
def convert(diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False) -> np.ndarray
```

Perform the conversion between dn/dlogdp and PMF formats.

**Parameters**

- `diameters` (`np.ndarray`) — Array of particle diameters.
- `concentration` (`np.ndarray`) — Distribution data in dn/dlogdp or PMF format.
- `inverse` (`bool`) (default: `False`) — If True, convert from PMF to dn/dlogdp; otherwise the opposite.

**Returns**

- (`np.ndarray`) — np.ndarray of the distribution in the target format.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L125-L142)

## particula.particles.properties.convert_size_distribution.PMFtoPDFConversionStrategy

*class*

```python
class PMFtoPDFConversionStrategy(ConversionStrategy)
```

Bases: `particula.particles.properties.convert_size_distribution.ConversionStrategy`

Conversion strategy for converting between PMF and PDF formats.

**Examples**

``` py title="Example Usage"
strategy = PMFtoPDFConversionStrategy()
result_pdf = strategy.convert(diameters, PMF_data, inverse=False)
# result_pdf is now in PDF format
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L145-L175)

### particula.particles.properties.convert_size_distribution.PMFtoPDFConversionStrategy.convert

*method*

```python
def convert(diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False) -> np.ndarray
```

Perform the conversion between PMF and PDF formats.

**Parameters**

- `diameters` (`np.ndarray`) — Array of particle diameters.
- `concentration` (`np.ndarray`) — Distribution data in PMF or PDF format.
- `inverse` (`bool`) (default: `False`) — If True, convert from PDF to PMF; otherwise from PMF to PDF.

**Returns**

- (`np.ndarray`) — np.ndarray of the distribution in the target format.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L156-L175)

## particula.particles.properties.convert_size_distribution.SameScaleConversionStrategy

*class*

```python
class SameScaleConversionStrategy(ConversionStrategy)
```

Bases: `particula.particles.properties.convert_size_distribution.ConversionStrategy`

Conversion strategy that returns the input concentration unchanged.

No conversion is performed because the input and output scales are the
same.

**Examples**

```py title="Example Usage"
strategy = SameScaleConversionStrategy()
result = strategy.convert(diameters, concentration)
# result is identical to concentration
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L81-L111)

### particula.particles.properties.convert_size_distribution.SameScaleConversionStrategy.convert

*method*

```python
def convert(diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False) -> np.ndarray
```

Return the concentration unchanged, since no conversion is needed.

**Parameters**

- `diameters` (`np.ndarray`) — Array of particle diameters (unused).
- `concentration` (`np.ndarray`) — The original distribution data.
- `inverse` (`bool`) (default: `False`) — Flag indicating direction (unused).

**Returns**

- (`np.ndarray`) — np.ndarray identical to the input concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L95-L111)

## particula.particles.properties.convert_size_distribution.SizerConverter

*class*

```python
class SizerConverter
```

A converter that composes a ConversionStrategy.

This class allows converting particle size distribution data between
different formats using a specified conversion strategy.
It provides a flexible interface to convert data without needing to
know the details of the conversion logic.

**Examples**

``` py title="Example Usage"
diameters = [1e-7, 1e-6, 1e-5]
concentration = [1e6, 1e5, 1e4]

strategy = DNdlogDPtoPMFConversionStrategy()
converter = SizerConverter(strategy)
new_conc = converter.convert(diameters, concentration)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L228-L272)

### particula.particles.properties.convert_size_distribution.SizerConverter.strategy

*attribute* · *instance attribute*

```python
strategy = strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L253-L253)

### particula.particles.properties.convert_size_distribution.SizerConverter.__init__

*method*

```python
def __init__(strategy: ConversionStrategy)
```

Initializes the converter with a conversion strategy.

**Parameters**

- `strategy` (`ConversionStrategy`) — The strategy to use for conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L247-L253)

### particula.particles.properties.convert_size_distribution.SizerConverter.convert

*method*

```python
def convert(diameters: np.ndarray, concentration: np.ndarray, inverse: bool = False) -> np.ndarray
```

Convert the particle size distribution data.

**Parameters**

- `diameters` (`np.ndarray`) — Array of particle diameters.
- `concentration` (`np.ndarray`) — Distribution data.
- `inverse` (`bool`) (default: `False`) — If True, reverse the conversion direction (if supported).

**Returns**

- (`np.ndarray`) — np.ndarray of the converted distribution.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L255-L272)

## particula.particles.properties.convert_size_distribution.get_distribution_conversion_strategy

*function*

```python
def get_distribution_conversion_strategy(input_scale: str, output_scale: str) -> ConversionStrategy
```

Factory function to obtain a conversion strategy.

**Parameters**

- `input_scale` (`str`) — Scale of the input distribution, e.g. 'dn/dlogdp' or 'pmf'.
- `output_scale` (`str`) — Desired scale of the output distribution, e.g. 'pmf' or 'pdf'.

**Returns**

- (`ConversionStrategy`) — A ConversionStrategy object supporting the requested conversion.

**Raises**

- `ValueError` — If scales are invalid or unsupported.

**Examples**

``` py title="Example Usage"
strategy = get_distribution_conversion_strategy('dn/dlogdp', 'pdf')
converter = SizerConverter(strategy)
converted_data = converter.convert(diameters, concentration)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L275-L332)

## particula.particles.properties.convert_size_distribution.get_distribution_in_dn

*function*

```python
def get_distribution_in_dn(diameter: np.ndarray, dn_dlogdp: np.ndarray, inverse: bool = False) -> np.ndarray
```

Convert the sizer data between dn/dlogdp and d_num formats.

If inverse=False, this function applies:
- d_num = dn_dlogdp × (log10(upper / lower))
    - The bin width is determined by upper and lower diameter limits,
      with log10 scaling.

If inverse=True, it reverts:
- dn/dlogdp = d_num / (log10(upper / lower))

**Parameters**

- `diameter` (`np.ndarray`) — Array of particle diameters.
- `dn_dlogdp` (`np.ndarray`) — Array representing either dn/dlogdp or d_num.
- `inverse` (`bool`) (default: `False`) — If True, converts from d_num to dn/dlogdp; otherwise the opposite.

**Returns**

- (`np.ndarray`) — A np.ndarray of the converted distribution.

**Examples**

```py
import numpy as np
from particula.util.size_distribution_convert import convert_sizer_dn

diam = np.array([1e-7, 2e-7, 4e-7])
dn_logdp = np.array([1e6, 1e5, 1e4])
result = convert_sizer_dn(diam, dn_logdp, inverse=False)
print(result)
# Output: d_num format for each diameter bin
```

> **References**
>
> - "dN/dlogD_p and dN/dD_p," TSI Application Note PR-001, 2010.
>     [link](
>     https://tsi.com/getmedia/1621329b-f410-4dce-992b-e21e1584481a/
>     PR-001-RevA_Aerosol-Statistics-AppNote?ext=.pd)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L335-L392)

## particula.particles.properties.convert_size_distribution.get_pdf_distribution_in_pmf

*function*

```python
def get_pdf_distribution_in_pmf(x_array: np.ndarray, distribution: np.ndarray, to_pdf: bool = True) -> np.ndarray
```

Convert from PMF to PDF or vice versa.

The conversion uses:
- y_pdf = y_PMF / Δx
- y_PMF = y_pdf * Δx
  - Δx is the bin width, determined by consecutive differences in x_array.

**Parameters**

- `x_array` (`np.ndarray`) — An array of diameters/radii for the distribution bins.
- `distribution` (`np.ndarray`) — The original distribution data (PMF or PDF).
- `to_pdf` (`bool`) (default: `True`) — If True, convert from PMF to PDF; if False, from PDF to PMF.

**Returns**

- (`np.ndarray`) — A np.ndarray of the converted distribution data.

**Examples**

```py
import numpy as np
import particula as par
x_vals = np.array([1.0, 2.0, 3.0])
PMF = np.array([10.0, 5.0, 2.5])
pdf = par.get_pdf_distribution_in_pmf(x_vals, PMF, to_pdf=True)
print(pdf)
# Output: [10.  5.  2.5] / [1.0, 1.0, ...] = ...
```

> **References**
>
> - Detailed bin width discussion in: TSI Application Note
>   "Aerosol Statistics and Densities."

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/convert_size_distribution.py#L395-L440)
