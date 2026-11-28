# particula.data.measurement

*module*

Measurement products with unit definitions, grids and source-scoped IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L1-L1)

## particula.data.measurement.Measurement

*class*

```python
class Measurement
```

Measurements from one source, with a row per accepted scan.

**Attributes**

- `lazy_frame` — Scan columns in profile output units, with UTC timestamps. The source digest and first physical line identify each scan.
- `lazy_issues` — Parse diagnostics with inclusive physical line spans.
- `manifest` — Import provenance, including the profile and source layouts.
- `grids` — Coordinate definitions keyed by the grid ID stored on each scan that contains spectral or array data.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L12-L184)

### particula.data.measurement.Measurement.grids

*attribute* · *instance attribute*

```python
grids = manifest['grids']
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L43-L43)

### particula.data.measurement.Measurement.lazy_frame

*attribute* · *instance attribute*

```python
lazy_frame = scans.with_columns(pl.lit(manifest['source_sha256']).alias('source_sha256'))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L38-L40)

### particula.data.measurement.Measurement.lazy_issues

*attribute* · *instance attribute*

```python
lazy_issues = issues
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L41-L41)

### particula.data.measurement.Measurement.manifest

*attribute* · *instance attribute*

```python
manifest = manifest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L42-L42)

### particula.data.measurement.Measurement.frame

*property*

```python
frame: pl.DataFrame
```

Collect the selected scans in memory, including their source IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L62-L64)

### particula.data.measurement.Measurement.issues

*property*

```python
issues: pl.DataFrame
```

Collect source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L67-L69)

### particula.data.measurement.Measurement.__init__

*method*

```python
def __init__(scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict) -> None
```

Validate scan columns and coordinates against the source layouts.

**Parameters**

- `scans` (`pl.LazyFrame`) — Lazy scan table with columns defined by the profile.
- `issues` (`pl.LazyFrame`) — Lazy diagnostic table for the original source.
- `manifest` (`dict`) — Import manifest containing the profile, segment headers, grids and source digest.

**Raises**

- `ValueError` — Scan column types or grids disagree with the layouts.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L24-L59)

### particula.data.measurement.Measurement.available_operations

*method*

```python
def available_operations() -> dict[str, tuple[str, ...]]
```

Map compatible operation names to their required input names.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L176-L184)

### particula.data.measurement.Measurement.filter

*method*

```python
def filter(predicate: pl.Expr | pl.Series | str | Iterable[pl.Expr | pl.Series | str] | bool | list[bool] | np.ndarray) -> Self
```

Select scans while retaining the original source diagnostics.

**Parameters**

- `predicate` (`pl.Expr | pl.Series | str | Iterable[pl.Expr | pl.Series | str] | bool | list[bool] | np.ndarray`) — Boolean expression, column, mask, or predicates accepted by Polars LazyFrame.filter.

**Returns**

- (`Self`) — Measurement of the same type with a lazily filtered scan table.
- (`Self`) — The manifest and diagnostics still describe the full source.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L71-L93)

### particula.data.measurement.Measurement.flag_summary

*method*

```python
def flag_summary() -> pl.LazyFrame
```

Count each vendor condition once per scan.

**Returns**

- (`pl.LazyFrame`) — Lazy table of flag labels, affected and total scan counts, and
- (`pl.LazyFrame`) — the first and last UTC time at which each flag appeared.

**Raises**

- `ValueError` — The profile does not declare vendor flags.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L147-L174)

### particula.data.measurement.Measurement.quality_summary

*method*

```python
def quality_summary(field: str, *, exclude_flags: str | Iterable[str] = (), exclude_repaired: bool = False) -> pl.LazyFrame
```

Summarize a scalar field after explicit source-quality exclusions.

**Parameters**

- `field` (`str`) — Name of a scalar field declared in the profile.
- `exclude_flags` (`str | Iterable[str]`) (default: `()`) — Omit scans carrying any of these vendor flags.
- `exclude_repaired` (`bool`) (default: `False`) — Omit scans reconstructed from logger fragments.

**Returns**

- (`pl.LazyFrame`) — One lazy row with the source, exclusion policy, scan and sample
- (`pl.LazyFrame`) — counts, and UTC bounds of excluded scans. The mean weights each
- (`pl.LazyFrame`) — retained, non-null sample equally and uses the field's output unit.

**Raises**

- `ValueError` — The field is undeclared or the requested flag
exclusions require a missing vendor status field.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L95-L145)

## particula.data.measurement.MobilityNumberSpectrum

*class*

```python
class MobilityNumberSpectrum(Measurement)
```

Bases: `particula.data.measurement.Measurement`

Number per log10 electrical-mobility diameter on a declared grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L211-L255)

### particula.data.measurement.MobilityNumberSpectrum.grids

*attribute* · *instance attribute*

```python
grids = manifest['grids']
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L43-L43)

### particula.data.measurement.MobilityNumberSpectrum.lazy_frame

*attribute* · *instance attribute*

```python
lazy_frame = scans.with_columns(pl.lit(manifest['source_sha256']).alias('source_sha256'))
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L38-L40)

### particula.data.measurement.MobilityNumberSpectrum.lazy_issues

*attribute* · *instance attribute*

```python
lazy_issues = issues
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L41-L41)

### particula.data.measurement.MobilityNumberSpectrum.manifest

*attribute* · *instance attribute*

```python
manifest = manifest
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L42-L42)

### particula.data.measurement.MobilityNumberSpectrum.frame

*property*

```python
frame: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect the selected scans in memory, including their source IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L62-L64)

### particula.data.measurement.MobilityNumberSpectrum.issues

*property*

```python
issues: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L67-L69)

### particula.data.measurement.MobilityNumberSpectrum.__init__

*method*

```python
def __init__(scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict) -> None
```

Require a number-per-decade spectrum and mobility diameter grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L214-L227)

### particula.data.measurement.MobilityNumberSpectrum.available_operations

*method*

```python
def available_operations() -> dict[str, tuple[str, ...]]
```

Report the spectrum integral.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L253-L255)

### particula.data.measurement.MobilityNumberSpectrum.integrated_number

*method*

```python
def integrated_number() -> pl.LazyFrame
```

Integrate number concentration over measured mobility diameters.

Each channel contributes its number density per decade multiplied by
the reciprocal of the grid's reported channels per decade.

**Returns**

- (`pl.LazyFrame`) — Lazy table of scan IDs, UTC times and integrated number in the
- (`pl.LazyFrame`) — profile's concentration unit. A missing channel yields a null
- (`pl.LazyFrame`) — total for that scan.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L229-L251)

### particula.data.measurement.MobilityNumberSpectrum.filter

*method*

```python
def filter(predicate: pl.Expr | pl.Series | str | Iterable[pl.Expr | pl.Series | str] | bool | list[bool] | np.ndarray) -> Self
```

Inherited from: `particula.data.measurement.Measurement`

Select scans while retaining the original source diagnostics.

**Parameters**

- `predicate` (`pl.Expr | pl.Series | str | Iterable[pl.Expr | pl.Series | str] | bool | list[bool] | np.ndarray`) — Boolean expression, column, mask, or predicates accepted by Polars LazyFrame.filter.

**Returns**

- (`Self`) — Measurement of the same type with a lazily filtered scan table.
- (`Self`) — The manifest and diagnostics still describe the full source.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L71-L93)

### particula.data.measurement.MobilityNumberSpectrum.flag_summary

*method*

```python
def flag_summary() -> pl.LazyFrame
```

Inherited from: `particula.data.measurement.Measurement`

Count each vendor condition once per scan.

**Returns**

- (`pl.LazyFrame`) — Lazy table of flag labels, affected and total scan counts, and
- (`pl.LazyFrame`) — the first and last UTC time at which each flag appeared.

**Raises**

- `ValueError` — The profile does not declare vendor flags.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L147-L174)

### particula.data.measurement.MobilityNumberSpectrum.quality_summary

*method*

```python
def quality_summary(field: str, *, exclude_flags: str | Iterable[str] = (), exclude_repaired: bool = False) -> pl.LazyFrame
```

Inherited from: `particula.data.measurement.Measurement`

Summarize a scalar field after explicit source-quality exclusions.

**Parameters**

- `field` (`str`) — Name of a scalar field declared in the profile.
- `exclude_flags` (`str | Iterable[str]`) (default: `()`) — Omit scans carrying any of these vendor flags.
- `exclude_repaired` (`bool`) (default: `False`) — Omit scans reconstructed from logger fragments.

**Returns**

- (`pl.LazyFrame`) — One lazy row with the source, exclusion policy, scan and sample
- (`pl.LazyFrame`) — counts, and UTC bounds of excluded scans. The mean weights each
- (`pl.LazyFrame`) — retained, non-null sample equally and uses the field's output unit.

**Raises**

- `ValueError` — The field is undeclared or the requested flag
exclusions require a missing vendor status field.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L95-L145)

## particula.data.measurement.ParticleCountRate

*class*

```python
class ParticleCountRate(Measurement)
```

Bases: `particula.data.measurement.Measurement`

A measurement containing detector count rate in declared units.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L187-L208)

### particula.data.measurement.ParticleCountRate.grids

*attribute* · *instance attribute*

```python
grids = manifest['grids']
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L43-L43)

### particula.data.measurement.ParticleCountRate.lazy_frame

*attribute* · *instance attribute*

```python
lazy_frame = scans.with_columns(pl.lit(manifest['source_sha256']).alias('source_sha256'))
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L38-L40)

### particula.data.measurement.ParticleCountRate.lazy_issues

*attribute* · *instance attribute*

```python
lazy_issues = issues
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L41-L41)

### particula.data.measurement.ParticleCountRate.manifest

*attribute* · *instance attribute*

```python
manifest = manifest
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L42-L42)

### particula.data.measurement.ParticleCountRate.frame

*property*

```python
frame: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect the selected scans in memory, including their source IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L62-L64)

### particula.data.measurement.ParticleCountRate.issues

*property*

```python
issues: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L67-L69)

### particula.data.measurement.ParticleCountRate.__init__

*method*

```python
def __init__(scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict) -> None
```

Require a profile that declares a detector count-rate field.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L190-L201)

### particula.data.measurement.ParticleCountRate.available_operations

*method*

```python
def available_operations() -> dict[str, tuple[str, ...]]
```

Include sample flow as a prerequisite for number concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L203-L208)

### particula.data.measurement.ParticleCountRate.filter

*method*

```python
def filter(predicate: pl.Expr | pl.Series | str | Iterable[pl.Expr | pl.Series | str] | bool | list[bool] | np.ndarray) -> Self
```

Inherited from: `particula.data.measurement.Measurement`

Select scans while retaining the original source diagnostics.

**Parameters**

- `predicate` (`pl.Expr | pl.Series | str | Iterable[pl.Expr | pl.Series | str] | bool | list[bool] | np.ndarray`) — Boolean expression, column, mask, or predicates accepted by Polars LazyFrame.filter.

**Returns**

- (`Self`) — Measurement of the same type with a lazily filtered scan table.
- (`Self`) — The manifest and diagnostics still describe the full source.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L71-L93)

### particula.data.measurement.ParticleCountRate.flag_summary

*method*

```python
def flag_summary() -> pl.LazyFrame
```

Inherited from: `particula.data.measurement.Measurement`

Count each vendor condition once per scan.

**Returns**

- (`pl.LazyFrame`) — Lazy table of flag labels, affected and total scan counts, and
- (`pl.LazyFrame`) — the first and last UTC time at which each flag appeared.

**Raises**

- `ValueError` — The profile does not declare vendor flags.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L147-L174)

### particula.data.measurement.ParticleCountRate.quality_summary

*method*

```python
def quality_summary(field: str, *, exclude_flags: str | Iterable[str] = (), exclude_repaired: bool = False) -> pl.LazyFrame
```

Inherited from: `particula.data.measurement.Measurement`

Summarize a scalar field after explicit source-quality exclusions.

**Parameters**

- `field` (`str`) — Name of a scalar field declared in the profile.
- `exclude_flags` (`str | Iterable[str]`) (default: `()`) — Omit scans carrying any of these vendor flags.
- `exclude_repaired` (`bool`) (default: `False`) — Omit scans reconstructed from logger fragments.

**Returns**

- (`pl.LazyFrame`) — One lazy row with the source, exclusion policy, scan and sample
- (`pl.LazyFrame`) — counts, and UTC bounds of excluded scans. The mean weights each
- (`pl.LazyFrame`) — retained, non-null sample equally and uses the field's output unit.

**Raises**

- `ValueError` — The field is undeclared or the requested flag
exclusions require a missing vendor status field.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L95-L145)

## particula.data.measurement.measurement

*function*

```python
def measurement(scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict) -> Measurement
```

Select the product by measured quantities and coordinate basis.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L258-L271)
