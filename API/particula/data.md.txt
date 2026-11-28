# particula.data

*module*

Import unit-described measurements with the optional data extra.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/__init__.py#L1-L1)

## particula.data.Measurement

*class*

```python
class Measurement
```

Re-exported from: `particula.data.measurement`

Measurements from one source, with a row per accepted scan.

**Attributes**

- `lazy_frame` — Scan columns in profile output units, with UTC timestamps. The source digest and first physical line identify each scan.
- `lazy_issues` — Parse diagnostics with inclusive physical line spans.
- `manifest` — Import provenance, including the profile and source layouts.
- `grids` — Coordinate definitions keyed by the grid ID stored on each scan that contains spectral or array data.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L12-L184)

### particula.data.Measurement.grids

*attribute* · *instance attribute*

```python
grids = manifest['grids']
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L43-L43)

### particula.data.Measurement.lazy_frame

*attribute* · *instance attribute*

```python
lazy_frame = scans.with_columns(pl.lit(manifest['source_sha256']).alias('source_sha256'))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L38-L40)

### particula.data.Measurement.lazy_issues

*attribute* · *instance attribute*

```python
lazy_issues = issues
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L41-L41)

### particula.data.Measurement.manifest

*attribute* · *instance attribute*

```python
manifest = manifest
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L42-L42)

### particula.data.Measurement.frame

*property*

```python
frame: pl.DataFrame
```

Collect the selected scans in memory, including their source IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L62-L64)

### particula.data.Measurement.issues

*property*

```python
issues: pl.DataFrame
```

Collect source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L67-L69)

### particula.data.Measurement.__init__

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

### particula.data.Measurement.available_operations

*method*

```python
def available_operations() -> dict[str, tuple[str, ...]]
```

Map compatible operation names to their required input names.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L176-L184)

### particula.data.Measurement.filter

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

### particula.data.Measurement.flag_summary

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

### particula.data.Measurement.quality_summary

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

## particula.data.MeasurementReader

*class*

```python
class MeasurementReader
```

Re-exported from: `particula.data.reader`

Read one source in batches while tracking layouts and diagnostics.

**Attributes**

- `manifest` (`dict | None`) — Import provenance, including the source digest, grids and scan count. Available only after batches() reaches end of file.
- `segment` — Zero-based source layout index for the current batch.
- `issue_counts` (`Counter[str]`) — Number of diagnostics emitted per code during decoding.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L335-L528)

### particula.data.MeasurementReader.count

*attribute* · *instance attribute*

```python
count = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L378-L378)

### particula.data.MeasurementReader.issue_counts

*attribute* · *instance attribute*

```python
issue_counts: Counter[str] = Counter()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L377-L377)

### particula.data.MeasurementReader.manifest

*attribute* · *instance attribute*

```python
manifest: dict | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L379-L379)

### particula.data.MeasurementReader.profile

*attribute* · *instance attribute*

```python
profile = load_profile(profile)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L372-L372)

### particula.data.MeasurementReader.segment

*attribute* · *instance attribute*

```python
segment = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L380-L380)

### particula.data.MeasurementReader.source

*attribute* · *instance attribute*

```python
source = Path(source)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L371-L371)

### particula.data.MeasurementReader.__init__

*method*

```python
def __init__(source: str | os.PathLike[str], profile: ProfileSource, *, input_chunk_bytes: int = 2 ** 20, batch_scans: int = 32768, batch_bytes: int = 2 ** 23, max_record_bytes: int = 2 ** 22) -> None
```

Configure the source and batching limits.

**Parameters**

- `source` (`str | os.PathLike[str]`) — Path to the delimited instrument export.
- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML profile.
- `input_chunk_bytes` (`int`) (default: `2 ** 20`) — Maximum bytes requested per raw input read.
- `batch_scans` (`int`) (default: `32768`) — Target maximum scan rows per emitted batch.
- `batch_bytes` (`int`) (default: `2 ** 23`) — Target working size of a batch in bytes. A single record and its diagnostics can exceed this target.
- `max_record_bytes` (`int`) (default: `2 ** 22`) — Maximum raw bytes in one logical record, including embedded newlines and any repaired fragments.

**Raises**

- `ValueError` — A limit is not a positive integer.
- `ProfileError` — The profile cannot be loaded or validated.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L345-L380)

### particula.data.MeasurementReader._decoded

*method*

```python
def _decoded(records, binding)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L459-L508)

### particula.data.MeasurementReader._issue

*method*

```python
def _issue(code, start, end, message)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L521-L528)

### particula.data.MeasurementReader._missing

*method*

```python
def _missing(row, binding, start, end)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L510-L519)

### particula.data.MeasurementReader.batches

*method*

```python
def batches() -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]
```

Read scans in source order and finalize the manifest at end of file.

**Yields**

- (`pl.DataFrame`) — Pairs of scan and diagnostic frames for one source layout.
- (`pl.DataFrame`) — Either frame may be empty. Scan times are UTC nanoseconds, and
- (`tuple[pl.DataFrame, pl.DataFrame]`) — source spans use inclusive, one-based physical line numbers.

**Raises**

- `ProfileError` — A layout or record fails the profile's input rules.
- `ValueError` — The source size or modification time changes during
import.
- `OSError` — The source cannot be opened or read.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L382-L457)

## particula.data.MobilityNumberSpectrum

*class*

```python
class MobilityNumberSpectrum(Measurement)
```

Bases: `particula.data.measurement.Measurement`

Re-exported from: `particula.data.measurement`

Number per log10 electrical-mobility diameter on a declared grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L211-L255)

### particula.data.MobilityNumberSpectrum.grids

*attribute* · *instance attribute*

```python
grids = manifest['grids']
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L43-L43)

### particula.data.MobilityNumberSpectrum.lazy_frame

*attribute* · *instance attribute*

```python
lazy_frame = scans.with_columns(pl.lit(manifest['source_sha256']).alias('source_sha256'))
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L38-L40)

### particula.data.MobilityNumberSpectrum.lazy_issues

*attribute* · *instance attribute*

```python
lazy_issues = issues
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L41-L41)

### particula.data.MobilityNumberSpectrum.manifest

*attribute* · *instance attribute*

```python
manifest = manifest
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L42-L42)

### particula.data.MobilityNumberSpectrum.frame

*property*

```python
frame: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect the selected scans in memory, including their source IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L62-L64)

### particula.data.MobilityNumberSpectrum.issues

*property*

```python
issues: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L67-L69)

### particula.data.MobilityNumberSpectrum.__init__

*method*

```python
def __init__(scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict) -> None
```

Require a number-per-decade spectrum and mobility diameter grid.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L214-L227)

### particula.data.MobilityNumberSpectrum.available_operations

*method*

```python
def available_operations() -> dict[str, tuple[str, ...]]
```

Report the spectrum integral.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L253-L255)

### particula.data.MobilityNumberSpectrum.integrated_number

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

### particula.data.MobilityNumberSpectrum.filter

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

### particula.data.MobilityNumberSpectrum.flag_summary

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

### particula.data.MobilityNumberSpectrum.quality_summary

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

## particula.data.ParticleCountRate

*class*

```python
class ParticleCountRate(Measurement)
```

Bases: `particula.data.measurement.Measurement`

Re-exported from: `particula.data.measurement`

A measurement containing detector count rate in declared units.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L187-L208)

### particula.data.ParticleCountRate.grids

*attribute* · *instance attribute*

```python
grids = manifest['grids']
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L43-L43)

### particula.data.ParticleCountRate.lazy_frame

*attribute* · *instance attribute*

```python
lazy_frame = scans.with_columns(pl.lit(manifest['source_sha256']).alias('source_sha256'))
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L38-L40)

### particula.data.ParticleCountRate.lazy_issues

*attribute* · *instance attribute*

```python
lazy_issues = issues
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L41-L41)

### particula.data.ParticleCountRate.manifest

*attribute* · *instance attribute*

```python
manifest = manifest
```

Inherited from: `particula.data.measurement.Measurement`

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L42-L42)

### particula.data.ParticleCountRate.frame

*property*

```python
frame: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect the selected scans in memory, including their source IDs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L62-L64)

### particula.data.ParticleCountRate.issues

*property*

```python
issues: pl.DataFrame
```

Inherited from: `particula.data.measurement.Measurement`

Collect source-located diagnostics.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L67-L69)

### particula.data.ParticleCountRate.__init__

*method*

```python
def __init__(scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict) -> None
```

Require a profile that declares a detector count-rate field.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L190-L201)

### particula.data.ParticleCountRate.available_operations

*method*

```python
def available_operations() -> dict[str, tuple[str, ...]]
```

Include sample flow as a prerequisite for number concentration.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/measurement.py#L203-L208)

### particula.data.ParticleCountRate.filter

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

### particula.data.ParticleCountRate.flag_summary

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

### particula.data.ParticleCountRate.quality_summary

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

## particula.data.ProfileError

*class*

```python
class ProfileError(ValueError)
```

Bases: `ValueError`

Re-exported from: `particula.data.profile`

A machine-readable diagnostic for a profile or its source binding.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L34-L67)

### particula.data.ProfileError.code

*attribute* · *instance attribute*

```python
code = code
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L64-L64)

### particula.data.ProfileError.expected

*attribute* · *instance attribute*

```python
expected = expected
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L61-L61)

### particula.data.ProfileError.message

*attribute* · *instance attribute*

```python
message = message
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L66-L66)

### particula.data.ProfileError.observed

*attribute* · *instance attribute*

```python
observed = observed
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L60-L60)

### particula.data.ProfileError.path

*attribute* · *instance attribute*

```python
path = path
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L65-L65)

### particula.data.ProfileError.severity

*attribute* · *instance attribute*

```python
severity = 'error'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L59-L59)

### particula.data.ProfileError.source_line_end

*attribute* · *instance attribute*

```python
source_line_end = source_line_end
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L63-L63)

### particula.data.ProfileError.source_line_start

*attribute* · *instance attribute*

```python
source_line_start = source_line_start
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L62-L62)

### particula.data.ProfileError.__init__

*method*

```python
def __init__(code: str, path: str, message: str, *, observed: object = None, expected: object = None, source_line_start: int | None = None, source_line_end: int | None = None) -> None
```

Describe a profile error and its location.

**Parameters**

- `code` (`str`) — Machine-readable diagnostic identifier.
- `path` (`str`) — JSON Pointer within the profile, or empty for its root.
- `message` (`str`) — Human-readable explanation.
- `observed` (`object`) (default: `None`) — Value that failed validation, when available.
- `expected` (`object`) (default: `None`) — Constraint or expected value, when available.
- `source_line_start` (`int | None`) (default: `None`) — First physical source line, counted from one.
- `source_line_end` (`int | None`) (default: `None`) — Last physical source line, inclusive.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L37-L67)

## particula.data.describe_profile

*function*

```python
def describe_profile(profile: ProfileSource | None = None) -> dict
```

Re-exported from: `particula.data.profile`

Describe the profile format or resolve a particular profile.

**Parameters**

- `profile` (`ProfileSource | None`) (default: `None`) — Built-in ID, mapping, or path to a JSON or YAML file. Omit to retrieve the installed JSON Schema.

**Returns**

- (`dict`) — A fresh dictionary containing the schema or the validated profile,
- (`dict`) — including its declared quantities, units and clock interpretation.

**Raises**

- `ProfileError` — The supplied profile cannot be loaded or validated.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L127-L143)

## particula.data.import_to_parquet

*function*

```python
def import_to_parquet(source: str | os.PathLike[str], destination: str | os.PathLike[str], *, profile: ProfileSource, **limits: int) -> Measurement
```

Re-exported from: `particula.data.storage`

Write bounded Parquet parts and atomically publish a completed import.

**Parameters**

- `source` (`str | os.PathLike[str]`) — Path to a delimited instrument export.
- `destination` (`str | os.PathLike[str]`) — Directory to publish after all parts have been written.
- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML profile.
- `**limits` (`int`) (default: `{}`) — Optional MeasurementReader chunk, record and batch limits.

**Returns**

- (`Measurement`) — Disk-backed measurement with lazy scans and source diagnostics.
- (`Measurement`) — An existing dataset is reused when the source digest, profile digest
- (`Measurement`) — and parser version match.

**Raises**

- `FileExistsError` — The destination holds an incomplete or different
dataset.
- `ProfileError` — The profile or source violates the input rules.
- `ValueError` — A reader limit is invalid or the source changes during
import.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/storage.py#L107-L183)

## particula.data.inspect_source

*function*

```python
def inspect_source(source: str | os.PathLike[str], profile: ProfileSource) -> dict
```

Re-exported from: `particula.data.reader`

Inspect the preamble, header and first observation in a delimited export.

**Parameters**

- `source` (`str | os.PathLike[str]`) — Path to the instrument export.
- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML profile.

**Returns**

- (`dict`) — Encoding, header, preamble and coordinate grids for the first layout.
- (`dict`) — The clock example contains raw tokens and its starting physical line.
- (`dict`) — Both are null when the layout has no following record.

**Raises**

- `ProfileError` — The profile or source layout fails validation.
- `OSError` — The source cannot be opened or read.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L207-L246)

## particula.data.migrate_settings

*function*

```python
def migrate_settings(settings: dict, *, fields: dict | None = None, on_bad_record: str | None = None) -> Migration
```

Re-exported from: `particula.data.migration`

Translate a beta 1D or AIM 2D settings mapping.

**Parameters**

- `settings` (`dict`) — Beta scalar or sizer stream settings, including its clock, column selectors and source discovery rules.
- `fields` (`dict | None`) (default: `None`) — Scalar field definitions keyed by legacy output label. Supply the quantity, source unit and output unit for unfamiliar channels.
- `on_bad_record` (`str | None`) (default: `None`) — Profile policy for malformed rows. An explicit choice is required to replace legacy character and length filters.

**Returns**

- (`Migration`) — Candidate profile, original settings and diagnostics for unresolved
- (`Migration`) — choices. Call resolved_profile() before using the result for import.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L185-L247)

## particula.data.open_measurement

*function*

```python
def open_measurement(destination: str | os.PathLike[str]) -> Measurement
```

Re-exported from: `particula.data.storage`

Open a completed Parquet measurement for lazy analysis.

**Parameters**

- `destination` (`str | os.PathLike[str]`) — Dataset directory created by import_to_parquet.

**Returns**

- (`Measurement`) — Measurement with lazy scans and diagnostics, validated profile units
- (`Measurement`) — and coordinates from the stored manifest.

**Raises**

- `OSError` — The manifest cannot be opened or read.
- `ValueError` — The format, profile or grid digests, part inventory, or
scan layout fails validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/storage.py#L49-L94)

## particula.data.read_measurement

*function*

```python
def read_measurement(source: str | os.PathLike[str], *, profile: ProfileSource, **limits: int) -> Measurement
```

Re-exported from: `particula.data.storage`

Read a single instrument export into memory.

**Parameters**

- `source` (`str | os.PathLike[str]`) — Path to the delimited export.
- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML profile.
- `**limits` (`int`) (default: `{}`) — Optional MeasurementReader chunk, record and batch limits.

**Returns**

- (`Measurement`) — Measurement selected from the profile's quantities and coordinate
- (`Measurement`) — basis. Scans use the declared output units and include UTC timestamps.

**Raises**

- `ProfileError` — The profile or source violates the input rules.
- `ValueError` — A reader limit is invalid or the source changes during
import.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/storage.py#L21-L46)

## particula.data.validate_profile

*function*

```python
def validate_profile(profile: ProfileSource, *, source: str | PathLike[str] | None = None) -> ValidationResult
```

Re-exported from: `particula.data.profile`

Check a profile and optionally bind it to the first source layout.

**Parameters**

- `profile` (`ProfileSource`) — Built-in ID, mapping, or path to a JSON or YAML file.
- `source` (`str | PathLike[str] | None`) (default: `None`) — Optional delimited export for checking column selectors, coordinate definitions and preamble requirements.

**Returns**

- (`ValidationResult`) — Validation result with the first profile diagnostic, or an empty
- (`ValidationResult`) — error tuple on success.

**Raises**

- `OSError` — The supplied source cannot be opened or read.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/profile.py#L255-L280)

**Modules**

- `particula.data._binding`
- `particula.data._exports`
- `particula.data.mcd43`
- `particula.data.measurement`
- `particula.data.migration`
- `particula.data.power`
- `particula.data.profile`
- `particula.data.reader`
- `particula.data.storage`
