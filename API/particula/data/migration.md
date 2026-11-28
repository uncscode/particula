# particula.data.migration

*module*

Translate beta stream settings while preserving the original filter audit.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L1-L1)

## particula.data.migration._AIM_FIELDS

*attribute* · *module attribute*

```python
_AIM_FIELDS = {'Lower Size (nm)': ('lower_size', 'length', 'nm', 'm'), 'Upper Size (nm)': ('upper_size', 'length', 'nm', 'm'), 'Sample Temp (C)': ('sample_temperature', 'temperature', 'degC', 'K'), 'Sample Pressure (kPa)': ('sample_pressure', 'pressure', 'kPa', 'Pa'), 'Relative Humidity (%)': ('relative_humidity', 'relative_humidity', '%', '%'), 'Median (nm)': ('median_diameter', 'length', 'nm', 'm'), 'Mean (nm)': ('mean_diameter', 'length', 'nm', 'm'), 'Geo. Mean (nm)': ('geometric_mean_diameter', 'length', 'nm', 'm'), 'Mode (nm)': ('mode_diameter', 'length', 'nm', 'm'), 'Geo. Std. Dev.': ('geometric_standard_deviation', 'ratio', '1', '1'), 'Total Conc. (#/cm³)': ('total_number', 'number_concentration', '1/cm^3', '1/m^3')}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L25-L47)

## particula.data.migration._SUPPORTED

*attribute* · *module attribute*

```python
_SUPPORTED = {'relative_data_folder', 'filename_regex', 'MIN_SIZE_BYTES', 'data_loading_function', 'header_row', 'data_checks', 'data_column', 'data_header', 'time_column', 'time_format', 'delimiter', 'time_shift_seconds', 'timezone_identifier', 'data_sizer_reader'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L8-L23)

## particula.data.migration.Migration

*class* · *dataclass*

```python
class Migration
```

Result of translating a beta stream configuration.

**Attributes**

- `profile` (`dict`) — Candidate import profile with the original settings under `legacy_settings`.
- `errors` (`list[ProfileError]`) — Diagnostics for settings requiring a migration decision.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L50-L72)

### particula.data.migration.Migration.errors

*attribute* · *instance attribute*

```python
errors: list[ProfileError]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L61-L61)

### particula.data.migration.Migration.profile

*attribute* · *instance attribute*

```python
profile: dict
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L60-L60)

### particula.data.migration.Migration.__init__

*method*

```python
def __init__(profile: dict, errors: list[ProfileError]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L0-L0)

### particula.data.migration.Migration.resolved_profile

*method*

```python
def resolved_profile() -> dict
```

Return the validated profile when every legacy option is resolved.

**Raises**

- `ProfileError` — Migration recorded an unresolved setting or the
candidate profile fails validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L63-L72)

## particula.data.migration._legacy_clock

*function*

```python
def _legacy_clock(settings)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L163-L182)

## particula.data.migration._legacy_header

*function*

```python
def _legacy_header(settings, on_bad_record, issue)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L114-L138)

## particula.data.migration._scalar_fields

*function*

```python
def _scalar_fields(settings, overrides, errors)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L75-L111)

## particula.data.migration._sizer_options

*function*

```python
def _sizer_options(profile, settings, issue)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L141-L160)

## particula.data.migration.migrate_settings

*function*

```python
def migrate_settings(settings: dict, *, fields: dict | None = None, on_bad_record: str | None = None) -> Migration
```

Translate a beta 1D or AIM 2D settings mapping.

**Parameters**

- `settings` (`dict`) — Beta scalar or sizer stream settings, including its clock, column selectors and source discovery rules.
- `fields` (`dict | None`) (default: `None`) — Scalar field definitions keyed by legacy output label. Supply the quantity, source unit and output unit for unfamiliar channels.
- `on_bad_record` (`str | None`) (default: `None`) — Profile policy for malformed rows. An explicit choice is required to replace legacy character and length filters.

**Returns**

- (`Migration`) — Candidate profile, original settings and diagnostics for unresolved
- (`Migration`) — choices. Call resolved_profile() before using the result for import.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/migration.py#L185-L247)
