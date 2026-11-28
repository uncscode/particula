# particula.instruments.mobility_sizer

*module*

Import particle number spectra resolved by electrical-mobility diameter.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L1-L1)

## particula.instruments.mobility_sizer.MobilitySizer

*class* · *dataclass*

```python
class MobilitySizer(Instrument)
```

Bases: `particula.instruments.instrument.Instrument`

Read an electrical-mobility sizer export with a configurable profile.

**Parameters**

- `profile` (`ProfileSource`) (default: `'smps-aim'`) — Import profile accepted by Instrument. The default AIM number export uses mobility diameter in metres and number concentration per decade of diameter in particles per cubic metre.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L11-L21)

### particula.instruments.mobility_sizer.MobilitySizer.profile

*attribute* · *class attribute* · *instance attribute*

```python
profile: ProfileSource = 'smps-aim'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L21-L21)

### particula.instruments.mobility_sizer.MobilitySizer.__init__

*method*

```python
def __init__(profile: ProfileSource = 'smps-aim') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L0-L0)

### particula.instruments.mobility_sizer.MobilitySizer.__post_init__

*method*

```python
def __post_init__() -> None
```

Inherited from: `particula.instruments.instrument.Instrument`

Resolve the supplied profile before reading any sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L25-L27)

### particula.instruments.mobility_sizer.MobilitySizer.read

*method*

```python
def read(source: str | PathLike[str], **limits: int) -> Measurement
```

Inherited from: `particula.instruments.instrument.Instrument`

Read one export into memory using this instrument's profile.

**Parameters**

- `source` (`str | PathLike[str]`) — Path to the delimited instrument export.
- `**limits` (`int`) (default: `{}`) — Optional chunk, record and batch limits accepted by MeasurementReader.

**Returns**

- (`Measurement`) — Measurement with scans in the profile's output units, source
- (`Measurement`) — diagnostics and the import manifest.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L29-L41)

## particula.instruments.mobility_sizer.read_mobility_sizer

*function*

```python
def read_mobility_sizer(source: str | PathLike[str], *, profile: ProfileSource = 'smps-aim', **limits: int) -> MobilityNumberSpectrum
```

Read number spectra and verify the electrical-mobility diameter axis.

**Parameters**

- `source` (`str | PathLike[str]`) — Path to the delimited sizer export.
- `profile` (`ProfileSource`) (default: `'smps-aim'`) — Built-in profile ID, JSON or YAML file path, or profile mapping describing the spectrum and diameter axis.
- `**limits` (`int`) (default: `{}`) — Optional MeasurementReader chunk, record and batch limits.

**Returns**

- (`MobilityNumberSpectrum`) — Number concentration per decade of mobility diameter, with each
- (`MobilityNumberSpectrum`) — scan linked to its diameter grid and declared output units.

**Raises**

- `ValueError` — The profile produces a different measurement type.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L24-L48)
