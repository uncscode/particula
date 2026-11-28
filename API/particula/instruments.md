# particula.instruments

*module*

Read instrument exports using built-in or user-defined profiles.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/__init__.py#L1-L1)

## particula.instruments.Instrument

*class* · *dataclass*

```python
class Instrument
```

Re-exported from: `particula.instruments.instrument`

Read delimited exports that share one profile.

**Parameters**

- `profile` (`ProfileSource`) — Built-in profile ID, JSON or YAML file path, or profile mapping. Construction validates and stores a copy of the profile.

**Raises**

- `ProfileError` — The profile cannot be loaded or fails validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L11-L41)

### particula.instruments.Instrument.profile

*attribute* · *instance attribute*

```python
profile: ProfileSource
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L23-L23)

### particula.instruments.Instrument.__init__

*method*

```python
def __init__(profile: ProfileSource) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L0-L0)

### particula.instruments.Instrument.__post_init__

*method*

```python
def __post_init__() -> None
```

Resolve the supplied profile before reading any sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L25-L27)

### particula.instruments.Instrument.read

*method*

```python
def read(source: str | PathLike[str], **limits: int) -> Measurement
```

Read one export into memory using this instrument's profile.

**Parameters**

- `source` (`str | PathLike[str]`) — Path to the delimited instrument export.
- `**limits` (`int`) (default: `{}`) — Optional chunk, record and batch limits accepted by MeasurementReader.

**Returns**

- (`Measurement`) — Measurement with scans in the profile's output units, source
- (`Measurement`) — diagnostics and the import manifest.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L29-L41)

## particula.instruments.MobilitySizer

*class* · *dataclass*

```python
class MobilitySizer(Instrument)
```

Bases: `particula.instruments.instrument.Instrument`

Re-exported from: `particula.instruments.mobility_sizer`

Read an electrical-mobility sizer export with a configurable profile.

**Parameters**

- `profile` (`ProfileSource`) (default: `'smps-aim'`) — Import profile accepted by Instrument. The default AIM number export uses mobility diameter in metres and number concentration per decade of diameter in particles per cubic metre.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L11-L21)

### particula.instruments.MobilitySizer.profile

*attribute* · *class attribute* · *instance attribute*

```python
profile: ProfileSource = 'smps-aim'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L21-L21)

### particula.instruments.MobilitySizer.__init__

*method*

```python
def __init__(profile: ProfileSource = 'smps-aim') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/mobility_sizer.py#L0-L0)

### particula.instruments.MobilitySizer.__post_init__

*method*

```python
def __post_init__() -> None
```

Inherited from: `particula.instruments.instrument.Instrument`

Resolve the supplied profile before reading any sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L25-L27)

### particula.instruments.MobilitySizer.read

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

## particula.instruments.ParticleCounter

*class* · *dataclass*

```python
class ParticleCounter(Instrument)
```

Bases: `particula.instruments.instrument.Instrument`

Re-exported from: `particula.instruments.particle_counter`

Read a particle counter export with a configurable profile.

**Parameters**

- `profile` (`ProfileSource`) (default: `'cpc-3010'`) — Import profile accepted by Instrument. The default 3010 logger profile reports detector count rate in counts per second and logger temperature in kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L11-L21)

### particula.instruments.ParticleCounter.profile

*attribute* · *class attribute* · *instance attribute*

```python
profile: ProfileSource = 'cpc-3010'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L21-L21)

### particula.instruments.ParticleCounter.__init__

*method*

```python
def __init__(profile: ProfileSource = 'cpc-3010') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L0-L0)

### particula.instruments.ParticleCounter.__post_init__

*method*

```python
def __post_init__() -> None
```

Inherited from: `particula.instruments.instrument.Instrument`

Resolve the supplied profile before reading any sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L25-L27)

### particula.instruments.ParticleCounter.read

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

## particula.instruments.read_mobility_sizer

*function*

```python
def read_mobility_sizer(source: str | PathLike[str], *, profile: ProfileSource = 'smps-aim', **limits: int) -> MobilityNumberSpectrum
```

Re-exported from: `particula.instruments.mobility_sizer`

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

## particula.instruments.read_particle_counter

*function*

```python
def read_particle_counter(source: str | PathLike[str], *, profile: ProfileSource = 'cpc-3010', **limits: int) -> ParticleCountRate
```

Re-exported from: `particula.instruments.particle_counter`

Read a particle counter export and verify its count-rate field.

**Parameters**

- `source` (`str | PathLike[str]`) — Path to the delimited counter export.
- `profile` (`ProfileSource`) (default: `'cpc-3010'`) — Built-in profile ID, JSON or YAML file path, or profile mapping describing the detector channels and their units.
- `**limits` (`int`) (default: `{}`) — Optional MeasurementReader chunk, record and batch limits.

**Returns**

- (`ParticleCountRate`) — Count-rate scans in the profile's output units with source diagnostics.

**Raises**

- `ValueError` — The profile produces a different measurement type.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L24-L47)

**Modules**

- `particula.instruments.instrument`
- `particula.instruments.mobility_sizer`
- `particula.instruments.particle_counter`
