# particula.instruments.particle_counter

*module*

Import particle detector count rates and associated logger channels.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L1-L1)

## particula.instruments.particle_counter.ParticleCounter

*class* · *dataclass*

```python
class ParticleCounter(Instrument)
```

Bases: `particula.instruments.instrument.Instrument`

Read a particle counter export with a configurable profile.

**Parameters**

- `profile` (`ProfileSource`) (default: `'cpc-3010'`) — Import profile accepted by Instrument. The default 3010 logger profile reports detector count rate in counts per second and logger temperature in kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L11-L21)

### particula.instruments.particle_counter.ParticleCounter.profile

*attribute* · *class attribute* · *instance attribute*

```python
profile: ProfileSource = 'cpc-3010'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L21-L21)

### particula.instruments.particle_counter.ParticleCounter.__init__

*method*

```python
def __init__(profile: ProfileSource = 'cpc-3010') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/particle_counter.py#L0-L0)

### particula.instruments.particle_counter.ParticleCounter.__post_init__

*method*

```python
def __post_init__() -> None
```

Inherited from: `particula.instruments.instrument.Instrument`

Resolve the supplied profile before reading any sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L25-L27)

### particula.instruments.particle_counter.ParticleCounter.read

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

## particula.instruments.particle_counter.read_particle_counter

*function*

```python
def read_particle_counter(source: str | PathLike[str], *, profile: ProfileSource = 'cpc-3010', **limits: int) -> ParticleCountRate
```

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
