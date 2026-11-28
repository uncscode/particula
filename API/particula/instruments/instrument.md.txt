# particula.instruments.instrument

*module*

Read measurements from an instrument with a declared import profile.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L1-L1)

## particula.instruments.instrument.Instrument

*class* · *dataclass*

```python
class Instrument
```

Read delimited exports that share one profile.

**Parameters**

- `profile` (`ProfileSource`) — Built-in profile ID, JSON or YAML file path, or profile mapping. Construction validates and stores a copy of the profile.

**Raises**

- `ProfileError` — The profile cannot be loaded or fails validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L11-L41)

### particula.instruments.instrument.Instrument.profile

*attribute* · *instance attribute*

```python
profile: ProfileSource
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L23-L23)

### particula.instruments.instrument.Instrument.__init__

*method*

```python
def __init__(profile: ProfileSource) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L0-L0)

### particula.instruments.instrument.Instrument.__post_init__

*method*

```python
def __post_init__() -> None
```

Resolve the supplied profile before reading any sources.

[View source](https://github.com/uncscode/particula/blob/main/particula/instruments/instrument.py#L25-L27)

### particula.instruments.instrument.Instrument.read

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
