# particula.data.storage

*module*

Collect reader batches in memory or publish a complete Parquet dataset.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/storage.py#L1-L1)

## particula.data.storage._sync

*function*

```python
def _sync(path: Path) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/storage.py#L97-L104)

## particula.data.storage.import_to_parquet

*function*

```python
def import_to_parquet(source: str | os.PathLike[str], destination: str | os.PathLike[str], *, profile: ProfileSource, **limits: int) -> Measurement
```

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

## particula.data.storage.open_measurement

*function*

```python
def open_measurement(destination: str | os.PathLike[str]) -> Measurement
```

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

## particula.data.storage.read_measurement

*function*

```python
def read_measurement(source: str | os.PathLike[str], *, profile: ProfileSource, **limits: int) -> Measurement
```

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
