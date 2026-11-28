# particula.data.reader

*module*

Stream one delimited measurement export, preserving physical source spans.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L1-L1)

## particula.data.reader.ISSUE_SCHEMA

*attribute* · *module attribute*

```python
ISSUE_SCHEMA = {'code': pl.String, 'source_line_start': pl.UInt64, 'source_line_end': pl.UInt64, 'message': pl.String}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L28-L33)

## particula.data.reader._COMPLETE

*attribute* · *module attribute*

```python
_COMPLETE = re.compile('[0-9]{10}(?:,[0-9]+(?:\\.[0-9]+)?){4}')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L40-L40)

## particula.data.reader._CSV_LIMIT_LOCK

*attribute* · *module attribute*

```python
_CSV_LIMIT_LOCK = Lock()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L25-L25)

## particula.data.reader._FRAGMENT

*attribute* · *module attribute*

```python
_FRAGMENT = re.compile('[0-9]{10},(?:[0-9]+(?:\\.[0-9]+)?,){0,2}')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L39-L39)

## particula.data.reader._LINE_END

*attribute* · *module attribute*

```python
_LINE_END = re.compile(b'[\\r\\n]')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L26-L26)

## particula.data.reader._PLAIN

*attribute* · *module attribute*

```python
_PLAIN = re.compile(b'[12][0-9]{9},[0-9]{1,9},[0-9]{1,3}\\.[0-9]{1,4},[0-9]{1,3},[0-9]{1,3}\\.[0-9]{1,6}\\r?\\n')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L36-L38)

## particula.data.reader.MeasurementReader

*class*

```python
class MeasurementReader
```

Read one source in batches while tracking layouts and diagnostics.

**Attributes**

- `manifest` (`dict | None`) — Import provenance, including the source digest, grids and scan count. Available only after batches() reaches end of file.
- `segment` — Zero-based source layout index for the current batch.
- `issue_counts` (`Counter[str]`) — Number of diagnostics emitted per code during decoding.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L335-L528)

### particula.data.reader.MeasurementReader.count

*attribute* · *instance attribute*

```python
count = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L378-L378)

### particula.data.reader.MeasurementReader.issue_counts

*attribute* · *instance attribute*

```python
issue_counts: Counter[str] = Counter()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L377-L377)

### particula.data.reader.MeasurementReader.manifest

*attribute* · *instance attribute*

```python
manifest: dict | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L379-L379)

### particula.data.reader.MeasurementReader.profile

*attribute* · *instance attribute*

```python
profile = load_profile(profile)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L372-L372)

### particula.data.reader.MeasurementReader.segment

*attribute* · *instance attribute*

```python
segment = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L380-L380)

### particula.data.reader.MeasurementReader.source

*attribute* · *instance attribute*

```python
source = Path(source)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L371-L371)

### particula.data.reader.MeasurementReader.__init__

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

### particula.data.reader.MeasurementReader._decoded

*method*

```python
def _decoded(records, binding)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L459-L508)

### particula.data.reader.MeasurementReader._issue

*method*

```python
def _issue(code, start, end, message)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L521-L528)

### particula.data.reader.MeasurementReader._missing

*method*

```python
def _missing(row, binding, start, end)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L510-L519)

### particula.data.reader.MeasurementReader.batches

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

## particula.data.reader._Lines

*class*

```python
class _Lines
```

Track inclusive physical line spans and bytes used by each CSV record.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L58-L123)

### particula.data.reader._Lines.block

*attribute* · *instance attribute*

```python
block = b''
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L64-L64)

### particula.data.reader._Lines.cursor

*attribute* · *instance attribute*

```python
cursor = 0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L65-L65)

### particula.data.reader._Lines.__init__

*method*

```python
def __init__(reader, encoding, maximum, first)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L61-L65)

### particula.data.reader._Lines.__iter__

*method*

```python
def __iter__()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L67-L68)

### particula.data.reader._Lines.__next__

*method*

```python
def __next__()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L97-L123)

### particula.data.reader._Lines._has_bytes

*method*

```python
def _has_bytes()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L70-L74)

### particula.data.reader._Lines._readline

*method*

```python
def _readline()
```

Frame CR, LF and CRLF lines while retaining their original bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L76-L95)

## particula.data.reader._Source

*class*

```python
class _Source(io.RawIOBase)
```

Bases: `io.RawIOBase`

Hash bytes received from the file and cap each underlying read.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L43-L55)

### particula.data.reader._Source.__init__

*method*

```python
def __init__(stream, chunk, hasher)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L46-L47)

### particula.data.reader._Source.readable

*method*

```python
def readable()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L49-L50)

### particula.data.reader._Source.readinto

*method*

```python
def readinto(buffer)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L52-L55)

## particula.data.reader._events

*function*

```python
def _events(records, profile)
```

Recognize only a comma-terminated prefix and its numeric continuation.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L303-L332)

## particula.data.reader._fast_counter

*function*

```python
def _fast_counter(block, start)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L249-L274)

## particula.data.reader._frames

*function*

```python
def _frames(rows, issues, binding)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L531-L539)

## particula.data.reader._layout

*function*

```python
def _layout(records, profile)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L157-L173)

## particula.data.reader._next_record

*function*

```python
def _next_record(rows, maximum)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L126-L137)

## particula.data.reader._plain_batches

*function*

```python
def _plain_batches(stream, hasher, size, row_limit)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L277-L300)

## particula.data.reader._records

*function*

```python
def _records(lines, delimiter)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L140-L154)

## particula.data.reader._sections

*function*

```python
def _sections(records, profile, binding, header)
```

Split concatenated AIM exports when the preamble or header recurs.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/reader.py#L176-L204)

## particula.data.reader.inspect_source

*function*

```python
def inspect_source(source: str | os.PathLike[str], profile: ProfileSource) -> dict
```

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
