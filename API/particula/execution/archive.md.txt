# particula.execution.archive

*module*

Versioned JSON/numeric archive transport with atomic file replacement.

The format uses standard ZIP, JSON and NPY primitives; it never loads pickle,
imports a named class, or executes metadata. Scientific schemas live above this
transport. Payload hashes detect corruption, not authenticity. All payloads are
detached before writing. Loading enforces a declared uncompressed byte budget.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L1-L1)

## particula.execution.archive._FORMAT

*attribute* · *module attribute*

```python
_FORMAT = 'particula-array-archive'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L20-L20)

## particula.execution.archive._decode_array

*function*

```python
def _decode_array(data, descriptor)
```

Validate NPY dimensions against actual bytes before allocating arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L139-L157)

## particula.execution.archive._json_bytes

*function*

```python
def _json_bytes(value)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L23-L26)

## particula.execution.archive._json_load

*function*

```python
def _json_load(data)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L38-L44)

## particula.execution.archive._numeric

*function*

```python
def _numeric(array)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L47-L53)

## particula.execution.archive._read_payloads

*function*

```python
def _read_payloads(archive, manifest)
```

Validate every named payload without extracting files.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L198-L219)

## particula.execution.archive._unique_object

*function*

```python
def _unique_object(pairs)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L29-L35)

## particula.execution.archive.read_archive

*function*

```python
def read_archive(path, *, kind, max_bytes=128 * 1024 * 1024)
```

Read a matching archive kind within a hard uncompressed byte budget.

Returns metadata, detached numeric arrays, immutable binary payloads. No
filesystem extraction is performed. Unknown versions/kinds fail explicitly.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L160-L195)

## particula.execution.archive.write_archive

*function*

```python
def write_archive(path, *, kind, metadata, arrays, blobs=None, compressed=True)
```

Atomically write detached arrays/bytes and JSON-compatible metadata.

A failure before os.replace leaves an existing destination unchanged.
Parent directories must already exist. Compression occurs on host outside
simulation execution/capture. blobs stores immutable checkpoint bytes.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/archive.py#L56-L136)
