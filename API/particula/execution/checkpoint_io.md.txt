# particula.execution.checkpoint_io

*module*

Safe disk transport for existing exact-device prepared checkpoints.

Only a fixed allowlist of data records/enums is decoded. No class name from a
file is imported and no pickle is accepted. Canonical bytes, RNG continuation,
process plans and auxiliary fields retain the existing restart authority; CPU
inspection arrays never replace canonical device payloads. Resuming still
requires the caller's compatible processes and explicit execution registry.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L1-L1)

## particula.execution.checkpoint_io._Encoder

*class*

```python
class _Encoder
```

Encode only known records and immutable/numeric primitive payloads.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L87-L129)

### particula.execution.checkpoint_io._Encoder.arrays

*attribute* · *instance attribute*

```python
arrays = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L92-L92)

### particula.execution.checkpoint_io._Encoder.blobs

*attribute* · *instance attribute*

```python
blobs = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L93-L93)

### particula.execution.checkpoint_io._Encoder.__init__

*method*

```python
def __init__()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L90-L93)

### particula.execution.checkpoint_io._Encoder.encode

*method*

```python
def encode(value)
```

Produce a JSON data tree and detached payload tables.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L95-L129)

## particula.execution.checkpoint_io._decode

*function*

```python
def _decode(value, arrays, blobs, records, enums, depth=0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L132-L163)

## particula.execution.checkpoint_io._record_class

*function*

```python
def _record_class(value, records)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L261-L265)

## particula.execution.checkpoint_io._types

*function*

```python
def _types()
```

List permitted data constructors explicitly, without dynamic imports.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L20-L84)

## particula.execution.checkpoint_io._validate

*function*

```python
def _validate(checkpoint)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L166-L173)

## particula.execution.checkpoint_io.load_checkpoint

*function*

```python
def load_checkpoint(path, *, max_bytes=512 * 1024 * 1024)
```

Load exact checkpoint bytes without launching kernels or restoring state.

Pass the result to the original compatible runnable.resume(). Never load an
analysis file as restart, silently migrate schema, or recreate RNG words.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L223-L258)

## particula.execution.checkpoint_io.save_checkpoint

*function*

```python
def save_checkpoint(checkpoint, path, *, compressed=True)
```

Atomically store an already synchronized prepared checkpoint on host.

Call owner.checkpoint() explicitly first; this function performs no device
transfers and cannot run in CUDA capture. Unregistered data record types
fail before touching the destination. Installed package/NumPy/Warp versions
must match at load; restart also enforces device and process compatibility.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/checkpoint_io.py#L176-L220)
