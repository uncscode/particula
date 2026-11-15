# particula.execution.field_buffers

*module*

Explicit upload/download of fixed-schema registered field storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L1-L1)

## particula.execution.field_buffers.FieldBuffers

*class* · *dataclass*

```python
class FieldBuffers
```

Own descriptors and device arrays without CPU simulation state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L29-L65)

### particula.execution.field_buffers.FieldBuffers.arrays

*attribute* · *instance attribute*

```python
arrays: dict[str, _DeviceArray]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L35-L35)

### particula.execution.field_buffers.FieldBuffers.dimensions

*attribute* · *instance attribute*

```python
dimensions: tuple[tuple[str, int], ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L34-L34)

### particula.execution.field_buffers.FieldBuffers.ledgers

*attribute* · *instance attribute*

```python
ledgers: dict[str, _DeviceArray]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L36-L36)

### particula.execution.field_buffers.FieldBuffers.specs

*attribute* · *instance attribute*

```python
specs: tuple[FieldSpec, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L33-L33)

### particula.execution.field_buffers.FieldBuffers.__init__

*method*

```python
def __init__(specs: tuple[FieldSpec, ...], dimensions: tuple[tuple[str, int], ...], arrays: dict[str, _DeviceArray], ledgers: dict[str, _DeviceArray]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L0-L0)

### particula.execution.field_buffers.FieldBuffers.download

*method*

```python
def download() -> ExtensionFields
```

Download detached CPU values at an explicit transfer boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L58-L65)

### particula.execution.field_buffers.FieldBuffers.upload

*method* · *classmethod*

```python
def upload(fields: ExtensionFields, dimensions, device: str)
```

Copy each supported dtype once on the caller-selected stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L38-L56)

## particula.execution.field_buffers._DeviceArray

*class*

```python
class _DeviceArray(Protocol)
```

Bases: `Protocol`

The explicit observation surface required from retained device arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L12-L26)

### particula.execution.field_buffers._DeviceArray.shape

*property*

```python
shape: tuple[int, ...]
```

Expose immutable dimensions without reading device values.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L16-L18)

### particula.execution.field_buffers._DeviceArray.numpy

*method*

```python
def numpy() -> NDArray[np.generic]
```

Download a host array at an explicit observation boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L24-L26)

### particula.execution.field_buffers._DeviceArray.reshape

*method*

```python
def reshape(shape: tuple[int, ...]) -> _DeviceArray
```

Make a setup-only view sharing the same device allocation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/field_buffers.py#L20-L22)
