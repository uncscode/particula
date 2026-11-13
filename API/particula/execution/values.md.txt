# particula.execution.values

*module*

Dependency-neutral values shared by execution package boundaries.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L1-L1)

## particula.execution.values.CPUStateAuthority

*class*

```python
class CPUStateAuthority(str, Enum)
```

Bases: `str`, `Enum`

Declare the asserted authority of CPU state for a fallback request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L23-L29)

### particula.execution.values.CPUStateAuthority.CPU_AUTHORITATIVE

*attribute* · *class attribute* · *instance attribute*

```python
CPU_AUTHORITATIVE = 'cpu_authoritative'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L26-L26)

### particula.execution.values.CPUStateAuthority.MUTATED

*attribute* · *class attribute* · *instance attribute*

```python
MUTATED = 'mutated'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L29-L29)

### particula.execution.values.CPUStateAuthority.RESIDENT

*attribute* · *class attribute* · *instance attribute*

```python
RESIDENT = 'resident'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L27-L27)

### particula.execution.values.CPUStateAuthority.UPLOADED

*attribute* · *class attribute* · *instance attribute*

```python
UPLOADED = 'uploaded'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L28-L28)

## particula.execution.values.FallbackBoundary

*class*

```python
class FallbackBoundary(str, Enum)
```

Bases: `str`, `Enum`

Declare the caller-visible boundary for an explicit fallback request.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L16-L20)

### particula.execution.values.FallbackBoundary.PRE_UPLOAD

*attribute* · *class attribute* · *instance attribute*

```python
PRE_UPLOAD = 'pre_upload'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L19-L19)

### particula.execution.values.FallbackBoundary.RESTORED

*attribute* · *class attribute* · *instance attribute*

```python
RESTORED = 'restored'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L20-L20)

## particula.execution.values.FallbackPolicy

*class*

```python
class FallbackPolicy(str, Enum)
```

Bases: `str`, `Enum`

Declare whether an eligible capability error is re-raised.

Or dispatched to CPU fallback.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L6-L13)

### particula.execution.values.FallbackPolicy.CPU

*attribute* · *class attribute* · *instance attribute*

```python
CPU = 'cpu'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L13-L13)

### particula.execution.values.FallbackPolicy.RAISE

*attribute* · *class attribute* · *instance attribute*

```python
RAISE = 'raise'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/values.py#L12-L12)
