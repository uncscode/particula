# particula.data._binding

*module*

Bind source selectors, measurement units, coordinates and clocks once.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L1-L1)

## particula.data._binding._EPOCH

*attribute* · *module attribute*

```python
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L16-L16)

## particula.data._binding._TICKS

*attribute* · *module attribute*

```python
_TICKS = {'s': 10 ** 9, 'ms': 10 ** 6, 'us': 1000, 'ns': 1}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L15-L15)

## particula.data._binding.Binding

*class*

```python
class Binding
```

Unit-checked mapping from one input layout to a typed scan frame.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L157-L306)

### particula.data._binding.Binding.clock

*attribute* · *instance attribute*

```python
clock = clock
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L170-L170)

### particula.data._binding.Binding.clock_columns

*attribute* · *instance attribute*

```python
clock_columns = [index(x, '/time') for x in selectors]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L181-L181)

### particula.data._binding.Binding.fields

*attribute* · *instance attribute*

```python
fields = []
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L189-L189)

### particula.data._binding.Binding.flag

*attribute* · *instance attribute*

```python
flag = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L221-L221)

### particula.data._binding.Binding.grid_id

*attribute* · *instance attribute*

```python
grid_id = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L191-L191)

### particula.data._binding.Binding.grids

*attribute* · *instance attribute*

```python
grids = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L190-L190)

### particula.data._binding.Binding.preamble

*attribute* · *instance attribute*

```python
preamble = preamble
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L192-L192)

### particula.data._binding.Binding.profile

*attribute* · *instance attribute*

```python
profile = profile
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L161-L161)

### particula.data._binding.Binding.schema

*attribute* · *instance attribute*

```python
schema = {'scan_id': pl.UInt64, 'source_line_start': pl.UInt64, 'source_line_end': pl.UInt64, 'repaired': pl.Boolean, 'time_utc': pl.Datetime('ns', 'UTC')}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L182-L188)

### particula.data._binding.Binding.shift_ns

*attribute* · *instance attribute*

```python
shift_ns = int(shift)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L178-L178)

### particula.data._binding.Binding.timing

*attribute* · *instance attribute*

```python
timing = {}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L231-L231)

### particula.data._binding.Binding.width

*attribute* · *instance attribute*

```python
width = profile['input'].get('record_width', len(header) if header else None)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L162-L164)

### particula.data._binding.Binding.zone

*attribute* · *instance attribute*

```python
zone = ZoneInfo(clock['timezone'])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L179-L179)

### particula.data._binding.Binding.__init__

*method*

```python
def __init__(profile, header, preamble)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L160-L240)

### particula.data._binding.Binding.decode

*method*

```python
def decode(row: list[str], start: int, end: int, repaired: bool = False) -> dict
```

Decode a record using its source line as its local scan ID.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L280-L306)

### particula.data._binding.Binding.timestamp

*method*

```python
def timestamp(row: list[str]) -> int
```

Parse epoch ticks or an unambiguous local wall time to UTC ns.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L242-L278)

## particula.data._binding._Field

*class*

```python
class _Field
```

Decode selected columns as a scalar or an array on the declared axes.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L47-L80)

### particula.data._binding._Field.dtype

*attribute* · *instance attribute*

```python
dtype = pl.Array(dtype, shape) if shape else dtype
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L61-L61)

### particula.data._binding._Field.indices

*attribute* · *instance attribute*

```python
indices = indices
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L52-L52)

### particula.data._binding._Field.maximum

*attribute* · *instance attribute*

```python
maximum = np.finfo(np.float32).max.item() if dtype == pl.Float32 else math.inf
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L58-L60)

### particula.data._binding._Field.missing

*attribute* · *instance attribute*

```python
missing = missing
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L53-L53)

### particula.data._binding._Field.name

*attribute* · *instance attribute*

```python
name = spec['name']
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L51-L51)

### particula.data._binding._Field.shape

*attribute* · *instance attribute*

```python
shape = shape
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L62-L62)

### particula.data._binding._Field.__init__

*method*

```python
def __init__(spec, indices, missing, shape=())
```

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L50-L62)

### particula.data._binding._Field.decode

*method*

```python
def decode(row: list[str])
```

Convert values to output units and retain missing tokens as nulls.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L64-L80)

## particula.data._binding._affine

*function* · *cached*

```python
def _affine(source: str, target: str) -> tuple[float, float]
```

Return scale and offset for converting source units to output units.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L36-L44)

## particula.data._binding._index

*function*

```python
def _index(selector, header, width, path)
```

Resolve a numeric offset or unique header label within the layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L19-L33)

## particula.data._binding._sizer_grid

*function*

```python
def _sizer_grid(profile, header, preamble, index)
```

Read diameter midpoints and bin width from the source layout.

[View source](https://github.com/uncscode/particula/blob/main/particula/data/_binding.py#L83-L154)
