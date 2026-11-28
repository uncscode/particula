# particula.execution._ccn_device

*module*

Resident equilibrium CCN sampling, using the prepared owner's stream.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L1-L1)

## particula.execution._ccn_device._TIE

*attribute* · *module attribute*

```python
_TIE = wp.constant(wp.float64(1.0 - 32 * np.finfo(float).eps))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L14-L14)

## particula.execution._ccn_device._VOLUME_FACTOR

*attribute* · *module attribute*

```python
_VOLUME_FACTOR = wp.constant(wp.float64(6.0 / np.pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L13-L13)

## particula.execution._ccn_device._ZERO

*attribute* · *module attribute*

```python
_ZERO = wp.constant(wp.float64(0.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L12-L12)

## particula.execution._ccn_device.DeviceCCNBatch

*class*

```python
class DeviceCCNBatch
```

Prepare constant inputs and reusable compact output buffers.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L142-L267)

### particula.execution._ccn_device.DeviceCCNBatch.common

*attribute* · *instance attribute*

```python
common = upload(np.zeros(boxes) if self.mixed else common)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L162-L162)

### particula.execution._ccn_device.DeviceCCNBatch.counts

*attribute* · *instance attribute*

```python
counts = wp.empty((capacity, *self.groups), dtype=wp.float64, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L171-L173)

### particula.execution._ccn_device.DeviceCCNBatch.diameters

*attribute* · *instance attribute*

```python
diameters = wp.empty((capacity if not self.mixed else 1, *self.groups), dtype=wp.float64, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L174-L178)

### particula.execution._ccn_device.DeviceCCNBatch.dynamic

*attribute* · *instance attribute*

```python
dynamic = length is None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L168-L168)

### particula.execution._ccn_device.DeviceCCNBatch.groups

*attribute* · *instance attribute*

```python
groups = (boxes, len(config.saturation_ratios))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L153-L153)

### particula.execution._ccn_device.DeviceCCNBatch.length

*attribute* · *instance attribute*

```python
length = upload(np.ones(boxes) if length is None else length)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L164-L164)

### particula.execution._ccn_device.DeviceCCNBatch.mixed

*attribute* · *instance attribute*

```python
mixed = common is None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L161-L161)

### particula.execution._ccn_device.DeviceCCNBatch.owner

*attribute* · *instance attribute*

```python
owner = owner
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L146-L146)

### particula.execution._ccn_device.DeviceCCNBatch.product

*attribute* · *instance attribute*

```python
product = wp.float64(config.kelvin_temperature_product if length is None else 0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L165-L167)

### particula.execution._ccn_device.DeviceCCNBatch.ratios

*attribute* · *instance attribute*

```python
ratios = upload(config.saturation_ratios)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L160-L160)

### particula.execution._ccn_device.DeviceCCNBatch.shape

*attribute* · *instance attribute*

```python
shape = (boxes, slots, self.groups[1])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L154-L154)

### particula.execution._ccn_device.DeviceCCNBatch.species

*attribute* · *instance attribute*

```python
species = upload(config.material_kappa or (0.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L163-L163)

### particula.execution._ccn_device.DeviceCCNBatch.status

*attribute* · *instance attribute*

```python
status = wp.zeros((capacity, boxes), dtype=int, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L179-L179)

### particula.execution._ccn_device.DeviceCCNBatch.thresholds

*attribute* · *instance attribute*

```python
thresholds = wp.empty(self.groups, dtype=wp.float64, device=device)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L170-L170)

### particula.execution._ccn_device.DeviceCCNBatch.water

*attribute* · *instance attribute*

```python
water = -1 if config.water_index is None else config.water_index
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L169-L169)

### particula.execution._ccn_device.DeviceCCNBatch.__init__

*method*

```python
def __init__(owner, config, capacity)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L145-L181)

### particula.execution._ccn_device.DeviceCCNBatch._launch

*method*

```python
def _launch(kernel, shape, args)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L183-L190)

### particula.execution._ccn_device.DeviceCCNBatch.download

*method*

```python
def download(count)
```

Return validated compact rows on explicit flush.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L249-L267)

### particula.execution._ccn_device.DeviceCCNBatch.enqueue

*method*

```python
def enqueue(row)
```

Sample the current accepted device state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L207-L247)

### particula.execution._ccn_device.DeviceCCNBatch.refresh

*method*

```python
def refresh()
```

Compute common thresholds once per setting and box.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L192-L205)

## particula.execution._ccn_device._begin

*function*

```python
def _begin(row: int, thresholds: wp.array2d(dtype=wp.float64), common: int, counts: wp.array3d(dtype=wp.float64), diameters: wp.array3d(dtype=wp.float64), status: wp.array2d(dtype=int))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L35-L50)

## particula.execution._ccn_device._count

*function*

```python
def _count(p: WarpParticleData, temperature: wp.array(dtype=wp.float64), ratios: wp.array(dtype=wp.float64), species: wp.array(dtype=wp.float64), length: wp.array(dtype=wp.float64), product: wp.float64, water: int, mixed: int, thresholds: wp.array2d(dtype=wp.float64), row: int, counts: wp.array3d(dtype=wp.float64), status: wp.array2d(dtype=int))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L83-L129)

## particula.execution._ccn_device._dry_volume

*function*

```python
def _dry_volume(p: WarpParticleData, box: int, slot: int, water: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L53-L62)

## particula.execution._ccn_device._normalize

*function*

```python
def _normalize(row: int, volume: wp.array(dtype=wp.float64), counts: wp.array3d(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L132-L139)

## particula.execution._ccn_device._thresholds

*function*

```python
def _thresholds(common: wp.array(dtype=wp.float64), ratios: wp.array(dtype=wp.float64), length: wp.array(dtype=wp.float64), temperature: wp.array(dtype=wp.float64), product: wp.float64, scales: wp.array2d(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L17-L32)

## particula.execution._ccn_device._volume_weighted_kappa

*function*

```python
def _volume_weighted_kappa(p: WarpParticleData, box: int, slot: int, water: int, dry: wp.float64, species: wp.array(dtype=wp.float64))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_ccn_device.py#L65-L80)
