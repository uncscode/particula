# particula.coupling.grid_warp

*module*

Float64 Warp operations for prescribed field sampling and transport.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L1-L1)

## particula.coupling.grid_warp.F

*attribute* · *module attribute*

```python
F = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L20-L20)

## particula.coupling.grid_warp.Index3

*attribute* · *module attribute*

```python
Index3 = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L18-L18)

## particula.coupling.grid_warp.IntMatrix

*attribute* · *module attribute*

```python
IntMatrix = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L16-L16)

## particula.coupling.grid_warp.Keys

*attribute* · *module attribute*

```python
Keys = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L17-L17)

## particula.coupling.grid_warp.Position3

*attribute* · *module attribute*

```python
Position3 = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L19-L19)

## particula.coupling.grid_warp._MIX_FIRST

*attribute* · *module attribute*

```python
_MIX_FIRST = wp.constant(wp.uint64(13787848793156543929))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L142-L142)

## particula.coupling.grid_warp._MIX_SECOND

*attribute* · *module attribute*

```python
_MIX_SECOND = wp.constant(wp.uint64(10723151780598845931))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L143-L143)

## particula.coupling.grid_warp._TWO_PI

*attribute* · *module attribute*

```python
_TWO_PI = wp.constant(wp.float64(2.0 * np.pi))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L144-L144)

## particula.coupling.grid_warp._array

*function*

```python
def _array(values, device, dtype=F)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L29-L30)

## particula.coupling.grid_warp._interp

*function*

```python
def _interp(table: FloatMatrix, z: F, col: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L46-L61)

## particula.coupling.grid_warp._mix64

*function*

```python
def _mix64(x: wp.uint64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L147-L152)

## particula.coupling.grid_warp._normal_draw

*function*

```python
def _normal_draw(key: wp.uint64, interval: int, step: int, axis: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L167-L171)

## particula.coupling.grid_warp._profile

*function*

```python
def _profile(table: FloatMatrix, query: FloatMatrix, out: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L64-L76)

## particula.coupling.grid_warp._uniform

*function*

```python
def _uniform(row: FloatVector, out: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L33-L36)

## particula.coupling.grid_warp._uniform_draw

*function*

```python
def _uniform_draw(key: wp.uint64, interval: int, step: int, lane: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L155-L164)

## particula.coupling.grid_warp.accumulate

*function*

```python
def accumulate(ledger: FloatMatrix, increment: FloatMatrix)
```

Accumulate host-reported boundary totals on device.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L359-L363)

## particula.coupling.grid_warp.diffuse_cells

*function*

```python
def diffuse_cells(interval: int, step: int, end: F, keys: Keys, fraction: FloatMatrix, neighbors: IntMatrix, shape: Index3, lo: Position3, width: Position3, jac_edge: FloatVector, bc: IntVector, pos: FloatMatrix, cells: IntVector, status: IntVector, event_time: FloatVector)
```

Sample conservative face exchanges on variable-K or mapped cells.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L272-L338)

## particula.coupling.grid_warp.field_totals

*function*

```python
def field_totals(state: FloatMatrix, total: FloatVector)
```

Reduce extensive fields at an observation boundary.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L341-L345)

## particula.coupling.grid_warp.move_particles

*function*

```python
def move_particles(interval: int, step: int, keys: Keys, diffusivity: Position3, start: F, dt: F, shape: Index3, lo: Position3, width: Position3, bc: IntVector, jac: FloatVector, speed: FloatMatrix, slip: FloatVector, release: IntVector, pos: FloatMatrix, cell: IntVector, status: IntVector, event_time: FloatVector)
```

Advance each stable record through mapped cell faces.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L174-L269)

## particula.coupling.grid_warp.particle_totals

*function*

```python
def particle_totals(state: IntVector, inventory: FloatMatrix, total: FloatMatrix)
```

Reduce particle payloads by transport fate.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L348-L356)

## particula.coupling.grid_warp.profile_sample

*function*

```python
def profile_sample(s, z, eta, jac, gradient, device)
```

Interpolate wind and thermodynamics using the explicit device.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L79-L92)

## particula.coupling.grid_warp.transport

*function*

```python
def transport(nb: IntMatrix, incoming: FloatMatrix, outgoing: FloatMatrix, volume: FloatVector, inlet: FloatTensor, old: FloatMatrix, updated: FloatMatrix, imports: FloatMatrix, exports: FloatMatrix)
```

Gather one conservative substep per cell and inventory lane.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L95-L122)

## particula.coupling.grid_warp.uniform_sample

*function*

```python
def uniform_sample(n, row, device)
```

Sample uniform forcing on the selected device.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L39-L43)

## particula.coupling.grid_warp.uptake

*function*

```python
def uptake(state: FloatMatrix, gas: int, number: int, material: int, fraction: F)
```

Transfer gas once into an occupied particle bin.

[View source](https://github.com/uncscode/particula/blob/main/particula/coupling/grid_warp.py#L125-L138)
