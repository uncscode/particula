# particula.gpu.kernels.parcel_replay

*module*

Batched float64 replay on a fixed schedule using extensive inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L1-L1)

## particula.gpu.kernels.parcel_replay.FIRST_BIN

*attribute* · *module attribute*

```python
FIRST_BIN = 4
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L22-L22)

## particula.gpu.kernels.parcel_replay.Indices

*attribute* · *module attribute*

```python
Indices = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L15-L15)

## particula.gpu.kernels.parcel_replay.WATER_MASS_FACTOR

*attribute* · *module attribute*

```python
WATER_MASS_FACTOR = wp.constant(F(WATER_FACTOR))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L12-L12)

## particula.gpu.kernels.parcel_replay.exchange

*function*

```python
def exchange(previous: Matrix, volume: Vector, edges: Indices, flows: Vector, enabled: Vector, vapor: Vector, dt: F, flux: Matrix)
```

Compute each transfer from the same unmodified donor inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L25-L50)

## particula.gpu.kernels.parcel_replay.gather

*function*

```python
def gather(previous: Matrix, edges: Indices, flux: Matrix, result: Matrix)
```

Apply signed incoming and outgoing transfers to each lane.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L53-L63)

## particula.gpu.kernels.parcel_replay.local

*function*

```python
def local(previous: Matrix, forcing: Matrix, dry: Matrix, kappa: Vector, uptake: Vector, counts: Vector, dt: F, amplitude: Vector, result: Matrix)
```

Advance water growth, organic uptake and source ledgers in each cell.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L77-L133)

## particula.gpu.kernels.parcel_replay.observe

*function*

```python
def observe(state: Matrix, cell: int, lane: int, result: Vector)
```

Reduce the selected terminal inventory to a scalar Warp loss.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L136-L145)

## particula.gpu.kernels.parcel_replay.population_count

*function*

```python
def population_count(previous: Matrix, count: Vector)
```

Save total represented count for uptake and its reverse pass.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_replay.py#L66-L74)
