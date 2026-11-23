# particula.execution._multivapor_parcel_kernels

*module*

Float64 multivapor parcel kernels and resident trial controller.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L1-L1)

## particula.execution._multivapor_parcel_kernels.Cube

*attribute* · *module attribute*

```python
Cube = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L21-L21)

## particula.execution._multivapor_parcel_kernels.IntMatrix

*attribute* · *module attribute*

```python
IntMatrix = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L23-L23)

## particula.execution._multivapor_parcel_kernels.Ints

*attribute* · *module attribute*

```python
Ints = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L22-L22)

## particula.execution._multivapor_parcel_kernels.Matrix

*attribute* · *module attribute*

```python
Matrix = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L20-L20)

## particula.execution._multivapor_parcel_kernels.ONE

*attribute* · *module attribute*

```python
ONE = wp.constant(wp.float64(1.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L35-L35)

## particula.execution._multivapor_parcel_kernels.PI4

*attribute* · *module attribute*

```python
PI4 = wp.constant(wp.float64(12.566370614359172))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L37-L37)

## particula.execution._multivapor_parcel_kernels.Vector

*attribute* · *module attribute*

```python
Vector = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L19-L19)

## particula.execution._multivapor_parcel_kernels.ZERO

*attribute* · *module attribute*

```python
ZERO = wp.constant(wp.float64(0.0))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L36-L36)

## particula.execution._multivapor_parcel_kernels.accumulate

*function*

```python
def accumulate(volume: Vector, dp: Vector, flux: Matrix, counts: Matrix, props: Matrix, scale: Matrix, air: wp.float64, p0: wp.float64, t0: wp.float64, cp: wp.float64, result: Vector, counters: Ints, invalid: Ints)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L190-L215)

## particula.execution._multivapor_parcel_kernels.decide

*function*

```python
def decide(y: Vector, ytrial: Vector, error: Vector, clock: Vector, invalid: Ints, counters: Ints, condition: Ints, rtol: wp.float64, minimum: wp.float64, limit: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L397-L439)

## particula.execution._multivapor_parcel_kernels.factor

*function*

```python
def factor(jac: Matrix, pivots: Ints, invalid: Ints)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L324-L346)

## particula.execution._multivapor_parcel_kernels.flux_guard

*function*

```python
def flux_guard(pressure: Vector, temperature: Vector, vapor: Vector, volume: Vector, masses: Cube, density: Vector, lanes: Ints, props: Matrix, free: Matrix, slopes: Matrix, flux: Matrix, errors: Cube, uncertain: IntMatrix)
```

Bound heat and mass flux errors about the trial surface solution.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L442-L516)

## particula.execution._multivapor_parcel_kernels.fluxes

*function*

```python
def fluxes(pressure: Vector, temperature: Vector, vapor: Vector, volume: Vector, masses: Cube, counts: Matrix, density: Vector, lanes: Ints, props: Matrix, sigma: wp.float64, activities: Cube, free: Matrix, slopes: Matrix, flux: Matrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L132-L187)

## particula.execution._multivapor_parcel_kernels.hydro

*function*

```python
def hydro(pressure: Vector, temperature: Vector, vapor: Vector, props: Matrix, rd: wp.float64, loading_gw: wp.float64, volume: Vector, dp: Vector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L114-L129)

## particula.execution._multivapor_parcel_kernels.linearize

*function*

```python
def linearize(base: Vector, f: Vector, delta: Vector, clock: Vector, jac: Matrix, column: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L306-L321)

## particula.execution._multivapor_parcel_kernels.mark_base

*function*

```python
def mark_base(invalid: Ints, counters: Ints)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L218-L221)

## particula.execution._multivapor_parcel_kernels.perturb

*function*

```python
def perturb(y: Vector, ytrial: Vector, column: int, delta: Vector, q0: Vector, m0: Matrix, lanes: Ints, scale: Matrix, counts: Matrix, air: wp.float64)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L264-L303)

## particula.execution._multivapor_parcel_kernels.rk_error

*function*

```python
def rk_error(clock: Vector, k1: Vector, k2: Vector, k3: Vector, k4: Vector, error: Vector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L247-L261)

## particula.execution._multivapor_parcel_kernels.rk_stage

*function*

```python
def rk_stage(y: Vector, clock: Vector, k1: Vector, k2: Vector, k3: Vector, stage: int, output: Vector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L224-L244)

## particula.execution._multivapor_parcel_kernels.ros_finish

*function*

```python
def ros_finish(y: Vector, k1: Vector, k2: Vector, k3: Vector, ytrial: Vector, error: Vector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L388-L394)

## particula.execution._multivapor_parcel_kernels.ros_stage

*function*

```python
def ros_stage(y: Vector, k1: Vector, ytrial: Vector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L382-L385)

## particula.execution._multivapor_parcel_kernels.solve

*function*

```python
def solve(jac: Matrix, pivots: Ints, clock: Vector, rhs: Vector, k1: Vector, k2: Vector, stage: int, out: Vector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L349-L379)

## particula.execution._multivapor_parcel_kernels.start

*function*

```python
def start(clock: Vector, counters: Ints, stop: wp.float64, condition: Ints)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L40-L44)

## particula.execution._multivapor_parcel_kernels.trial

*function*

```python
def trial(clock: Vector, counters: Ints, invalid: Ints, detail: Ints)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L47-L52)

## particula.execution._multivapor_parcel_kernels.unpack

*function*

```python
def unpack(y: Vector, m0: Matrix, q0: Vector, scale: Matrix, lanes: Ints, counts: Matrix, p0: wp.float64, t0: wp.float64, air: wp.float64, pressure: Vector, temperature: Vector, vapor: Vector, masses: Cube, invalid: Ints, detail: Ints, stage: int)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_multivapor_parcel_kernels.py#L55-L111)
