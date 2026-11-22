# particula.gpu.kernels.parcel_motion

*module*

Compute straight-line particle trajectories and face arrival times.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_motion.py#L1-L1)

## particula.gpu.kernels.parcel_motion.F

*attribute* · *module attribute*

```python
F = wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_motion.py#L7-L7)

## particula.gpu.kernels.parcel_motion.arrivals

*function*

```python
def arrivals(position: Matrix, velocity: Matrix, lower: Matrix, upper: Matrix, until: F, times: Matrix)
```

Compute time to the first outward face on each axis, in seconds.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_motion.py#L10-L27)

## particula.gpu.kernels.parcel_motion.displace

*function*

```python
def displace(position: Matrix, velocity: Matrix, dt: F, output: Matrix)
```

Advance all positions by the accepted duration.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/parcel_motion.py#L30-L34)
