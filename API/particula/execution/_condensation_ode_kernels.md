# particula.execution._condensation_ode_kernels

*module*

Device state algebra for native deterministic condensation RK3(2).

The existing mapped condensation kernel supplies the physical mass rates.
These kernels only unpack scaled coordinates, couple conservative reservoirs,
check physical domains and evaluate the embedded max-component error estimate.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L1-L1)

## particula.execution._condensation_ode_kernels.FloatMatrix

*attribute* · *module attribute*

```python
FloatMatrix = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L14-L14)

## particula.execution._condensation_ode_kernels.FloatTensor

*attribute* · *module attribute*

```python
FloatTensor = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L15-L15)

## particula.execution._condensation_ode_kernels.FloatVector

*attribute* · *module attribute*

```python
FloatVector = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L13-L13)

## particula.execution._condensation_ode_kernels.IntVector

*attribute* · *module attribute*

```python
IntVector = Any
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L16-L16)

## particula.execution._condensation_ode_kernels.check_slots

*function*

```python
def check_slots(masses: FloatTensor, counts: FloatMatrix, invalid: IntVector, resolved: wp.int32)
```

Reject active evaporation; only resolved free slots require zero mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L48-L64)

## particula.execution._condensation_ode_kernels.error_norm

*function*

```python
def error_norm(old: FloatVector, candidate: FloatVector, k1: FloatVector, k2: FloatVector, k3: FloatVector, k4: FloatVector, atol: FloatVector, rtol: FloatVector, h: wp.float64, norm: FloatVector)
```

Reduce the shared Bogacki-Shampine local estimate across every lane.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L128-L155)

## particula.execution._condensation_ode_kernels.pack_rate

*function*

```python
def pack_rate(rate: FloatTensor, counts: FloatMatrix, volume: FloatVector, gas_to_particle: IntVector, scale: FloatVector, derivative: FloatVector, invalid: IntVector, resolved: wp.int32)
```

Couple gas to particle flux with native population weights.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L67-L101)

## particula.execution._condensation_ode_kernels.stage

*function*

```python
def stage(state: FloatVector, k1: FloatVector, k2: FloatVector, k3: FloatVector, h: wp.float64, a: wp.float64, b: wp.float64, c: wp.float64, result: FloatVector)
```

Form one out-of-place stage without touching accepted state.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L104-L125)

## particula.execution._condensation_ode_kernels.unpack

*function*

```python
def unpack(state: FloatVector, scale: FloatVector, masses: FloatTensor, gas: FloatMatrix, invalid: IntVector)
```

Unpack a trial and flag nonfinite/negative physical inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ode_kernels.py#L24-L45)
