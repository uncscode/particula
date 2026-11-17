# particula.execution._condensation_ros3_kernels

*module*

Resident single-exchange Rosenbrock algebra and conservative linear solves.

For one exchanging material per particle, the Jacobian has independent particle
blocks coupled through one gas reservoir per box. A Schur complement solves
this arrowhead system in linear storage. Other material rates are zero in this
profile; their stage increments are zero, so composition cross derivatives
into those fixed coordinates need not be stored. No physical rate is evaluated
here: the native mapped condensation kernel supplies every derivative.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L1-L1)

## particula.execution._condensation_ros3_kernels.C21

*attribute* · *module attribute*

```python
C21 = wp.constant(wp.float64(1.0156171083877703))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L184-L184)

## particula.execution._condensation_ros3_kernels.C31

*attribute* · *module attribute*

```python
C31 = wp.constant(wp.float64(4.07599564525377))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L185-L185)

## particula.execution._condensation_ros3_kernels.C32

*attribute* · *module attribute*

```python
C32 = wp.constant(wp.float64(9.20767942983308))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L186-L186)

## particula.execution._condensation_ros3_kernels.E2

*attribute* · *module attribute*

```python
E2 = wp.constant(wp.float64(2.907955871680547))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L189-L189)

## particula.execution._condensation_ros3_kernels.E3

*attribute* · *module attribute*

```python
E3 = wp.constant(wp.float64(0.2235406989781157))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L190-L190)

## particula.execution._condensation_ros3_kernels.GAMMA

*attribute* · *module attribute*

```python
GAMMA = wp.constant(wp.float64(0.435866521508459))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L181-L181)

## particula.execution._condensation_ros3_kernels.GAMMA2

*attribute* · *module attribute*

```python
GAMMA2 = wp.constant(wp.float64(0.24291996454816805))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L182-L182)

## particula.execution._condensation_ros3_kernels.GAMMA3

*attribute* · *module attribute*

```python
GAMMA3 = wp.constant(wp.float64(2.185138002766406))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L183-L183)

## particula.execution._condensation_ros3_kernels.M2

*attribute* · *module attribute*

```python
M2 = wp.constant(wp.float64(6.1697947043828245))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L187-L187)

## particula.execution._condensation_ros3_kernels.M3

*attribute* · *module attribute*

```python
M3 = wp.constant(wp.float64(0.42772256543218573))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L188-L188)

## particula.execution._condensation_ros3_kernels.extract_derivative

*function*

```python
def extract_derivative(plus: FloatVector, minus: FloatVector, span: FloatVector, counts: FloatMatrix, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, gas_columns: wp.int32, result: FloatMatrix, invalid: IntVector)
```

Extract local derivatives from colored physical-rate evaluations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L63-L94)

## particula.execution._condensation_ros3_kernels.finish

*function*

```python
def finish(state: FloatVector, first: FloatVector, second: FloatVector, third: FloatVector, atol: FloatVector, rtol: FloatVector, candidate: FloatVector, norm: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L225-L245)

## particula.execution._condensation_ros3_kernels.partial_time

*function*

```python
def partial_time(base: FloatVector, shifted: FloatVector, delta: wp.float64, result: FloatVector)
```

Differentiate forcing at fixed state inside one smooth time segment.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L248-L257)

## particula.execution._condensation_ros3_kernels.perturb

*function*

```python
def perturb(state: FloatVector, scale: FloatVector, counts: FloatMatrix, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, gas_columns: wp.int32, plus: FloatVector, minus: FloatVector, span: FloatVector)
```

Perturb independent local mass or per-box gas columns simultaneously.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L20-L60)

## particula.execution._condensation_ros3_kernels.solve_gas

*function*

```python
def solve_gas(local: FloatMatrix, gas_column: FloatMatrix, counts: FloatMatrix, volume: FloatVector, scale: FloatVector, rhs: FloatVector, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, resolved: wp.int32, inverse_step_gamma: wp.float64, solution: FloatVector, invalid: IntVector)
```

Eliminate particle coordinates with conservative weighted gas rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L97-L140)

## particula.execution._condensation_ros3_kernels.solve_remaining

*function*

```python
def solve_remaining(local: FloatMatrix, gas_column: FloatMatrix, rhs: FloatVector, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, inverse_step_gamma: wp.float64, solution: FloatVector, invalid: IntVector)
```

Back-substitute all local coordinates without a dense factorization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L143-L176)

## particula.execution._condensation_ros3_kernels.stage_rhs

*function*

```python
def stage_rhs(rate: FloatVector, first: FloatVector, second: FloatVector, partial: FloatVector, h: wp.float64, stage: wp.int32, rhs: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L193-L214)

## particula.execution._condensation_ros3_kernels.stage_state

*function*

```python
def stage_state(state: FloatVector, increment: FloatVector, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L217-L222)
