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

## particula.execution._condensation_ros3_kernels.extract_derivative

*function*

```python
def extract_derivative(plus: FloatVector, minus: FloatVector, span: FloatVector, counts: FloatMatrix, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, gas_columns: wp.int32, result: FloatMatrix, invalid: IntVector)
```

Extract local derivatives from colored physical-rate evaluations.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L75-L106)

## particula.execution._condensation_ros3_kernels.finish

*function*

```python
def finish(state: FloatVector, first: FloatVector, second: FloatVector, third: FloatVector, atol: FloatVector, rtol: FloatVector, candidate: FloatVector, norm: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L223-L243)

## particula.execution._condensation_ros3_kernels.partial_time

*function*

```python
def partial_time(base: FloatVector, shifted: FloatVector, delta: wp.float64, result: FloatVector)
```

Differentiate forcing at fixed state inside one smooth time segment.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L246-L255)

## particula.execution._condensation_ros3_kernels.perturb

*function*

```python
def perturb(state: FloatVector, scale: FloatVector, counts: FloatMatrix, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, gas_columns: wp.int32, plus: FloatVector, minus: FloatVector, span: FloatVector)
```

Perturb independent local mass or per-box gas columns simultaneously.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L32-L72)

## particula.execution._condensation_ros3_kernels.solve_gas

*function*

```python
def solve_gas(local: FloatMatrix, gas_column: FloatMatrix, counts: FloatMatrix, volume: FloatVector, scale: FloatVector, rhs: FloatVector, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, resolved: wp.int32, inverse_step_gamma: wp.float64, solution: FloatVector, invalid: IntVector)
```

Eliminate particle coordinates with conservative weighted gas rows.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L109-L152)

## particula.execution._condensation_ros3_kernels.solve_remaining

*function*

```python
def solve_remaining(local: FloatMatrix, gas_column: FloatMatrix, rhs: FloatVector, materials: wp.int32, gases: wp.int32, target: wp.int32, source: wp.int32, inverse_step_gamma: wp.float64, solution: FloatVector, invalid: IntVector)
```

Back-substitute all local coordinates without a dense factorization.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L155-L188)

## particula.execution._condensation_ros3_kernels.stage_rhs

*function*

```python
def stage_rhs(rate: FloatVector, first: FloatVector, second: FloatVector, partial: FloatVector, h: wp.float64, stage: wp.int32, rhs: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L191-L212)

## particula.execution._condensation_ros3_kernels.stage_state

*function*

```python
def stage_state(state: FloatVector, increment: FloatVector, result: FloatVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_ros3_kernels.py#L215-L220)
