# particula.execution._condensation_block_kernels

*module*

Conservative block-arrowhead linear algebra for material-coupled ROS3.

Each particle has a dense local material Jacobian A and gas derivative B.
Inventory conservation fixes the gas rows to minus the weighted particle
rows. Eliminate local blocks M=qI-A, then solve the small per-box Schur
system q*(I+sum(W*M^-1*B)). No population-sized dense matrix is allocated.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L1-L1)

## particula.execution._condensation_block_kernels.assemble_reservoirs

*function*

```python
def assemble_reservoirs(local: FloatTensor, response: FloatTensor, particular: FloatMatrix, counts: FloatMatrix, volume: FloatVector, scale: FloatVector, rhs: FloatVector, targets: IntVector, sources: IntVector, materials: int, gases: int, resolved: int, h: FloatVector, schur: FloatTensor)
```

Assemble independent Schur entries in parallel, in fixed slot order.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L131-L175)

## particula.execution._condensation_block_kernels.back_substitute

*function*

```python
def back_substitute(response: FloatTensor, particular: FloatMatrix, rhs: FloatVector, targets: IntVector, sources: IntVector, reverse_target: IntVector, reverse_source: IntVector, slots: int, materials: int, gases: int, h: FloatVector, solution: FloatVector, invalid: IntVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L220-L255)

## particula.execution._condensation_block_kernels.eliminate_local

*function*

```python
def eliminate_local(inverse: FloatTensor, gas_columns: FloatTensor, rhs: FloatVector, targets: IntVector, materials: int, response: FloatTensor, particular: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L108-L128)

## particula.execution._condensation_block_kernels.extract

*function*

```python
def extract(plus: FloatVector, minus: FloatVector, span: FloatVector, counts: FloatMatrix, targets: IntVector, sources: IntVector, materials: int, gases: int, column: int, gas_column: int, derivative: FloatTensor, invalid: IntVector)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L20-L56)

## particula.execution._condensation_block_kernels.invert_local

*function*

```python
def invert_local(local: FloatTensor, h: FloatVector, augmented: FloatTensor, inverse: FloatTensor, invalid: IntVector)
```

Invert each small local block with partial pivoting in float64.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L59-L105)

## particula.execution._condensation_block_kernels.solve_reservoirs

*function*

```python
def solve_reservoirs(schur: FloatTensor, sources: IntVector, mass_size: int, gases: int, solution: FloatVector, invalid: IntVector)
```

Pivot the small per-box Schur matrix.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_block_kernels.py#L178-L217)
