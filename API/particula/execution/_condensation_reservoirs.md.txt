# particula.execution._condensation_reservoirs

*module*

Deterministic parallel reservoir reductions with bounded device scratch.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_reservoirs.py#L1-L1)

## particula.execution._condensation_reservoirs.assemble_parts

*function*

```python
def assemble_parts(local: FloatTensor, response: FloatTensor, particular: FloatMatrix, counts: FloatMatrix, volume: FloatVector, scale: FloatVector, rhs: FloatVector, targets: IntVector, sources: IntVector, materials: int, gases: int, resolved: int, h: FloatVector, parts: FloatTensor)
```

Sum fixed strided slot groups without atomic floating-point additions.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_reservoirs.py#L56-L95)

## particula.execution._condensation_reservoirs.finish_gas

*function*

```python
def finish_gas(parts: FloatTensor, mass_size: int, scale: FloatVector, derivative: FloatVector, invalid: IntVector)
```

Finish in fixed lane order and write scaled gas derivatives.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_reservoirs.py#L36-L53)

## particula.execution._condensation_reservoirs.finish_schur

*function*

```python
def finish_schur(parts: FloatTensor, rhs: FloatVector, sources: IntVector, mass_size: int, gases: int, h: FloatVector, schur: FloatTensor)
```

Add the diagonal/RHS once after deterministic partial reduction.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_reservoirs.py#L98-L120)

## particula.execution._condensation_reservoirs.gas_parts

*function*

```python
def gas_parts(rate: FloatTensor, counts: FloatMatrix, volume: FloatVector, mapping: IntVector, resolved: int, parts: FloatTensor)
```

Retain native slot weights and reduce independent fixed slot groups.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_condensation_reservoirs.py#L14-L33)
