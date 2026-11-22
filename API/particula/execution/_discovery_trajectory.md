# particula.execution._discovery_trajectory

*module*

Warp kernels for deterministic gas chemistry and seeded chamber growth.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L1-L1)

## particula.execution._discovery_trajectory.PI

*attribute* · *module attribute*

```python
PI = wp.constant(wp.float64(3.141592653589793))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L17-L17)

## particula.execution._discovery_trajectory.THIRD

*attribute* · *module attribute*

```python
THIRD = wp.constant(wp.float64(1 / 3))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L18-L18)

## particula.execution._discovery_trajectory._chemistry

*function*

```python
def _chemistry(b: int, s: int, kind: int, volume: wp.float64, state: FloatMatrix, rates: FloatVector, phase: IntVector, first: IntVector, second: IntVector, change: wp.array2d(dtype=int))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L108-L128)

## particula.execution._discovery_trajectory.advance

*function*

```python
def advance(state: FloatMatrix, slope: FloatMatrix, dt: wp.float64, result: FloatMatrix, invalid: IntVector)
```

Advance one explicit stage and flag failed domain checks.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L176-L189)

## particula.execution._discovery_trajectory.cohort_observables

*function*

```python
def cohort_observables(bulk: FloatMatrix, diam: FloatMatrix, count: FloatMatrix)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L192-L198)

## particula.execution._discovery_trajectory.derivative

*function*

```python
def derivative(state: FloatMatrix, bulk: FloatMatrix, flux: FloatMatrix, gas: IntVector, particle: IntVector, rates: FloatVector, phase: IntVector, first: IntVector, second: IntVector, change: wp.array2d(dtype=int), seed: FloatVector, result: FloatMatrix)
```

Compute molar tendencies and accumulate deposited inventories.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L131-L173)

## particula.execution._discovery_trajectory.exchange

*function*

```python
def exchange(state: FloatMatrix, bulk: FloatMatrix, gas: IntVector, particle: IntVector, diffusion: FloatVector, mean_path: FloatVector, saturation: FloatVector, flux: FloatMatrix)
```

Evaluates reversible ideal-solution transfer at unit accommodation.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L84-L105)

## particula.execution._discovery_trajectory.geometry

*function*

```python
def geometry(inventory: FloatMatrix, bulk: FloatMatrix)
```

Use the saved volume reduction when differentiating the cube root.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L72-L81)

## particula.execution._discovery_trajectory.initialize

*function*

```python
def initialize(dose: FloatVector, seed: FloatVector, quantile: FloatVector, fractions: FloatVector, molar_volume: FloatVector, state: FloatMatrix)
```

Initialize gas and equal-number lognormal seed cohorts in mol/m³ air.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L21-L52)

## particula.execution._discovery_trajectory.inventory

*function*

```python
def inventory(state: FloatMatrix, molar_volume: FloatVector, bulk: FloatMatrix)
```

Return suspended volume fraction, number, diameter and total moles.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L55-L69)

## particula.execution._discovery_trajectory.observe

*function*

```python
def observe(state: FloatMatrix, spectra: FloatMatrix, masses: FloatVector, organic: IntVector, gases: IntVector, output: FloatVector)
```

Observe gas and organic mass in kg/m³ and number per instrument bin.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_trajectory.py#L201-L223)
