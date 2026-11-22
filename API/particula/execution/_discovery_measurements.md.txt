# particula.execution._discovery_measurements

*module*

Differentiable log-diameter response for the SiMBA inverse workflow.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_measurements.py#L1-L1)

## particula.execution._discovery_measurements._below

*function*

```python
def _below(edge: wp.float64, diameter: wp.float64, width: wp.float64)
```

Probability of a reported diameter below a finite detector edge.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_measurements.py#L11-L15)

## particula.execution._discovery_measurements.observe_number_spectrum

*function*

```python
def observe_number_spectrum(diameter: FloatMatrix, number: FloatMatrix, edges: FloatVector, log_width: wp.float64, output: FloatMatrix)
```

Integrate counts per m³ by diameter, including underflow and overflow.

Diameter and strictly ascending edges are in metres. Each cohort has a
Gaussian response in natural-log diameter with positive log_width. The
output columns are number concentrations per bin, including both tails.

[View source](https://github.com/uncscode/particula/blob/main/particula/execution/_discovery_measurements.py#L18-L42)
