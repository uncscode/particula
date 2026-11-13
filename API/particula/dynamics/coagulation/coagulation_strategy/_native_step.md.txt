# particula.dynamics.coagulation.coagulation_strategy._native_step

*module*

Staged native CPU coagulation using the existing rate/selector algorithms.

PMF/PDF concentrations are densities at unit volume. Resolved slots carry
equal positive counts during a merge; their common multiplicity scales the
selector's duration, leaving the physical volume and raw counts unchanged.
No caller array is written until the complete proposed state is validated.
RNG draws are not rolled back if selection or proposal validation fails.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy._native_step._density_step

*function*

```python
def _density_step(strategy: Any, particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Stage the existing number update, retaining fixed masses and charge.

Size-dependent composition can produce persistent per-species inventory
defects. This accepted PMF/PDF limitation is documented separately from
the resolved merge's species/charge conservation contract.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L257-L285)

## particula.dynamics.coagulation.coagulation_strategy._native_step._direct_kernel

*function*

```python
def _direct_kernel(strategy: Any, particle: ParticleData, temp: float, press: float) -> Any
```

Retain the existing charged-model direct callback and fallback policy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L166-L204)

## particula.dynamics.coagulation.coagulation_strategy._native_step._merge

*function*

```python
def _merge(particle: ParticleData, active: NDArray[np.int64], pairs: Any) -> None
```

Validate selector output, stage the merge, then publish in place.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L207-L254)

## particula.dynamics.coagulation.coagulation_strategy._native_step._validate_storage

*function*

```python
def _validate_storage(particle: ParticleData, resolved: bool) -> None
```

Reject unwritable or overlapping destinations before draws.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L143-L163)

## particula.dynamics.coagulation.coagulation_strategy._native_step.binned_kernel_data

*function*

```python
def binned_kernel_data(particle: ParticleData, edges: NDArray[np.float64]) -> ParticleData
```

Build temporary count-weighted material representatives, not new state.

Counts and species inventories are preserved across occupied bins. Charge
retains the existing median approximation for kernel evaluation only; this
temporary charge is never used for the authoritative merge bookkeeping.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L94-L140)

## particula.dynamics.coagulation.coagulation_strategy._native_step.kernel_edges

*function*

```python
def kernel_edges(particle: ParticleData, edges: Any, total_bins: Any, bins_per_decade: Any) -> NDArray[np.float64]
```

Generate the existing logarithmic grid from active native slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L68-L91)

## particula.dynamics.coagulation.coagulation_strategy._native_step.step_native

*function*

```python
def step_native(strategy: Any, particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Apply an admitted native step without changing process/RNG ownership.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L288-L375)

## particula.dynamics.coagulation.coagulation_strategy._native_step.validate_bin_controls

*function*

```python
def validate_bin_controls(edges: Any, total_bins: Any, bins_per_decade: Any) -> None
```

Validate mutable bin controls even on a zero-work step.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L48-L65)

## particula.dynamics.coagulation.coagulation_strategy._native_step.validate_edges

*function*

```python
def validate_edges(edges: Any) -> NDArray[np.float64]
```

Require real, positive, strictly increasing kernel radius edges.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/_native_step.py#L32-L45)
