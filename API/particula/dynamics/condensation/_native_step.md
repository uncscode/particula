# particula.dynamics.condensation._native_step

*module*

Staged native condensation inventories and storage preflight.

Particle changes are per-particle kg. Weights are already representation-aware
number densities, including radius quadrature for PDFs. Gas stays in its full
ordered kg/m3 storage. No helper here mutates caller-owned arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L1-L1)

## particula.dynamics.condensation._native_step._finalize_species

*function*

```python
def _finalize_species(masses: NDArray[np.float64], weights: NDArray[np.float64], proposal: NDArray[np.float64], gas_concentration: float) -> tuple[NDArray[np.float64], float]
```

Limit evaporation first, then uptake against gas plus actual release.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L98-L135)

## particula.dynamics.condensation._native_step._weighted_sum

*function*

```python
def _weighted_sum(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float
```

Sum species inventory independently of cancellation/reduction order.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L79-L95)

## particula.dynamics.condensation._native_step.finalize_native_transfer

*function*

```python
def finalize_native_transfer(masses: NDArray[np.float64], weights: NDArray[np.float64], gas_concentration: NDArray[np.float64], proposal: NDArray[np.float64], sources: NDArray[np.intp], targets: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.float64]]
```

Return staged particle/gas states for a validated explicit lane map.

Each proposal is per-particle mass change in kg, already multiplied by
its local timestep. Zero-weight nodes, unmapped particle material, and all
unmapped gas remain unchanged. Actual per-particle evaporation is clipped
before its release is made available to uptake. This avoids crediting a
request to evaporate more mass than a particle owns.

**Parameters**

- `masses` (`NDArray[np.float64]`) — Validated nonnegative particle masses, shape (nodes, species).
- `weights` (`NDArray[np.float64]`) — Validated inventory density per node, shape (nodes,).
- `gas_concentration` (`NDArray[np.float64]`) — Validated full ordered gas concentration, kg/m3.
- `proposal` (`NDArray[np.float64]`) — Finite per-particle changes with exactly the mass-array shape.
- `sources` (`NDArray[np.intp]`) — Participating full-gas indices from native state admission.
- `targets` (`NDArray[np.intp]`) — Matching unique particle indices from native state admission.

**Returns**

- (`NDArray[np.float64]`) — Fresh particle masses and full gas concentration; input arrays retain
- (`NDArray[np.float64]`) — their values and identities. The caller decides whether to commit gas
- (`tuple[NDArray[np.float64], NDArray[np.float64]]`) — changes or retain its externally maintained reservoir.

**Raises**

- `ValueError` — If proposals have invalid shape/values or a transfer cannot
be represented conservatively with finite floating-point storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L138-L188)

## particula.dynamics.condensation._native_step.retire_evaporated_slots

*function*

```python
def retire_evaporated_slots(particle: ParticleData) -> None
```

Release fully evaporated resolved slots after inventory accounting.

The particle no longer exists: clear its count and charge, preserving
fixed storage and every surviving particle. No gas-phase charge model is
implied. Density-grid populations retain their prescribed bin semantics.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L64-L76)

## particula.dynamics.condensation._native_step.validate_native_step_storage

*function*

```python
def validate_native_step_storage(particle: ParticleData, gas: GasData, update_gases: bool) -> None
```

Reject read-only destinations and aliases with any protected field.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L29-L61)

## particula.dynamics.condensation._native_step.validate_native_time_step

*function*

```python
def validate_native_time_step(time_step: float) -> None
```

Require a finite scalar duration, excluding Boolean control values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/_native_step.py#L18-L26)
