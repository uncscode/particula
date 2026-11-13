# particula.gpu.kernels._condensation_native

*module*

Private mapped condensation reductions and read-only native admission.

Particle transfer arrays remain per-particle kg in particle-material order.
Gas reductions use kg/m^3 in complete gas order. Explicit device maps contain
-1 for unmapped/skipped lanes; no species identity is inferred from widths.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L1-L1)

## particula.gpu.kernels._condensation_native._density_weight

*function*

```python
def _density_weight(concentration: Any, volume: Any, kind: wp.int32, box: int, slot: int) -> wp.float64
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L85-L96)

## particula.gpu.kernels._condensation_native._validate_primary

*function*

```python
def _validate_primary(masses: Any, concentration: Any, charge: Any, density: Any, volume: Any, gas_concentration: Any, molar_mass: Any, kind: wp.int32, invalid: Any) -> None
```

Check physical native values without writing caller storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L13-L58)

## particula.gpu.kernels._condensation_native.energy_transfer

*function*

```python
def energy_transfer(total: Any, latent_heat: Any, energy: Any, gas_to_particle: Any, concentration: Any, volume: Any, kind: wp.int32) -> None
```

Write signed extensive energy in J in full gas order.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L227-L247)

## particula.gpu.kernels._condensation_native.finalize_transfer

*function*

```python
def finalize_transfer(masses: Any, candidate: Any, scale: Any, finalized: Any, particle_to_gas: Any) -> None
```

Retain the representable particle change after mapped limiting.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L150-L169)

## particula.gpu.kernels._condensation_native.gate_transfer

*function*

```python
def gate_transfer(partitioning: Any, concentration: Any, proposal: Any, particle_to_gas: Any) -> None
```

Zero disabled full-gas lanes, absent links and inactive particles.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L99-L114)

## particula.gpu.kernels._condensation_native.reduce_candidates

*function*

```python
def reduce_candidates(candidate: Any, concentration: Any, demand: Any, release: Any, gas_to_particle: Any, volume: Any, kind: wp.int32, masses: Any) -> None
```

Reduce bounded particle proposals into full-gas density inventory.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L117-L147)

## particula.gpu.kernels._condensation_native.reduce_finalized

*function*

```python
def reduce_finalized(finalized: Any, concentration: Any, gas_delta: Any, gas_to_particle: Any, volume: Any, kind: wp.int32, masses: Any, gas_concentration: Any) -> None
```

Reconcile rounded uptake before either primary inventory is updated.

One lane owns each mapped species, so corrections and the weighted sum
use the same deterministic order. A downward relative-epsilon step moves
gaining masses by at least one representable value (Warp has no nextafter).
If four corrections cannot fit the budget, suppress uptake for that species.
Nonfinite reductions suppress all transfers for that species.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L172-L224)

## particula.gpu.kernels._condensation_native.retire_evaporated_slots

*function*

```python
def retire_evaporated_slots(masses: Any, concentration: Any, charge: Any) -> None
```

Clear fully evaporated resolved slots after gas/energy accounting.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L250-L262)

## particula.gpu.kernels._condensation_native.validate_primary_state

*function*

```python
def validate_primary_state(particles: Any, gas: Any, kind: int) -> None
```

Read back only a private validation flag before any state writer.

[View source](https://github.com/uncscode/particula/blob/main/particula/gpu/kernels/_condensation_native.py#L61-L82)
