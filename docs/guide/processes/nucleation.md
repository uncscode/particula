# Nucleation Strategy System

For the supported CUDA configuration of this runnable, use
[prepared GPU simulations](../gpu/prepared.md). The CPU and direct-kernel
contracts below retain their own, sometimes broader, capabilities.

## Complete direct-process context

The direct nucleation boundary is shown last in the illustrative [explicit-transfer sequence](https://github.com/Gorkowski/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py). Its [direct-process validation evidence](https://github.com/Gorkowski/particula/blob/main/particula/gpu/tests/process_sequence_test.py) and [example regression](https://github.com/Gorkowski/particula/blob/main/examples_tests/gpu_complete_process_sequence_example_test.py) do not turn the example into a public coordinator, GPU `Runnable`, backend selector, resident loop, or CPU fallback.

> Convert partitioning precursor gas into fixed-capacity particle slots with a
> public, conservation-accounted CPU runnable.

The CPU execution method of the nucleation runnable executes each resolved box independently
through the shared transactional driver. It combines immutable potential-rate construction (public construction) with the public
`Nucleation` runnable (runnable execution). Its CPU execution method is not a GPU scheduler or general
empirical new-particle-formation model.

## Public API, equations, and units

Use public process imports and the concrete species-map configuration. `EnvironmentData` supplies each box’s thermodynamic input;
`ExhaustionControls` selects fixed-capacity recovery behavior.

```python
import numpy as np

from particula.dynamics import (
    ActivationNucleationBuilder,
    Nucleation,
    NucleationCommitConfig,
    NucleationSourceConfig,
)
from particula import Aerosol
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from particula.dynamics.nucleation.species_mapping import NucleationSpeciesMap
from particula.particles.exhaustion import ExhaustionControls
```

Activation uses $J=A C S$ and kinetic nucleation uses $J=K C^2 S$, where
$C$ is precursor number concentration [#/m³], $A$ is [s⁻¹], $K$ is [m³/s],
and $S$ is a caller-provided dimensionless survival factor. Input precursor
mass concentration [kg/m³] is converted with $C=cN_A/M$. These strategies
return a potential rate [#/m³/s]; they do not mutate state.

`NucleationSourceConfig`, `NucleationSourceConfigBuilder`, the activation and
kinetic strategies/builders, `NucleationFactory`, `Nucleation`, and
`NucleationCommitConfig` are exported by `particula.dynamics`. Demand planning/source transaction records
and `finalize_particle_source`, `commit_particle_source`, and
`ParticleSourceCommitConfig` are concrete-only implementation details in
`nucleation.particle_source`, not supported package imports.

Build an immutable potential-rate strategy through the public builder or
factory, then select its precursor lane in `NucleationSourceConfig`:

```python
particles = ParticleData(
    masses=np.zeros((1, 3, 1)), concentration=np.zeros((1, 3)),
    charge=np.zeros((1, 3)), density=np.array([1000.0]),
    volume=np.array([0.25]), distribution_type="particle_resolved",
)
gas = GasData(
    name=["precursor", "inert"], molar_mass=np.array([0.1, 0.04]),
    concentration=np.array([[1e-12, 2e-6]]), partitioning=np.array([True, False]),
)
environment = EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0, 1.0]],
)
aerosol = Aerosol(particles=particles, gas=gas, environment=environment)
strategy = ActivationNucleationBuilder().set_parameters(
    {
        "coefficient": 1.0e-2,
        "coefficient_units": "s^-1",
        "coefficient_provenance": "configured source",
        "precursor_number_concentration_lower": 1.0,
        "precursor_number_concentration_lower_units": "1/m^3",
        "precursor_number_concentration_upper": 1.0e30,
        "precursor_number_concentration_upper_units": "1/m^3",
        "temperature_lower": 200.0,
        "temperature_lower_units": "K",
        "temperature_upper": 400.0,
        "temperature_upper_units": "K",
        "injection_composition": [1, 0],
        "formation_diameter": 1.0,
        "formation_diameter_units": "nm",
    }
).build()
source = NucleationSourceConfig(
    strategy=strategy, precursor_index=0,
    species_map=NucleationSpeciesMap(tuple(gas.name), ((0, 0),)),
)
```

## Getting started

Use the [supported CPU example](https://github.com/uncscode/particula/blob/main/docs/examples/processes/nucleation/cpu_nucleation.py) as
the runnable starting point.
It constructs native `Aerosol`, unified `GasData` with precursor and inert lanes,
fixed resolved particle capacity, aligned `EnvironmentData`, and
the public `NucleationCommitConfig`. Its commit controls use
`ExhaustionControls()` and per-box `float64` `requested_scale`,
`minimum_scale`, and `minimum_volume` sidecars (one element in this example). Do not replace that setup with
imports from `nucleation.particle_source`.

## Execution, fixed capacity, and accounting

The runnable retains the native aggregate and its containers by identity.
It supports resolved batches with fixed-capacity slots in each box. The shared
driver slices exhaustion controls and stages volume changes per box. A failed
box rejects the entire current substep. Full ordered
gas names and explicit gas-to-particle pairs define the source mapping; gas
and particle widths may differ. Every positive injection lane must be mapped
and partitioning-enabled; other gas lanes remain unchanged. It does not resize
or compact slots. It divides the duration into equal sequential substeps and
re-reads current gas after each successful substep:

```python
nucleation = Nucleation(
    source,
    NucleationCommitConfig(
        maximum_slot_weight=1e20, source_charge=0.0,
        exhaustion_controls=ExhaustionControls(),
        requested_scale=np.array([1.0]), minimum_scale=np.array([1.0]),
        minimum_volume=np.array([0.01]),
    ),
)
result = nucleation.execute(aerosol, time_step=1.0, sub_steps=2)
assert result is aerosol
```

Demand planning is nonmutating. Source transaction rejects invalid preflight atomically. Runnable execution makes
each attempted substep a transaction, but is not whole-call atomic: earlier
successful substeps remain if a later substep fails. Zero-duration and
zero-admitted-demand paths are no-ops after their applicable validation.
`Nucleation.execute()` returns the mutated `Aerosol`; it does not return demand planning/source transaction
diagnostic records. The concrete-only demand planning/source transaction transaction diagnostics include
potential, admitted, gas-limited, represented, and reduced events; limiting
species; requested, activated, and released slots; policy/scale; gas mass
removed; and conservation residual. Particle inventory is
concentration-weighted. Per-box/per-species particle-plus-gas conservation uses
`rtol=1e-12, atol=1e-30`.

The source relies on fixed-slot activation and slot exhaustion policy resolution. Demand
is never silently truncated. On exhaustion, enabled resampling is considered
first and representative-volume scaling is its fallback; if neither can
represent the requested demand, the attempt raises rather than silently
discarding it. Native representative-volume scaling scales counts and volume together,
preserving pre-existing particle density and gas concentration. It differs from
physical volume evolution. The added source is accounted against that scaled
extensive inventory. CPU admitted event density is multiplied by current
volume; `maximum_slot_weight` bounds represented counts per new slot.

## Scope, dependencies, and example

This contract does not claim universal empirical validity, implicit survival or
growth correction, full Vehkamäki physics, GPU execution, dynamic slots,
automatic scheduling, hidden CPU/GPU transfer or fallback, or performance.
The survival relation is an explicit caller factor, not an implicit
Kerminen--Kulmala calculation.

See [Fixed-Capacity Slot Exhaustion Primitives](../gpu/slot-exhaustion.md),
the [equations](../../theory/dynamics/nucleation-equations.md), and the
[supported CPU example](https://github.com/uncscode/particula/blob/main/docs/examples/processes/nucleation/cpu_nucleation.py).

## Direct-Warp low-level step

Direct GPU nucleation ships a bounded direct-Warp nucleation step. Import only its supported
low-level entry point:

```python
from particula.gpu.kernels import nucleation_step_gpu
```

`NucleationConfig`, demand planning/capacity staging records, exhaustion resolution controls,
`NucleationScratchBuffers`, `NucleationFinalizedDemandBuffers`,
`NucleationDiagnosticBuffers`, `NucleationExhaustionBuffers`, and helpers are
concrete-only imports from `particula.gpu.kernels.nucleation`. Nested
`ResamplingBuffers` is concrete-only in `particula.gpu.kernels.exhaustion`.
They are deliberately not package exports. The CPU `Nucleation` execution method
remains the supported high-level CPU process; it is not a GPU fallback.

Callers explicitly convert `ParticleData`, `GasData`, and `EnvironmentData`,
place arrays on one Warp device, and call `wp.synchronize()` before host
inspection. Fixed schemas use `B` boxes, `N` particle slots, and `S` species:
particle masses are `(B, N, S)`, particle concentration/charge are `(B, N)`,
gas concentration is `(B, Sg)`, partitioning is `(B, Sg)`, and temperature
is `(B,)`. Native particle width `Sp` may differ from full gas width `Sg`.
Direct and prepared calls require explicitly tagged resolved inputs and reject
untagged, discrete, and continuous-PDF metadata before particle-array access.
Each call requires a `NucleationSpeciesBinding` prepared with
`prepare_nucleation_species_map` from
`particula.gpu.nucleation_species_mapping`, using preserved full gas names and
the explicit source map. Preparation binds the exact transferred arrays;
replacement or restart needs fresh preparation. The direct GPU path uses
integral int32 event counts and one unit-count slot per event; it does not
implement the CPU weighted-slot packing scheme.
Sidecars are caller-owned, same-device, contiguous `wp.float64` or `wp.int32`
arrays with documented `(B,)`, `(B, N)`, `(B, S)`, and `(B, N, S)` schemas.
Full-gas demand/transfer sidecars use gas width; resampling scratch uses
particle width. Exhaustion resolution owns nested slot exhaustion resampling/scaling sidecars; see
[fixed-slot exhaustion](../gpu/slot-exhaustion.md) for their fields.
Preflight and phase handoffs may perform bounded aggregate device-status
readbacks for validation; these are not CPU state transfers or a fallback.

Input validation performs read-only preflight, demand planning admits one inventory-limited
demand, capacity staging stages counts/free-slot prefixes, and exhaustion resolution uses fully viable resampling
before representative-volume-scaling fallback. Commit fuses selected-slot activation
and gas transfer. Free-index tails are `-1`; demand is never silently truncated.
A valid configured gate, no-admission, or zero-work result is a successful
no-op, not an error or hidden recovery.

Invalid input rejected before volume scaling primitive entry preserves particle/gas state.
Resampling--volume scaling can write documented sidecars before a later rejection. Once slot exhaustion is
entered, or a policy integration writer launches, rollback is not promised. Success returns the
identical particle and gas containers.

This boundary excludes hidden CPU↔Warp state transfer, CPU fallback,
resize/compaction, GPU `Runnable`, scheduler/backend selection, direct-process integration
orchestration, expanded physics, graph capture, autodiff, and performance
guarantees. Fixed-slot management fixed-slot activation, slot exhaustion primitives, and CPU nucleation are dependencies; direct-process integration is only a downstream explicit-transfer
integration consumer. See the [equations](../../theory/dynamics/nucleation-equations.md),
[direct example](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_nucleation.py), and
[GPU roadmap](../../contribute/roadmap.md).

Focused validation:

```bash
python docs/examples/processes/nucleation/cpu_nucleation.py
python -Werror docs/examples/gpu/direct/gpu_direct_nucleation.py
pytest examples_tests/nucleation_example_test.py -q -Werror
pytest examples_tests/gpu_direct_nucleation_example_test.py -q -Werror
pytest particula/dynamics/nucleation/tests/ \
  particula/dynamics/tests/nucleation_runnable_test.py \
  particula/integration_tests/nucleation_process_test.py -q -Werror
npm --prefix website run build
```
