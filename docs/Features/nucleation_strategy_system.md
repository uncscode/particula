# Nucleation Strategy System

> Convert partitioning precursor gas into fixed-capacity particle slots with a
> public, conservation-accounted CPU runnable.

The shipped CPU-only nucleation boundary is a deliberately bounded, single-box
process. It combines immutable potential-rate construction (P4) with the public
`Nucleation` runnable (P5). It is not a GPU process, scheduler, or general
empirical new-particle-formation model.

## Public API, equations, and units

Use public imports only. `EnvironmentData` is the one-box thermodynamic input;
`ExhaustionControls` selects fixed-capacity recovery behavior.

```python
import numpy as np

from particula.dynamics import (
    ActivationNucleationBuilder,
    Nucleation,
    NucleationCommitConfig,
    NucleationSourceConfig,
)
from particula.gas import EnvironmentData
from particula.particles.exhaustion import ExhaustionControls
```

Activation uses $J=A C S$ and kinetic nucleation uses $J=K C^2 S$, where
$C$ is precursor number concentration [#/m³], $A$ is [s⁻¹], $K$ is [m³/s],
and $S$ is a caller-provided dimensionless survival factor. Input precursor
mass concentration [kg/m³] is converted with $C=cN_A/M$. These strategies
return a potential rate [#/m³/s]; they do not mutate state.

`NucleationSourceConfig`, `NucleationSourceConfigBuilder`, the activation and
kinetic strategies/builders, `NucleationFactory`, `Nucleation`, and
`NucleationCommitConfig` are exported by `particula.dynamics`. P2/P3 records
and `finalize_particle_source`, `commit_particle_source`, and
`ParticleSourceCommitConfig` are concrete-only implementation details in
`nucleation.particle_source`, not supported package imports.

Build an immutable potential-rate strategy through the public builder or
factory, then select its precursor lane in `NucleationSourceConfig`:

```python
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
        "injection_composition": [1],
        "formation_diameter": 1.0,
        "formation_diameter_units": "nm",
    }
).build()
source = NucleationSourceConfig(strategy=strategy, precursor_index=0)
```

## Getting started

Use the [supported CPU example](../Examples/Nucleation/cpu_nucleation.py) as
the runnable starting point. It constructs the one-box legacy `Aerosol`, a
partitioning `GasData` lane, fixed particle capacity, `EnvironmentData`, and
the public `NucleationCommitConfig`. Its commit controls use
`ExhaustionControls()` and one-element `float64` `requested_scale`,
`minimum_scale`, and `minimum_volume` sidecars. Do not replace that setup with
imports from `nucleation.particle_source`.

## Execution, fixed capacity, and accounting

The runnable adapts a legacy `Aerosol` backing `ParticleData` and partitioning
`GasData` by identity. It supports exactly one box, fixed-capacity slots, and
partitioning gas with matching particle/gas species widths. It does not resize
or compact slots. It divides the duration into equal sequential substeps and
re-reads current gas after each successful substep:

```python
result = nucleation.execute(aerosol, time_step=1.0, sub_steps=2)
assert result is aerosol
```

P2 planning is nonmutating. P3 rejects invalid preflight atomically. P5 makes
each attempted substep a transaction, but is not whole-call atomic: earlier
successful substeps remain if a later substep fails. Zero-duration and
zero-admitted-demand paths are no-ops after their applicable validation.
`Nucleation.execute()` returns the mutated `Aerosol`; it does not return P2/P3
diagnostic records. The concrete-only P2/P3 transaction diagnostics include
potential, admitted, gas-limited, represented, and reduced events; limiting
species; requested, activated, and released slots; policy/scale; gas mass
removed; and conservation residual. Particle inventory is
concentration-weighted. Per-box/per-species particle-plus-gas conservation uses
`rtol=1e-12, atol=1e-30`.

The source relies on E6-F5 slot activation and E6-F6 policy resolution. Demand
is never silently truncated. On exhaustion, enabled resampling is considered
first and representative-volume scaling is its fallback; if neither can
represent the requested demand, the attempt raises rather than silently
discarding it. Selected scaling scales pre-existing particle and gas state
before source removal. Therefore an unscaled run conserves against its direct
pre-step particle-plus-gas total, while a scaled row conserves against the
selected scale times that pre-step total.

## Scope, dependencies, and example

This contract does not claim universal empirical validity, implicit survival or
growth correction, full Vehkamäki physics, GPU execution, dynamic slots,
automatic scheduling, hidden CPU/GPU transfer or fallback, or performance.
The survival relation is an explicit caller factor, not an implicit
Kerminen--Kulmala calculation.

See [Fixed-Capacity Slot Exhaustion Primitives](slot_exhaustion_policies.md),
the [equations](../Theory/Technical/Dynamics/Nucleation_Equations.md), and the
[supported CPU example](../Examples/Nucleation/cpu_nucleation.py).

## Direct-Warp low-level step

E6-F8 ships a bounded direct-Warp nucleation step. Import only its supported
low-level entry point:

```python
from particula.gpu.kernels import nucleation_step_gpu
```

`NucleationConfig`, P2/P3 records, P4 controls,
`NucleationScratchBuffers`, `NucleationFinalizedDemandBuffers`,
`NucleationDiagnosticBuffers`, `NucleationExhaustionBuffers`, and helpers are
concrete-only imports from `particula.gpu.kernels.nucleation`. Nested
`ResamplingBuffers` is concrete-only in `particula.gpu.kernels.exhaustion`.
They are deliberately not package exports. The CPU-only `Nucleation` runnable
remains the supported high-level CPU process; it is not a GPU fallback.

Callers explicitly convert `ParticleData`, `GasData`, and `EnvironmentData`,
place arrays on one Warp device, and call `wp.synchronize()` before host
inspection. Fixed schemas use `B` boxes, `N` particle slots, and `S` species:
particle masses are `(B, N, S)`, particle concentration/charge are `(B, N)`,
gas concentration/partitioning are `(B, S)`, and temperature is `(B,)`.
Sidecars are caller-owned, same-device, contiguous `wp.float64` or `wp.int32`
arrays with documented `(B,)`, `(B, N)`, `(B, S)`, and `(B, N, S)` schemas.
P4 owns nested E6-F6 resampling/scaling sidecars; see
[fixed-slot exhaustion](slot_exhaustion_policies.md) for their fields.
Preflight and phase handoffs may perform bounded aggregate device-status
readbacks for validation; these are not CPU state transfers or a fallback.

P1 performs read-only preflight, P2 plans and admits one inventory-limited
demand, P3 stages counts/free-slot prefixes, and P4 uses fully viable resampling
before representative-volume-scaling fallback. P5 fuses selected-slot activation
and gas transfer. Free-index tails are `-1`; demand is never silently truncated.
A valid configured gate, no-admission, or zero-work result is a successful
no-op, not an error or hidden recovery.

Invalid input rejected before P4 primitive entry preserves particle/gas state.
P2--P4 can write documented sidecars before a later rejection. Once E6-F6 is
entered, or a P5 writer launches, rollback is not promised. Success returns the
identical particle and gas containers.

This boundary excludes hidden CPU↔Warp state transfer, CPU fallback,
resize/compaction, GPU `Runnable`, scheduler/backend selection, E6-F9
orchestration, expanded physics, graph capture, autodiff, and performance
guarantees. E6-F5 fixed-slot activation, E6-F6 exhaustion primitives, and E6-F7
CPU nucleation are dependencies; E6-F9 is only a downstream explicit-transfer
integration consumer. See the [equations](../Theory/Technical/Dynamics/Nucleation_Equations.md),
[direct example](../Examples/Nucleation/gpu_direct_nucleation.py), and
[GPU roadmap](Roadmap/data-oriented-gpu.md).

Focused validation:

```bash
python docs/Examples/Nucleation/cpu_nucleation.py
python -Werror docs/Examples/Nucleation/gpu_direct_nucleation.py
pytest particula/tests/nucleation_docs_test.py -q -Werror
pytest particula/gpu/tests/gpu_direct_nucleation_example_test.py -q -Werror
pytest particula/dynamics/nucleation/tests/ \
  particula/dynamics/tests/nucleation_runnable_test.py \
  particula/integration_tests/nucleation_process_test.py -q -Werror
mkdocs build --strict
```
