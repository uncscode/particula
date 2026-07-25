# CPU Nucleation Strategy System

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
selected scale times that pre-step total plus its finalized source.

## Scope, dependencies, and example

This contract does not claim universal empirical validity, implicit survival or
growth correction, full Vehkamäki physics, GPU execution, dynamic slots,
automatic scheduling, hidden CPU/GPU transfer or fallback, or performance.
The survival relation is an explicit caller factor, not an implicit
Kerminen--Kulmala calculation.

See [Fixed-Capacity Slot Exhaustion Primitives](slot_exhaustion_policies.md),
the [equations](../Theory/Technical/Dynamics/Nucleation_Equations.md), and the
[supported CPU example](../Examples/Nucleation/cpu_nucleation.py). E6-F8 direct
Warp nucleation and E6-F9 GPU integration/example orchestration remain
deferred; see the [GPU roadmap](Roadmap/data-oriented-gpu.md).

Focused validation:

```bash
python docs/Examples/Nucleation/cpu_nucleation.py
pytest particula/tests/nucleation_docs_test.py -q -Werror
pytest particula/dynamics/nucleation/tests/ \
  particula/dynamics/tests/nucleation_runnable_test.py \
  particula/integration_tests/nucleation_process_test.py -q -Werror
mkdocs build --strict
```
