---
title: Dynamics Migration
---

# Dynamics Migration

## Using ParticleData and GasData in dynamics

For the canonical support-boundary summary, including the leading `n_boxes`
axis and explicit particle, gas, and environment helper boundaries, see
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md).

Condensation and coagulation strategies accept both legacy facades and the new
data containers. The return type matches the input type, but container
compatibility does not mean every CPU dynamics path supports multi-box
execution.

| CPU dynamics path | Containers accepted | Current CPU execution support | Multi-box CPU alternative |
| --- | --- | --- | --- |
| Condensation | Legacy `ParticleRepresentation` and `GasSpecies`, or `ParticleData` and `GasData` | `n_boxes == 1` only | Run a caller-managed per-box loop and pass one box at a time. |
| Coagulation | Legacy `ParticleRepresentation`, or `ParticleData` | `n_boxes == 1` only | Run a caller-managed per-box loop and pass one box at a time. |

The storage schema is broader than the audited CPU support boundary.
`ParticleData` and `GasData` can store multiple boxes, but current CPU
condensation and coagulation execution remains single-box. Multi-box inputs
fail fast instead of silently reading or mutating box `0`.

Existing process entry points may continue to accept scalar `temperature` and
`pressure`. Only migrated process code should read `EnvironmentData` directly,
and environment fields should remain read-only unless the physical model owns
the update and refreshes derived state such as `saturation_ratio`.

Supported single-box CPU usage:

```python
import particula as par

activity_strategy = par.particles.ActivityIdealMass()
surface_strategy = par.particles.SurfaceStrategyMass()
vapor_pressure_strategy = par.gas.ConstantVaporPressureStrategy(2330.0)

condensation = par.dynamics.CondensationIsothermal(
    molar_mass=0.018,
    activity_strategy=activity_strategy,
    surface_strategy=surface_strategy,
    vapor_pressure_strategy=vapor_pressure_strategy,
)
particle_out, gas_out = condensation.step(
    particle=particle_data,
    gas_species=gas_data,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)

coagulation = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete"
)
particle_out = coagulation.step(
    particle=particle_out,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

For multi-box CPU execution, manage the box loop in application code. This is
pseudocode, not a built-in Particula helper:

```python
for box_index in range(particle_data.n_boxes):
    single_box_particle = build_single_box_particle_data(
        particle_data,
        box_index,
    )
    single_box_gas = build_single_box_gas_data(gas_data, box_index)

    particle_box_out, gas_box_out = condensation.step(
        particle=single_box_particle,
        gas_species=single_box_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )

    particle_box_out = coagulation.step(
        particle=particle_box_out,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )

    # Reassemble results in caller-owned storage.
```

Use `EnvironmentData` for CPU-owned per-box `temperature`, `pressure`, and
`saturation_ratio`. `GasData` does not own these fields. Keep scalar inputs
where process APIs have not migrated, and use explicit conversion helpers for
GPU round trips. Kernels and runnables do not move environment state for the
caller.

## `condensation_step_gpu` environment inputs

The bounded direct GPU condensation path is imported with
`from particula.gpu.kernels import condensation_step_gpu`. Explicitly convert
with `to_warp_*` before the call and restore with `from_warp_*` afterward;
preserve ordered species names outside GPU containers. Callers also own required
synchronization and checkpoint/snapshot responsibility. See the authoritative
[Data Containers and GPU Foundations](../data-containers-and-gpu-foundations.md)
page for its modes and schema matrix rather than duplicating that matrix here.

The non-executable signature is
`particles_out, mass_transfer = condensation_step_gpu(...,
thermodynamics=thermodynamics)`. `thermodynamics=` is required and is
caller-owned device-local configuration. Particle masses and
`gas.concentration` mutate in place; there is no second gas return value.
Caller-supplied GPU vapor pressure is derived, non-authoritative state that the
step overwrites.

Direct temperature and pressure inputs may be positive-finite scalars,
same-device Warp arrays with shape `(n_boxes,)`, or hybrid scalar/Warp-array
inputs. Alternatively, pass `environment=WarpEnvironmentData` with both direct
inputs omitted; its `(n_boxes,)` values must be positive finite and on the same
device. Temperature and pressure remain environment-owned state, not `GasData`
fields.

The step executes four fixed equal substeps, uses caller-owned reusable scratch
and diagnostic buffers, P2 inventory finalization, and gas coupling for later
proposals. Its transfer is a whole-call total; optional caller-owned energy
output is not a third return value. This is not a high-level `Aerosol` or
`Runnable` API, automatic fallback, or hidden simulation-state transfer.
Callers synchronize before host observation or restoration; CUDA preflight
validation-flag readbacks may synchronize without transferring simulation
state. It also does not add adaptive stepping, new physics or container
support, BAT, or staggered/Gauss-Seidel support.
