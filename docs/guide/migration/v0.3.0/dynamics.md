---
title: Native dynamics migration
---

# Native dynamics migration

## Single-box CPU processes

Native runnables consume the current `Aerosol.particles`, `.gas` and
`.environment` objects. They preserve those identities during in-place process
execution. Their schema may store several boxes, but the supported CPU runnable
path is single-box; it does not silently process box zero of a multi-box input.

Run this excerpt from the selected native checkout. Its maintained constructors
are in `docs/examples/fundamentals/native_state_access.py`:

```python
from docs.examples.fundamentals.native_state_access import build_aerosol, build_condensation

aerosol = build_aerosol()
condensation = build_condensation(aerosol.gas)
particles, gas, environment = aerosol.particles, aerosol.gas, aerosol.environment
assert condensation.execute(aerosol, time_step=4e8, sub_steps=4) is aerosol
assert aerosol.particles is particles and aerosol.gas is gas
assert aerosol.environment is environment
```

The example explicitly maps water at gas lane 2 to particle material 0 and
organic at gas lane 1 to material 1, retaining inert gas at lane 0. It supplies
full gas-order molar masses and process-owned activity, surface and vapor-pressure
strategies. Do not infer chemistry from equal array widths or discard inert gas.
The runnable reads current environment and gas state during its substeps.

| Process | Native guidance |
|---|---|
| `MassCondensation` | Explicit expected full gas order and material mapping; finite inventories; process-owned thermodynamics |
| `Coagulation` | Match `distribution_type`; resolved native coagulation requires equal positive active counts; retained binned methods have composition limitations |
| `Dilution` | Exponential finite-step decay of particle populations and every gas lane; masses per particle remain unchanged |
| `WallLoss` | Match the population kind and geometry; resolved removal retires lost slots; gas is not a wall sink |
| `Nucleation` | Explicit source/commit configuration and mapping; uses held environment and current gas; fixed capacity and exhaustion rules apply |

See the [condensation guide](../../processes/condensation.md),
[coagulation guide](../../processes/coagulation.md), and
[nucleation guide](../../processes/nucleation.md). The binned coagulation
limitation is not a tolerance to relax: radius-varying composition can require a
resolved model. A composed `first | second` sequence executes each full cycle
with equal substep durations and reads the state left by preceding processes.

Direct array mutation is caller-owned. Process validation still applies on the
next invocation. Aggregate replacement validates shape/representation before
publication, but it does not rewrite a process's expected gas order or physical
configuration. Rebuild that configuration explicitly when its meaning changes.
Validation rejection is distinct from rollback after a process writer runs;
there is no blanket whole-sequence rollback guarantee.

## Direct Warp steps

Import steps from `particula.gpu.kernels`. Conversion helpers remain under
`particula.gpu`; maps, scratch and resident configuration remain at their concrete
modules. See the maintained
[direct quick start](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_direct_kernels_quick_start.py)
and [five-process sequence](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/direct/gpu_complete_process_sequence.py).

Explicitly convert native discrete/resolved data; radius PDFs cannot be
uploaded. Raw resolved concentration stays counts. Condensation couples gas
through counts/volume, and extensive energy weights transfers by counts.
Keep full ordered gas names externally. Prepare a condensation or nucleation
species binding against the exact transferred carriers, with explicit lane
pairs and full expected gas order. Replacing storage invalidates that binding;
normal dispatch does not silently recreate it.

`condensation_step_gpu` requires `thermodynamics=` and native `species_binding=`.
It returns the identical particle carrier and whole-call transfer accumulator;
gas mutates in place and optional energy is a caller-owned output, not a third
return. Four equal fixed substeps refresh derived vapor pressure and couple later
proposals to updated gas. This is the bounded direct method, not general CPU
strategy parity, adaptive stepping or a high-level runnable.

Temperature and pressure may be positive finite scalars or same-device `(B,)`
Warp arrays, including scalar/array hybrids. Alternatively supply
`environment=WarpEnvironmentData` and omit both direct inputs. Never mix these
forms. Synchronize explicitly before host observation/restoration. Validation
may synchronize flags internally; simulation-state transfers remain explicit.

## Resident and capture paths

The [resident multi-timestep example](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_multi_timestep.py)
prepares exact session/registry/guard ownership and process mappings, then runs
the maintained twelve-node composition. After explicit same-device restart it
prepares new mappings against fresh carriers. Its prescribed gas and wall loss
make it an open system: zero-ledger residuals report net inventory change,
not zero closed-system drift.

Uncaptured examples use Warp CPU when installed. The
[native capture example](https://github.com/uncscode/particula/blob/main/docs/examples/gpu/resident/gpu_resident_graph_capture.py)
requires CUDA and never substitutes CPU or Warp CPU. Optional CUDA absence is
not performance evidence. See [container foundations](../../data-model/index.md)
for transfers and [resident checkpoints](../../gpu/resident/checkpoints.md) for
canonical-byte continuation and lossy inspection boundaries.
