# Architecture Design

## High-Level Design

The equation helpers remain pure NumPy functions. A CPU strategy owns or is
configured with the chamber dilution coefficient (directly or from volume and
flow) and exposes rate/step behavior over Particula containers. The `Dilution`
runnable delegates each internal substep to that strategy and updates the same
`Aerosol` instance.

```text
volume [m³] + inlet flow [m³/s]
        -> validate -> alpha = Q / V [s^-1]
                              |
                              v
                    DilutionStrategy.rate/step
                     /                    \
ParticleRepresentation                    GasSpecies
number concentration [1/m³]         mass concentration [kg/m³]
                     \                    /
                      preserve all metadata
                              |
                              v
             Dilution.execute(aerosol, dt, sub_steps)
```

P1 must select and document one canonical finite-step update derived from
`dc/dt = -alpha*c` (including whether integration is exact exponential or the
repository's bounded explicit-step convention). Both particle and gas paths
must use that same update, and E6-F2 must treat it as the parity oracle. No path
may encode dilution by changing particle mass.

## Data / API / Workflow Changes

- **Data model:** No schema or ownership changes. Only particle and gas
  concentration storage mutates. Particle mass, charge, density, distribution,
  volume and atmospheric temperature/pressure retain identity and value.
- **API surface:** Preserve `get_volume_dilution_coefficient()` and
  `get_dilution_rate()`. Add a named CPU dilution strategy and
  `particula.dynamics.Dilution` runnable with `rate(aerosol)` and
  `execute(aerosol, time_step, sub_steps=1)`. Final constructor naming and
  whether it accepts coefficient or volume/flow must be frozen in P1 and kept
  unambiguous in units.
- **Container workflow:** Preflight all inputs before writes, compute updated
  particle and gas concentrations, verify finite/nonnegative outputs, then
  commit both updates. This avoids half-mutated aerosol state.
- **Substeps:** Validate `sub_steps` as a positive integer, use
  `time_step / sub_steps`, and execute exactly that many strategy steps. Zero
  coefficient/flow or zero time returns the original aerosol without writes.
- **Exports:** Re-export supported symbols from `particula.dynamics`; preserve
  existing helper imports.
- **Downstream:** E6-F2 consumes the formula, validation ordering, scalar
  semantics, and deterministic fixtures. E6-F9 consumes the runnable in an
  integrated process example.

## Security & Compliance

There are no permissions, network, secret, or persistence changes. Robustness
requires rejecting malformed, negative, or nonfinite physical inputs before
mutation. Scientific documentation must state units and cite the dilution
equation. Tests must demonstrate that invalid calls cannot leave particle and
gas state partially updated.
