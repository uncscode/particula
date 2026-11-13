---
title: Native container access and ownership
---

# Native container access and ownership

The native `Aerosol(particles=..., gas=..., environment=...)` aggregate
retains three supplied containers by identity.
The [construction guide](construction.md) covers initializers and builders;
[the executable access example](https://github.com/uncscode/particula/blob/main/docs/examples/fundamentals/native_state_access.py)
covers access, replacement and a subsequent mapped process.

## Raw fields and derived quantities

Let `B` be boxes, `N` slots, `P` particle materials and `G` full gas species.
Particle and gas widths may differ. Gas order includes inert species; matching
array widths alone never establishes chemical correspondence.

| Native field or property | Shape | Meaning and ownership |
|---|---|---|
| `aerosol.particles` / `.gas` / `.environment` | Container | Actual held object, not a copy |
| `particles.masses` | `(B,N,P)` | Writable kg per represented particle/material |
| `particles.concentration` | `(B,N)` | Writable counts, PMF density or radius-PDF density according to the explicit kind |
| `particles.charge` | `(B,N)` | Writable elementary-charge units |
| `particles.density` | `(P,)` | Material density, kg/m³ |
| `particles.volume` | `(B,)` | Authoritative represented volume, m³ |
| `particles.radii` | `(B,N)` | Fresh radii in m, derived from mass/material density |
| `particles.total_mass` | `(B,N)` | Fresh total kg per particle; not bulk inventory |
| `particles.effective_density` | `(B,N)` | Fresh retained mass-weighted material-density diagnostic |
| `particles.mass_fractions` | `(B,N,P)` | Fresh per-particle mass fractions |
| `particles.slot_concentration_density` | `(B,N)` | Fresh m⁻³ for discrete/resolved, m⁻⁴ for radius PDFs |
| `particles.number_density` | `(B,)` | Fresh integrated/summed number density, m⁻³ |
| `particles.species_mass_density` | `(B,P)` | Fresh particle mass density, kg/m³ |
| `particles.species_mass_inventory` | `(B,P)` | Fresh extensive particle mass, kg |
| `gas.name` | `G` names | Full ordered CPU-owned provenance |
| `gas.concentration` | `(B,G)` | Writable gas mass density, kg/m³ |
| `gas.molar_mass` / `.partitioning` | `(G,)` | kg/mol and Boolean gas flags |
| `environment.temperature` / `.pressure` | `(B,)` | Writable K and Pa |
| `environment.saturation_ratio` | `(B,G)` | Full gas-order dimensionless state |

Raw array access aliases storage. Derived particle properties compute detached
arrays; modifying a derived radius or number-density result does not update
masses or counts. `copy()` explicitly detaches a container. Population-derived
properties validate the current representation on each read; raw mutation
itself does not enforce process-specific physics.

For resolved data, `concentration` is **counts**, and per-slot number density is
`concentration / volume[:, None]`. For PMFs, sum densities. For radius PDFs,
integrate `dN/dr` and mass times `dN/dr` over the explicit radius grid; do not
sum PDF nodes. Discrete/PDF volume is exactly 1 m³. See
[distribution rules](construction.md#distribution-and-volume-rules).

## Executable access and replacement walkthrough

Run from the selected native checkout:

```bash
python docs/examples/fundamentals/native_state_access.py
pytest examples_tests/native_state_access_example_test.py -q --no-cov
```

The following excerpt uses that maintained example's native constructors:

```python
from docs.examples.fundamentals.native_state_access import build_aerosol, replacement_layout

aerosol = build_aerosol()
held = aerosol.particles
assert held is aerosol.particles
counts = held.concentration
assert counts is held.concentration

# Compatible individual replacement retains the supplied identity.
candidate = held.copy()
aerosol.particles = candidate
assert aerosol.particles is candidate

# Adding a gas requires a matching environment width. Publish both together.
particles, gas, environment = replacement_layout(aerosol)
aerosol.replace_data(particles=particles, gas=gas, environment=environment)
assert aerosol.gas is gas and aerosol.environment is environment
```

All three arguments to `replace_data` are required. Each setter and coordinated
replacement validates the whole candidate triple before publishing references.
A rejection preserves old and candidate state. Same-object assignments also
revalidate. This is validation atomicity, not thread safety or rollback of an
already launched process.

Replacement selects new state; it does not conserve, rescale or merge the old
inventory. Shape compatibility does not validate an unchanged process's gas
order. Rebuild process-owned mappings when the intended chemistry/order changes.
Direct mutation can invalidate physical values even when shapes stay unchanged;
the next process call must reject invalid current inputs before its documented
write boundary. Prepared GPU/resident resources still refer to their original
carriers and require explicit new preparation after replacement or restart.

## Explicit Warp boundary

Native discrete and resolved particle transfers retain population metadata and
raw concentration values. Radius-PDF upload is unsupported. Preserve the full
ordered gas names externally; `WarpGasData` has no names. A gas restore without
names produces placeholders, and always drops GPU-only vapor-pressure storage.
Native condensation/nucleation require explicit process species bindings for the
exact transferred carriers. See [dynamics migration](../migration/v0.3.0/dynamics.md) and
[container foundations](index.md).
