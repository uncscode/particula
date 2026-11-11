# Native construction capability and behavioral-test ledger

This ledger follows delivered M1 semantics and the approved M2 implementation
plan. It replaces historical proposals to multiply PMF/PDF values by volume.
Scientific strategies remain process-owned. Legacy tests remain operational in
the temporary path until their owning migration/removal phase.

| Retained capability | Native successor / owner | Enduring behavioral evidence |
|---|---|---|
| Aggregate assembly and reads | `Aerosol`, P1 | `aerosol_test.py`: identity, current reads, full gas, unequal species, read-only rejection; old rendering checks retained in `legacy_aerosol_test.py` |
| Individual/coordinated replacement | `Aerosol` setters and `replace_data`, P2 | `aerosol_replacement_test.py`: same/distinct identity, incompatible individual and compatible joint layouts, immutable rejected old/candidate snapshots |
| Aggregate fluent assembly | `AerosolBuilder`, P3 | `aerosol_builder_test.py`: required fields, repeated builds, current data, constructor-consistent rejection, full gas; historical strategy-count behavior retained only in `legacy_aerosol_builder_test.py` |
| Direct mass and explicit species mass | Existing `ParticleDataBuilder`, P3 inventory | `particle_data_builder_test.py`, plus native aggregate tests; explicit kind and storage units, no facade strategy inference |
| Direct radius and lognormal radius-bin PDF/PMF | `create_particle_data_from_radii`, `create_lognormal_binned_particle_data`, P4 | Native initialization tests migrate useful radius/mass, distribution shape/default/unit assertions; independent weighted sums and mass-times-PDF trapezoids, fixed V=1 |
| Direct resolved mass / unequal weights | `create_resolved_particle_data`, P5 | Native initialization tests: explicit species density/slot charge, preserved fractional counts or density-to-count conversion, V=.25/1/4 and independent per-species mass |
| Sampled resolved lognormal | `create_lognormal_resolved_particle_data`, P5 | Controlled sampled radii, unchanged sampler/defaults, unit counts, no implicit density rescaling |
| Gas and environment assembly | Existing `GasDataBuilder`, explicit `EnvironmentData` | Retain full gas/name/mask order and environment width; process-owned mapping cannot be certified by unnamed arrays |
| CPU/Warp conversion | Existing explicit helpers, retained | Untagged legacy transfer preserved; tagged upload rejects before runtime probe because metadata support is absent; no M2 GPU interpretation or checkpoint claim |

P4/P5 implementation/test nodes and literal validation are recorded in their
phase evidence as they land. All helper results own their arrays; native
aggregate construction retains existing objects. Existing data-builder ownership
is unchanged. Lossy first-row/column coercions stay legacy-only; native helpers
accept only unambiguous or redundant reductions. No constructor or initializer
infers multi-box CPU execution support.

Legacy old-node mapping: `test_mass_particle_representation_builder` maps to
native direct mass builder checks; `test_particle_radius_representation_builder`
and `test_limited_particle_radius_builder` map to P4 radius and preset checks;
`test_resolved_mass_particle_representation_builder` and singleton density/charge
checks map to P5 direct-resolved checks; `test_preset_resolved_mass_particle_builder`
maps to P5 controlled sampling/default checks. Legacy normalization tests that
assert discarded matrix values are historical behavior, not native expectations:
P5 explicitly tests rejection of differing values and acceptance of redundant
rows/columns. Facade strategy-count admission has no native successor because
aggregate species widths need not match; process admission owns mapping.
