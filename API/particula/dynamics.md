# particula.dynamics

*module*

Provide public dynamics strategies, construction APIs, and runnables.

Re-exports wall loss, dilution, condensation, and coagulation utilities,
strategies, builders, factories, and runnable processes for access through
:mod:`particula.dynamics`. Condensation exports include mass-transfer
helpers such as the thermal resistance factor. Includes neutral and charged
wall loss for spherical and rectangular chambers. Charged wall loss adds
Image-charge enhancement even when wall potential is zero and optional
electric-field drift. Also provides ``MassCondensation``, ``Coagulation``,
``WallLoss``, ``Dilution``, and ``Nucleation`` runnables. ``DilutionStrategy``
and
``Dilution`` are the supported dilution construction APIs; low-level aerosol
mutation and finite-step helpers remain available only from the concrete
``particula.dynamics.dilution`` module.

Nucleation exports provide the bounded public construction surface for immutable
activation and kinetic potential-rate strategies. ``NucleationSourceConfig``
selects one of those strategies and a precursor index. The CPU-only single-box
``Nucleation`` runnable and its immutable ``NucleationCommitConfig`` provide
the runnable execution process boundary. Nucleation uses equal sequential
substeps, preserves
the aerosol and backing-container identities, and provides atomicity per
attempted substep rather than for the whole call. Demand planning/source
transaction ``particle_source``
helpers remain concrete-only.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/__init__.py#L1-L1)

**Modules**

- `particula.dynamics.coagulation`
- `particula.dynamics.condensation`
- `particula.dynamics.dilution`
- `particula.dynamics.nucleation`
- `particula.dynamics.particle_process`
- `particula.dynamics.properties`
- `particula.dynamics.wall_loss`
