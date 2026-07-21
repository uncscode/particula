# Infrastructure Reuse

- `particula/dynamics/particle_process.py:25-124` -- follow the
  strategy-backed `MassCondensation` `rate`/`execute` and substep pattern for
  the CPU `Nucleation` runnable.
- `particula/dynamics/condensation/condensation_strategies.py:239-364` -- reuse
  abstract strategy, typed NumPy, data-container, and gas-coupled validation
  conventions; do not reuse condensation physics as nucleation.
- `particula/gas/gas_data.py` and `particula/gas/species.py:42-185` -- use
  `GasData` as authoritative concentration/molar-mass state and retain facade
  compatibility only where current dynamics APIs require it.
- `particula/particles/particle_data.py` -- preserve array shapes, identities,
  density, charge, and volume except for explicit E6-F6 scaling.
- Planned `particula/particles/slot_management.py` from E6-F5 -- consume its
  predicates, ascending free indices, request shape, exact counts, and atomic
  activation rather than implementing a second slot model.
- Planned `particula/particles/exhaustion.py` from E6-F6 -- consume its
  resampling-first transaction, optional scaling fallback, admitted demand, and
  conservation diagnostics rather than truncating source demand.
- `particula/abc_builder.py`, `particula/abc_factory.py`, and existing
  condensation/wall-loss builders and factories -- follow construction and
  validation conventions.
- `particula/dynamics/__init__.py:37-64` -- expose only supported strategy,
  builder, factory, and runnable APIs.
- `docs/Theory/Technical/Dynamics/Nucleation_Equations.md:117-180` -- equations
  7-11 define empirical rates, injection semantics, coupled depletion, and
  fixed-slot source behavior.
- `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py` --
  migrate useful example concepts while identifying illustrative coefficients.
