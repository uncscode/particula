# Scope

P1-P3 deliver CPU-only, concrete-module-only boundaries that compute bounded
empirical potential rates, finalize immutable inventory-limited source demand,
and atomically commit its final represented form. P2 remains read-only; P3
consumes P2 records and stages particle-slot/exhaustion work privately.

## In Scope

- A `NucleationStrategy` interface and activation-type `J=A[H2SO4]` and
  kinetic-type `J=K[H2SO4]^2` strategies.
- SI-normalized inputs and outputs: precursor mass concentration in `kg/m^3`,
  molar mass in `kg/mol`, temperature in K, and `J` in `#/m^3/s`.
- Configured closed validity ranges for precursor concentration, temperature,
  and optional saturation gate; out-of-domain evaluation fails closed.
- Fixed injection composition as molecules per event (or exactly equivalent
  species mass), formation-size metadata, and optional explicit survival factor.
- Frozen validated closed intervals, validity domains, composition, and
  formation metadata; strict scalar-only evaluation and overflow-safe
  mass-to-number conversion.
- Exact zero-rate paths and optional saturation-gate semantics, with isolated
  tests for equations, boundaries, ordering, invalid inputs, immutability, and
  the constrained P4 export boundary.
- `PotentialEventData`, `SourceDemandData`, and `SourceDiagnostics` in
  `particula/dynamics/nucleation/particle_source.py`, with fresh read-only
  `float64` payloads and a read-only `int32` limiting-species diagnostic.
- Shared per-box gas-inventory admission from survival-adjusted P1 rate and
  duration, deterministic lowest-index tie selection, `-1` no-limiting
  sentinel handling, and bounded vectorized ULP correction for rounding-only
  inventory overshoot.
- Read-only validation of all gas concentration lanes and participating molar
  masses, plus co-located source-record tests for success, zero, validation,
  ownership, nonaliasing, and no-mutation behavior.
- `commit_particle_source` and frozen commit configuration/final diagnostics in
  `particula/dynamics/nucleation/particle_source.py`, without package exports.
- Private `ParticleData.copy()` staging that consumes E6-F5 activation and
  E6-F6 resampling-first/scaling-fallback policy; caller particle and gas
  arrays are written only after all-box validation.
- Equal-weight final slot packaging, scaled pre-existing particle/gas state for
  representative-volume rows, and finite per-box/species conservation checks.
- P4 strict activation/kinetic builders, `NucleationSourceConfigBuilder`, and
  `NucleationFactory`; accepted units normalize into SI and mappings are staged
  atomically before replacing a builder configuration.
- Immutable `NucleationSourceConfig` metadata and constrained P4 exports from
  both dynamics namespaces. P2/P3 records and transaction helpers remain
  concrete-module-only.
- Public CPU-only, single-box `Nucleation` and immutable
  `NucleationCommitConfig` exports from `particula.dynamics`.
- Legacy `Aerosol` adaptation that retains `ParticleData` and partitioning
  `GasData` identity, validates one-box topology, and leaves gas-only facade
  state untouched.
- Equal sequential substeps that re-read current gas and perform one P2/P3
  transaction per positive-rate substep; P3 atomicity is per substep only.
- Test-only independent NumPy `float64` P2/P3 expected-value and snapshot
  coverage across multi-box/multi-species conservation, capacity policies,
  repeated current-state coupling, and rejection atomicity; P5 regressions
  remain single-box and integration tests directly exercise concrete P2/P3.

## Out of Scope

- Full Vehkamäki binary parameterization, classical-nucleation free-energy
  solver, ion-induced/heterogeneous nucleation, cluster dynamics, chemistry, or
  automatic coefficient calibration.
- Hidden survival/growth correction; Kerminen-Kulmala correction applies only
  through a caller-supplied documented factor.
- GPU kernels (E6-F8), integrated scheduling (E6-F9/Epic G), dynamic storage,
  hidden transfers, backend selection, graph capture, differentiability, or
  performance claims.
- Silent clipping of unsupported environmental inputs, partial multi-box
  commits, or silent loss of slot-exhausted source demand.
- Broad top-level exports or exports of P2/P3 records/helpers.
