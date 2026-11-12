# E9-M6 native API migration closeout

**Status: validation in progress; not approved for release.** This record
tracks the development-tree removal for [E9-M6](https://github.com/uncscode/particula/issues/1607).
It does not announce a released wheel, change the package version, or qualify
GPU performance. The maintainer owns final scientific and release approval.

## Removed APIs and replacements

| Removed boundary | Supported replacement |
| --- | --- |
| `ParticleRepresentation` and obsolete representation builders/factory | `ParticleData`, native initializers and `ParticleDataBuilder` |
| `GasSpecies` and obsolete species builders/factory | Full-order `GasData` and `GasDataBuilder` |
| `Atmosphere`, `AtmosphereBuilder` and legacy aerosol aggregate | Flat `Aerosol` holding particles, gas and `EnvironmentData` |
| Facade conversion bridges and legacy process dispatch | Native construction, validated container replacement and process-owned physics/maps |
| Untagged GPU execution and population transfer | Explicit supported population metadata and exact-carrier bindings |
| Untagged schema-v5 resident checkpoint admission | Tagged v5 canonical payloads with explicit independent gas width; ambiguous records reject before setup/upload |

Native initialization, scientific property functions, distribution algorithms,
explicit CPU-Warp transfers and supported runnable composition remain available.
Historical names may occur in migration comparisons and negative import tests;
they are not supported executable APIs.

## Preserved scientific and ownership boundaries

Resolved concentration is represented count; discrete concentration is number
density; continuous PDF is dN/dr. Use represented volume when comparing density
and extensive inventory. GPU PDF populations remain unsupported. Particle
material width and full gas width are independent; preserve ordered names,
nonpartitioning lanes and explicit process maps.

Native `Aerosol` holds its three containers by identity. Validated individual
or coordinated replacement does not rebind prepared GPU state. Raw array edits
remain caller-owned and require revalidation. CPU process execution remains
single-box; batched storage does not imply general CPU multi-box support.

Transfers, synchronization and device selection are explicit. Resident restart
retains schema v5, canonical device bytes and RNG continuation on an exactly
compatible device. CPU inspection carriers remain lossy and cannot replace
canonical continuation state. No inferred population tags, hidden fallback,
automatic restart or cross-device replay is introduced.

M5's fixed-composition binned transport, cloud dry-phase oscillation and soot
property limitations remain. No numerical oracle, conservation tolerance or
coverage threshold is relaxed to make the migration pass.

## Validation evidence

Evidence is revision-specific. Focused checks are assertion evidence; only the
untargeted repository runner supplies full-package coverage evidence.

| Gate | Current evidence |
| --- | --- |
| GPU/execution boundary suite | 4,182 passed, 679 skipped at `72f179eeb` boundary content; no CUDA timing claim |
| Native benchmark consumers | 95 helper tests passed; real Warp CPU uptake/evaporation and material conservation checks; committed in `48771293c` |
| Untargeted normal-coverage runner | Initial run: 8,331 passed, 2 failed, 684 skipped; 92% coverage against normal 80% threshold. Two untagged thermodynamics fixtures corrected; full rerun pending |
| Corrected thermodynamics integration | 2 passed, 1 optional CUDA skip; original test bodies and tolerances retained |
| Installed-wheel package suite | 4,559 passed, 25 skipped, 820 deselected; imported distribution provenance checked outside checkout, including removed-module and native aggregate tests |
| Separate installed-wheel examples | 20 passed; native dilution, nucleation and latent-heat examples |
| Both wall-loss suites | Public suite 16 passed; separately collected adjacent suite 43 passed |
| Infrastructure | 132 passed, including publication-routing regression |
| Plugin distribution consistency | Six skills and host packages validated |
| Ruff and mypy | Scoped Ruff/format passed; package mypy passed 535 files at benchmark revision; final changed-file checks pending |
| Standard and full strict documentation | Passed at `2f90d6c17` routing content: 1.56s and 49.06s, zero warning/error lines; rerun required after final documentation/output edits |
| Source example suite | Earlier 225 passed, 2 skipped; final rerun pending |
| Full supported source and notebook workloads | 59 source entrypoints and 46 notebook pairs being reexecuted; final ledger and plot/output review pending |
| Final compatibility inventory and PR | Pending |

The native benchmark workload identifiers distinguish them from historical
untagged cases. CPU interpolated-grid and GPU direct-pair coagulation samplers
are distinct algorithms; timing ratios do not establish statistical or
trajectory parity. No new benchmark timings are published by this migration.

The detailed commands, assertion-retention ledgers and intermediate failures
are recorded in `.opencode/plans/sections/maintenance/E9-M6/implementation.md`
and adjacent plan files. Required failures or missing evidence leave readiness
incomplete. Optional CUDA skips never stand in for native-CUDA capture evidence.
