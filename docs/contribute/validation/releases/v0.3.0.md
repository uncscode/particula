# Native API migration validation and release gates

**Status: implementation validated; maintainer review and release approval pending.** This record
tracks the development-tree removal for [v0.3.0](https://github.com/uncscode/particula/issues/1607).
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

V0.3.0's fixed-composition binned transport, cloud dry-phase oscillation and soot
property limitations remain. No numerical oracle, conservation tolerance or
coverage threshold is relaxed to make the migration pass.

## Validation evidence

Evidence is revision-specific. Focused checks are assertion evidence; only the
untargeted repository runner supplies full-package coverage evidence.

| Gate | Current evidence |
| --- | --- |
| GPU/execution boundary suite | 4,182 passed, 679 skipped at `72f179eeb` boundary content; no CUDA timing claim |
| Native benchmark consumers | 95 helper tests passed; real Warp CPU uptake/evaporation and material conservation checks; committed in `48771293c` |
| Untargeted normal-coverage runner | Final production cleanup run: 8,333 passed, 684 skipped; 92% coverage against unchanged normal 80% threshold. Earlier two untagged thermodynamics fixtures corrected with original assertions retained |
| Corrected thermodynamics integration | 2 passed, 1 optional CUDA skip; original test bodies and tolerances retained |
| Installed-wheel package suite | Final production cleanup: 4,559 passed, 25 skipped, 820 deselected in 41.74s; isolated Warp cache and installed distribution provenance checked outside checkout, including removed-module and native aggregate tests |
| Separate installed-wheel examples | 20 passed; native dilution, nucleation and latent-heat examples |
| Both wall-loss suites | Public suite 16 passed; separately collected adjacent suite 43 passed |
| Infrastructure | 132 passed, including publication-routing regression |
| Plugin distribution consistency | Six skills and host packages validated |
| Ruff and mypy | Final production lint runner passed, including Ruff and mypy over 535 files |
| Standard and full strict documentation | Both passed with final notebook outputs: standard 1.46s and full publication 47.69s, zero warning/error lines. Supported API pages remain included; no warning suppression. |
| Source example suite | 225 passed, 2 skipped after final dilution import guard and simulation prose cleanup |
| Full supported source and notebook workloads | 58 source executions and all 46 notebooks passed; 98 PNG plots reviewed, code pairs matched and image counts preserved. One native-CUDA-only source skipped cleanly without fallback. Full workload and output hashes recorded in the execution/review ledgers |
| Final compatibility inventory | Reconciled after regenerated outputs. Only intentional removal-safe test guards retain removed production symbol names; 1,606 native successor references resolve. |

The native benchmark workload identifiers distinguish them from historical
untagged cases. CPU interpolated-grid and GPU direct-pair coagulation samplers
are distinct algorithms; timing ratios do not establish statistical or
trajectory parity. No new benchmark timings are published by this migration.

The detailed commands, assertion-retention ledgers and intermediate failures
are retained with the [migration pull request](https://github.com/uncscode/particula/pull/1635)
and its implementation records. Required failures or missing evidence leave readiness
incomplete. Optional CUDA skips never stand in for native-CUDA capture evidence.
