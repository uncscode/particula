# M3 implementation contract and evidence

## Entry and delivery

2026-11-11: Kyle confirmed M2 completion and authorized the eleven-commit,
single-PR plan in the attached goal objective. Entry baseline is merged M2
`fd114270a3e847a6b7b51a2c2ef2f7e5c27ecb4b` (PR #1630).
Branch: `codex/e9-m3-scientific-processes`; issue #1605.
This record supersedes contradictory September M3 planning wording. All
implementation and validation below remain pending unless explicitly recorded.

## Scientific contract

- Discrete PMF stores bin number density; radius PDF stores dn/dr. Both require
  volume exactly 1 m3. Do not divide either density by volume. Integrate PDF
  products on the explicit positive increasing radius grid with trapezoids.
- Resolved slots store counts; density is count/physical volume. Test volumes
  0.25, 1 and 4 m3 and unequal counts where the algorithm supports them.
- Per-particle species mass remains kg. Population density and inventory are
  separate quantities. Gas concentration remains kg/m3 in full gas order.
- Condensation processes own explicit ordered gas/particle lane pairs and
  expected gas names. Transport/vapor parameters follow full gas order;
  activity and surface composition follow full particle order. Unmapped,
  skipped and nonpartitioning lanes retain their state. No inferred identity
  map, mask compaction, duplicate mappings or implicit chemical stoichiometry.
- CPU processes validate representation and physical state again after caller
  mutation and require one box. State arrays stay caller-owned. Rejections
  precede writes; no new post-device-launch rollback promise.

## Capability matrix and owning tests

| Family | Existing algorithms to preserve | Native acceptance / rejection | Owning evidence |
|---|---|---|---|
| CPU condensation | Isothermal, staggered half/random/batch, latent heat; uptake, evaporation, reservoir | PMF/PDF at V=1, resolved positive V; weighted inventories and mapped species | condensation strategies, mass-transfer, staggered conservation/stability and new native tests |
| CPU coagulation rates | Brownian, charged, sedimentation, turbulent shear, turbulent DNS, combined | Discrete/PDF density rates; explicit matching representation; physical kernel inputs | all seven strategy test modules and independent rate fixtures |
| CPU coagulation steps | Existing discrete/PDF update and resolved binned/direct selector | Audit resolved selector weights before admission; reject unsupported unequal weights; preserve per-species mass | ABC and particle_resolved_step tests |
| CPU wall loss | Spherical/rectangular neutral/charged; image charge and field; deterministic binned and stochastic resolved | PMF/PDF at V=1, resolved positive V; preserve survivor and removed inventories | BOTH dynamics/tests and dynamics/wall_loss/tests suites, separately |
| Direct/prepared GPU condensation | Existing fixed-four-substep scheme and latent correction | Explicit count basis, species mapping, non-unit V; no GPU PDF algorithm | condensation, graph capture, direct/prepared integration tests |
| Direct/prepared GPU coagulation | Existing mechanism-specific kernels/selectors | Explicit supported count/weight matrix, volume correction; no inferred mode expansion | coagulation, validation, stochastic and prepared tests |
| Direct/prepared GPU wall loss | Existing resolved removal and RNG contracts | Explicit resolved metadata, count-preserving sink accounting | wall_loss, parity and prepared tests |

Algorithm-specific support rows must be backed by executable assertions before
the owning phase closes; this table is the implementation target, not passing
evidence or approval to exclude an existing supported combination.

## GPU metadata and downstream boundary

M3 must preserve distribution provenance at explicit transfer and direct/
prepared admission. Keep numeric array shapes/precision unchanged and retain
legacy untagged consumers as a separately tested temporary path. Introduce a
small explicit distribution tag on the Warp carrier, with process-owned
mapping/name provenance at host preparation and fixed device lane maps. Do
not silently assign a representation to unknown legacy input. PDF execution
must reject. Any newly admitted tagged state must round-trip its supported
metadata without density conversion.

Prepared calls freeze semantic configuration and device references; changed
metadata/mapping requires re-preparation. Normal enqueue must not upload,
allocate, read back, synchronize, resolve mutable configuration or rebind.
M4 owns resident/checkpoint composition and schema migration. Until then,
resident setup/checkpoint paths must reject newly tagged state they cannot
preserve, rather than discard metadata. Minimal rejection guards protecting
that boundary belong with M3 metadata changes; no M4 functionality is claimed.

### Concrete P1 metadata design

- `WarpParticleData.distribution_kind` is one `wp.int32` scalar: 0 unknown
  legacy, 1 discrete bin density, 2 unsupported radius PDF, 3 resolved counts.
  The existing five numerical arrays keep their shapes, precision and values.
  Explicit uploads validate native CPU representation; supported tags survive
  restore. Invalid/unsupported restore tags reject before synchronization.
- Concrete-only `prepare_condensation_species_map` takes exact particle/gas
  carriers, the caller-retained actual full gas names, an immutable CPU
  `CondensationSpeciesMap`, and optional full-gas skip indices. Names must match
  the process expectation. It uploads two newly owned int32 configuration arrays
  of lengths particle-species and gas-species respectively, with -1 for absent
  links. This explicit configuration upload does not transfer simulation state.
- `CondensationSpeciesBinding` pins exact carriers, primary array identities,
  layouts/devices, population code and full gas order. Validation rejects
  incompatible reuse without allocation, launch, synchronization or readback.
  Callers must keep these arrays and map contents alive/unmodified while bound;
  new names, maps, storage or semantics require fresh preparation. Partitioning
  remains dynamic full-gas device state, not a compacted or frozen host mask.
- These host preparation helpers are concrete-only, not new package exports.
  P2 integrates the binding into the existing fixed-four-substep implementation;
  it must not add per-enqueue host resolution or transfers.

| GPU boundary | Executable boundary evidence | Numerical acceptance owner |
|---|---|---|
| Tagged discrete/resolved transfer | population_metadata_test: raw round trips, nonunit resolved inventory, explicit legacy retention | transfer complete |
| Tagged PDF / unknown code | particle_population_test and population_metadata_test: rejection before Warp probe/readback | no PDF execution |
| Species maps / independent widths | population_metadata_test: full gas order, skipped/unmapped lanes, stale binding and no-device-work checks | condensation P2 |
| Condensation direct/prepared | condensation_native_test: explicit binding, mapped four-substep science, N/V gas coupling and extensive energy | P2 verified discrete/resolved checkpoint |
| Coagulation direct/prepared | coagulation_native_test: all eleven masks, equal counts, independent species/charge inventory and stochastic expectations; population_boundary_test rejects PMF/PDF | P4 GPU checkpoint complete; CPU moments have separate p4_moments_validation.md evidence |
| Wall loss direct/selected/prepared | population_boundary_test: interim rejection before payload access | P5 admits resolved removal with sink accounting |
| Dilution/nucleation/exhaustion and resident setup/revalidation | population_boundary_test and gpu_session_test: reject newly tagged consumers before mutation or metadata loss | M4 |

Interim science rejection rows are replaced, not counted as completed scientific
support, by their owning M3 numerical commits. Legacy untagged regressions remain
separate. A population tag alone never expands an algorithm's support matrix.

The inspected GPU coagulation selector currently executes masks 1, 2, 4, 8,
3, 5, 6, 9, 10, 12 and 15 (Brownian, charged hard sphere, SP2016 sedimentation,
ST1956 turbulent shear, their approved pairs, and all four). Triple masks
7/11/13/14 and GPU DNS remain unsupported. M3 preserves these mechanism gates.
`apply_coagulation_kernel` retains recipient concentration while adding donor
mass and charge, then clears the donor. Therefore native active weights must
be equal within each box for that merge to conserve weighted species mass and
charge. P4 tests verify positive equal fractional/multiple weights, reject
unequal weights, and include common weight in collision scheduling. The legacy
selector retains its sedimentation-only weight factor; the native selector
applies current common count to all eleven supported masks. See
`p4_gpu_validation.md` for the independent scientific evidence.

## Serial commits and gates

1. Entry contracts, matrix, ledgers and M2 status reconciliation.
2. P1 CPU condensation admission/configuration and builder/factory coverage.
3. P1 GPU metadata/mapping admission and boundary regressions.
4. P2 CPU isothermal/staggered/latent stepping and independent inventories.
5. P2 direct/prepared GPU condensation correction and regressions.
6. P3 CPU coagulation mechanisms/rates and scientific fixtures.
7. P4 CPU stepping, temporary binning and conservative weight admission.
8. P4 direct/prepared GPU coagulation correction and regressions.
9. P5 native neutral/charged CPU wall loss and both explicit suites.
10. P5 direct/prepared GPU wall-loss correction and regressions.
11. P6 native aggregate integration, developer guide and final evidence.

Each implementation commit includes its adjacent tests. Complete the owning
phase checks before the next phase; repair findings in follow-up commits as
needed without combining independent PRs. Final checks use the repository
.venv: focused science with --no-cov, both wall-loss suites separately,
GPU direct/prepared and transfer/export checks, integration, untargeted
.opencode/tools/run_pytest.py, lint/type, strict docs, examples_tests and
scripts/tests. Preserve the configured coverage and warning policies.
Warp CPU is required when installed; CUDA results/skips are explicit.

## Assertion and compatibility ledger

| Existing coverage/seam | Required successor / retention | Owner / state |
|---|---|---|
| Condensation `_make_data_inputs` via from_representation/from_species | species_mapping_test plus native_step/native_staggered/native_step_admission and GPU condensation_native_test establish facade-free configuration, rate, stepping, rejection and conservation assertions; legacy fixtures retained separately | M3 P1/P2 checkpoint complete; final ledger review P6 |
| Coagulation native helper/ABC and mechanism fixtures | native_rates_test supplies explicit metadata, physical kernel inputs and density/PDF measures; native_coagulation_step_test adds weighted resolved merges, binning and admission; native_coagulation_moments_test adds independent finite-step species/charge transport and refinement through the retained gain quadrature | P3 committed b41c908d1; P4 CPU/GPU checkpoints and moment correction; see p4_validation.md, p4_gpu_validation.md and p4_moments_validation.md |
| Both wall-loss strategy suites | Both original suites now use explicit native fixtures, retaining every original test function; native_wall_loss_test adds independent density/resolved sinks, weighted stochastic evidence and admission | M3 P5 CPU checkpoint; see p5_cpu_validation.md; GPU P5 pending |
| `_legacy_aerosol`, legacy builder, facade scientific dispatch | Retain for unmigrated runnables/adapters/examples; no duplicate authoritative arrays | consumers M4/M5, removal M6 |
| Untagged CPU/Warp conversions and legacy prepared consumers | Retain explicit old path; distinguish from native tagged evidence | native admission M3; resident M4; final removal review M6 |
| CPU/Warp transfer APIs | Preserve explicit transfer, values, shapes and ownership | retained API, not an M6 deletion target |

Expand old-node to new-node mapping as tests migrate. No enduring assertion
may disappear without an equivalent native check or explicit facade-only
classification for M6. Final evidence records revision, literal commands,
results, skips and reviewed M4 handoff; no release or merge authorization is
inferred from implementation completion.
