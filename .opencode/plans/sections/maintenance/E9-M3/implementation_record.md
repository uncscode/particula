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
| Condensation `_make_data_inputs` via from_representation/from_species | Facade-free native fixtures with independent physics assertions; retain legacy-only branches separately | M3 P1/P2, pending |
| Coagulation native helper/ABC and mechanism fixtures | Explicit metadata, physical rate units, weighted species/charge inventory | M3 P3/P4, pending |
| Both wall-loss strategy suites | Native fixtures preserving geometry, helpers, RNG and sink assertions | M3 P5, pending |
| `_legacy_aerosol`, legacy builder, facade scientific dispatch | Retain for unmigrated runnables/adapters/examples; no duplicate authoritative arrays | consumers M4/M5, removal M6 |
| Untagged CPU/Warp conversions and legacy prepared consumers | Retain explicit old path; distinguish from native tagged evidence | native admission M3; resident M4; final removal review M6 |
| CPU/Warp transfer APIs | Preserve explicit transfer, values, shapes and ownership | retained API, not an M6 deletion target |

Expand old-node to new-node mapping as tests migrate. No enduring assertion
may disappear without an equivalent native check or explicit facade-only
classification for M6. Final evidence records revision, literal commands,
results, skips and reviewed M4 handoff; no release or merge authorization is
inferred from implementation completion.
