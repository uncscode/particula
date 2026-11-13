# Appendix

## Review decisions: 2026-09-27

This record incorporates the maintainer's interactive review of PR #1603.
It supersedes conflicting first-draft alternatives in E9 and E9-M1–M6.
Decisions are planning requirements, not implementation or passing evidence.
Kyle Gorkowski (`Gorkowski`) is the final plan, scientific and release approver.
The six tracks remain serial; no release date or implementation completion is
implied. Research used the checkout at `561736c6f` and issue #1602.

### D1 — Shared distribution metadata and volume normalization

**Later maintainer clarification (2026-09-28, issue #1612):** The original D1
table and M1/M2/M3/M4 normalization examples below are historical planning
proposals where they conflict with the updated P1 handoff in the E9-M1
`appendix.md` ("Maintainer clarification for P1 handoff"). Discrete/PMF counts
per bin and continuous radius PDF `dn/dr` are already **per-volume** quantities;
both use fixed `volume == 1 m³` and must not be divided by V for bulk density.
Resolved slot weights are represented counts and require division by their
positive finite physical V for density. Only resolved density-to-storage
conversion multiplies by V. PDF bulk count and mass density use trapezoidal
integration on an explicitly supplied, positive, finite, strictly increasing
radius grid (at least two nodes), with mass times PDF integrated together and
no extrapolation. A valid PDF grid may have zero PDF and zero particle mass.
PMF/resolved bulk properties use weighted sums. Callers own
representation-consistent additions/changes. No legacy untagged compatibility
is required; builders must require explicit kind and validate the
representation-specific volume rule on creation. No global constructor
validation or broadened nucleation support is authorized here.
Direct construction and subsequent caller mutation use an explicit read-only
`ParticleData` representation-validation method: callers invoke it after
updates, new bulk helpers invoke it before interpretation, and process
admission using that interpretation must also call it. The validator rejects
invalid kind, volume, grid or schema without changing caller data; it cannot
intercept writable-array edits or silently repair their meaning. Builder
creation and this post-mutation validation are distinct from process-specific
physical checks. Revised P1 ledger/tests/scientific approval and P2
acceptance/ownership validation remain pending; this is not an implementation
or phase-completion claim.

Reuse a single shared vocabulary, proposed at
`particula/particles/distribution_types.py`:
`DISTRIBUTION_TYPES = ("discrete", "continuous_pdf", "particle_resolved")`.
Use the same typed values on `ParticleData.distribution_type` and process
configuration. Data declares storage; injected strategies select algorithms
and reject incompatible storage before mutation. Metadata is not a physics
strategy or a second authoritative array. No inference from shape, volume,
integer-looking weights or the process's preferred algorithm is permitted.
Builders must set the type explicitly; M1 must inventory legacy construction
before selecting any backward-transition default. A default cannot silently
reinterpret existing data.

| Type | Raw `concentration` storage | Physical concentration |
|---|---|---|
| `discrete` | Number represented in each bin, N_i | N_i / V, m^-3 |
| `particle_resolved` | Number represented by each slot, N_i; usually 1 for an ordinary resolved particle | N_i / V, m^-3 |
| `continuous_pdf` | dN/dr in the simulation volume, m^-1 | (dN/dr) / V, m^-4 |

The maintainer confirmed all distribution coordinates are radius-based.
The continuous coordinate is radius in metres, matching the existing
continuous coagulation calls. Mass/log-radius/log-diameter PDFs require an
explicit Jacobian conversion before entering this representation; a PDF sum
is not a particle count. Preserve existing quadrature and scientific intent;
freeze grid validity and quadrature in M1 and verify them in M3. Do not
silently change PDF algorithms into PMF algorithms. Counts may be fractional
population weights, but metadata does not grant an algorithm support for
unequal weights: selectors that assume unit weights must reject unsupported
weights or receive a separately reviewed conservative implementation.

For discrete/resolved populations, particle species mass is
`sum_i(N_i * m_i,s)` kg and its density is that sum divided by V. For PDFs,
replace the sum with `integral(m_s(r) * dN/dr, dr)`. Gas remains kg/m^3, so
gas species mass is `C_g,s * V`. `total_mass` remains per-particle mass.
All count-density conversions divide by V once; density-rate updates write
back `delta_N = V * delta_n` (and the corresponding PDF equation). Public rate
units must distinguish per-particle kg/s, population kg/m^3/s, number-density
rates and PDF-density rates. Do not label coagulation number rates kg/s.

Required fix matrix:

- M1 specifies metadata, read-only admission, number-density/population-mass
  helpers, PDF quadrature and independent expected values. It characterizes
  old behavior without declaring legacy agreement to be physical correctness.
- M2 native initializers receiving number density store `N = n * V`; PDF
  initializers store `dN/dr = V * dn/dr`. Direct count inputs do not multiply
  again. Retain explicit units and defaults; sampled resolved slots remain
  count weights rather than secretly rescaling them to a requested density.
- M3 condensation uses N/V (or PDF quadrature weights/V) for inventory limits
  and gas coupling. Coagulation density kernels receive normalized quantities
  and density increments are converted back to storage. Resolved collision
  volume factors are audited rather than divided twice. Both wall-loss paths
  preserve the storage basis and represented population moments.
- M4 dilution scales counts/PDF values and gas density at fixed V. Nucleation
  converts rate J [m^-3 s^-1] into admitted number `J * V * dt`; gas removal
  uses source mass/V. Physical volume evolution at fixed inventory holds
  particle counts fixed and rescales gas density. Representative-volume
  resampling is a different operation: scaling V and represented counts by
  the same factor preserves density; it must not be confused with expansion.

Independent checks use V=0.25, 1 and 4 m^3. For N=8 and m=2e-18 kg, number
density is 32, 8 and 2 m^-3, while particle mass is always 1.6e-17 kg.
For a per-particle gain of 1e-19 kg, gas-density loss is respectively
3.2e-18, 8e-19 and 2e-19 kg/m^3. Include multiple species, unequal supported
weights, zero/inactive slots, PDF integrals, rate/step agreement and rejection
without mutation. Equivalent physical populations at scaled N and V must
have equal intensive results where the algorithm supports those populations.

### D2 — Narrow CPU/GPU scope amendment

The maintainer explicitly approved correcting normalization on CPU and GPU.
The original blanket unchanged-schema/unchanged-GPU-semantics restriction is
amended only for distribution metadata and its required normalization,
conservation, transfer and checkpoint consequences. Array shapes, precision,
container names, explicit transfer APIs, device ownership and no-fallback /
no-automatic-rebind boundaries remain protected.

Transfer helpers preserve raw values and distribution metadata; they do not
hide a counts-to-density conversion. Warp carrier metadata, copies,
checkpoint versioning/restore and prepared/capture compatibility must preserve
or validate the interpretation. Legacy checkpoints lacking sufficient semantic
provenance must reject or use an explicit reviewed migration, never guess.
Changing distribution metadata invalidates incompatible prepared bindings.
GPU algorithms that only support resolved populations reject discrete/PDF
inputs explicitly; this work does not add GPU PDF execution.

Ownership stays serial: M1 freezes the cross-backend contract and metadata
admission; M3 owns direct/prepared condensation, coagulation and wall-loss
normalization with their scientific tests; M4 owns dilution, nucleation,
resampling/volume evolution, communication, resident composition and checkpoint
integration consequences. M5 updates examples and M6 validates the complete
post-removal contract. Split oversized phase work into reviewed serial changes;
do not hide a kernel-wide correction in a documentation-only phase. Installed
Warp CPU is required evidence; CUDA remains optional pass-or-clean-skip.

### D3 — Species alignment

Environment saturation-ratio lanes follow full GasData order. Particle species
may differ. Coupled processes own an explicit ordered gas-to-particle lane map;
nonpartitioning gas and particle-only material are retained. Use validated
integer lane pairs, rejecting duplicate source/target lanes and out-of-range
indices for ordinary one-to-one transfer; chemical stoichiometric mappings are
outside this migration. An omitted map must not silently compact a mask or
assume chemical identity from equal widths. Process configuration records its
expected ordered gas names so reordering can reject before mutation. Particle
lane chemistry remains caller-declared because those lanes have no names.
Names used for alignment must be nonempty and unique; do not silently rename
or sort legacy duplicates. Explicitly classify unmapped/nonparticipating lanes.
All-false masks are valid aggregate state, not a reason to discard gas.

Aggregate validation checks container structure, box compatibility, gas/name
metadata and gas/environment width, not particle/gas width equality or physics
configuration. Process admission additionally checks the mapping, distribution,
physical domains and single-box CPU limit. Shared read-only structural checks
belong in a concrete `particula/aerosol_validation.py`; process-specific checks
remain with processes. This module is proposed, not a delivered public export.

### D4 — Aggregate and environment API

Approved target: keyword-only
`Aerosol(particles=..., gas=..., environment=...)`, identity-returning
`particles`, `gas`, `environment` properties with validated setters, and
`replace_data(particles=..., gas=..., environment=...)` requiring all three
arguments. Validate the complete candidate triple before publication; retain
candidate identity, including same-object assignments. Never implement the
coordinated operation by chaining mutating setters. Rejection preserves old
and candidate data. Raw array fields remain writable; existing derived
properties calculate fresh results; `copy()` detaches arrays/metadata and
preserves distribution type. No redundant aggregate get_*/set_* method family.

`Nucleation` reads the current `aerosol.environment` during execution. Remove
the constructor's redundant environment argument at the final breaking API.
Replace the MassBasedMovingBin class test with explicit native distribution,
fixed-slot, species-map and physical admission. Do not infer newly supported
PDF/weighted behavior from removing a class check; M4 must retain the proven
topology and reject unsupported cases.

### D5 — Construction, transition and evidence policies

Retain useful direct mass/radius, speciated mass, PDF/PMF and sampled lognormal
capabilities through native builders and small native construction utilities.
Reuse ParticleDataBuilder/GasDataBuilder, explicit EnvironmentData and migrated
AerosolBuilder. Keep scientific distribution generation separate from facades;
retain native binning/conversion algorithms used by coagulation. Preserve
sample defaults and current random-state behavior unless a specific reviewed
change is approved; dependency injection is not permission to change RNG.

Prefer retaining existing legacy call paths while native paths are introduced.
If aggregate isolation is necessary, move the old aggregate to an explicitly
temporary concrete legacy module for unmigrated consumers; do not attach an
Atmosphere-emulating property to the new aggregate. No duplicated authoritative
arrays or permanent public shim. M2 must record exact consumers/imports,
aliasing and mutation behavior, M3/M4/M5 migration owners and M6 deletion proof.
The bridge implementation is a phase design deliverable, not an excuse to skip
full-suite gates or coerce native counts into an unannotated legacy convention.

Retain every currently supported example and scientific strategy/configuration
family, including latent-heat/staggered condensation, turbulent-DNS coagulation,
charged/neutral wall loss and their documented distribution combinations.
This is preservation of existing supported combinations, not a new Cartesian
product. M3 freezes the executable capability matrix; exclusions need explicit
maintainer approval. M5 expands its existing family inventory to every path,
pair and published snippet, including unaffected rows. No historical exceptions
are approved by this review; failing examples cannot be silently reclassified.
Full published workloads and bounded regression checks are both required.
Record dependencies, environment, runtime budget and numerical baselines per
example before its migration; missing resources remain a handoff blocker.

Use `docs/Features/Roadmap/data-native-v030-closeout.md` as the final evidence
index, with revision-specific PR/artifact links and Kyle's acceptance. Actual
revisions, test results, execution operators and environment inventories are
future gate evidence, not unanswered API questions or facts to fabricate now.
Retain schema-supported epic milestones and child phases. Plan approval does
not authorize release publication or mark any implementation phase complete.

### Source findings grounding the normalization decision

| Current source | Observation; not a passing scientific result |
|---|---|
| `particles/particle_data.py:58–70` | Documentation mixes resolved counts and binned density; no distribution metadata |
| `particles/representation.py:566–581` | Getter always divides raw storage by volume |
| `dynamics/condensation/condensation_strategies.py:995–1008,1101–1109` | rate uses raw weights; step uses weights/volume |
| `dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py:88–94,395–439,525–579,657–670` | Process has distribution_type; helpers read raw weights; continuous path integrates radius; native binned update adds returned rate directly |
| `particles/representation_builders.py:346–358,427–440,562–573,594–608` | PDF/PMF presets use unit-volume defaults; sampled resolved construction stores ones |
| `particles/particle_data_builder.py:10–19` | Native builder example labels concentration as 1/m^3, requiring migration |
| `particles/exhaustion.py:11–25` | Weighted extensive inventories and representative-volume scaling already expose count semantics |
| `dynamics/particle_process.py:209–234,247–305` | Nucleation retains separate environment, facade type check and equal-width assumptions |
| `gpu/conversion.py`; `gpu/kernels/condensation.py` | Transfers preserve raw arrays; direct condensation uses concentration-weighted gas deltas, requiring non-unit-volume audit |

Paths in the table are relative to `particula/`. These observations justify a
contract correction, not a claim that every named path has a demonstrated bug.
Implementation must retain reproductions and independent expected quantities.

## Authorities and technical references

Paths below are repository-relative. Line references describe the drafting
checkout and may move during migration.

| Reference | Relevance |
|---|---|
| Issue #1602, workflow `52e231a2` issue state | Agreed target contracts and six serial maintenance tracks |
| Prior `plan-scope-analyzer` message | Epic; feature/research tracks none; maintenance auto |
| `particula/aerosol.py:37–49` | Existing atmosphere/particle facade aggregate |
| `particula/particles/representation.py:566–581` | Concentration getter divides storage by volume |
| `particula/gas/gas_data.py:54–80` | CPU-owned ordered names, concentration units and partitioning mask |
| `particula/gas/environment_data.py:22–43` | Temperature, pressure and saturation-ratio authority |
| `particula/dynamics/condensation/condensation_strategies.py:294–364` | Existing native process-owned configuration |
| `particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py:49–111` | Native single-box admission and raw concentration helper |
| `particula/dynamics/dilution.py:339–406` | Separate legacy gas groups and normalized particle access |
| `particula/runnable.py:108–137` | Retained composition and ordered sequence |
| `particula/execution/adapters/condensation.py`, `coagulation.py` | CPU adapter migration review points |
| `particula/gpu/conversion.py` | Protected CPU↔Warp transfer boundary, not obsolete facade machinery |
| `docs/Examples/data_containers_and_gpu_foundations.py` | Existing data-first example pattern from issue |
| `docs/Features/particle-data-migration/` | Coexistence guidance to migrate, preserving labeled history |
| `docs/Features/Roadmap/data-oriented-gpu.md` | Migration-before-Epic-I roadmap context |
| `.opencode/guides/testing_guide.md:70–95,234–255` | Focused assertions/full runner and adjacent wall-loss requirements |
| `.opencode/guides/documentation_guide.md:19–35,77–93` | Strict docs and source-first notebook sync/execute |
| `.opencode/guides/linting_guide.md:15–26` | Ruff and mypy completion checks |
| `.opencode/plans/templates/epic/` | Thirteen canonical section structures |

## Rejected approaches

- Mechanical replacement of facade getters by raw fields: units and volume
  normalization differ and require explicit scientific regression evidence.
- Deletion first or parallel tracks: violates migration/validation ordering.
- Replacement compatibility facade: recreates duplicated ownership/behavior.
- Dropping legacy-fixture scientific tests: destroys enduring behavior checks.
- Broad removal of all conversion helpers: breaks supported CPU↔Warp APIs.
- New GPU integration or performance promises: beyond this maintenance scope.

## Drafting limitations

Researcher delegation was attempted but blocked by the runtime subagent-depth
limit. The draft uses direct repository reads, issue references and guides;
full builder/example inventories remain owning-track work. The exposed tools
provide `apply_patch` rather than a full-file `write` tool; section files were
replaced, not appended. Epic `add-phase` is unsupported; six Not Started
milestones are the schema-valid program representation. `schema` unexpectedly
generated four schema files despite being exposed through the read wrapper;
the orchestrator should inspect any generated-file diff before committing.

Canonical returned paths are the thirteen relative E9 section paths with no
traversal, all accessed beneath the designated worktree. No shell/lstat tool
was available for an independent symlink audit. No implementation tests,
linters or examples were executed by this planning-only task.
