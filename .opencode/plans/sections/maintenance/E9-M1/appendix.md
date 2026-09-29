# Appendix

## Evidence and references

| Source | Drafting evidence / use |
|---|---|
| Issue #1602 via workflow issue state | Agreed T1 scope and replacement invariants |
| E9 `appendix` D1–D5, `implementation_strategy`, `dependency_map` | Approved review decisions, normalization correction and strict serial implementation gates |
| `particula/particles/particle_data.py:58–80,166–232` | Distribution-dependent concentration documentation, existing derived properties and deep copy |
| `particula/particles/representation.py:566–592` | Getter divides raw concentration by volume before summation |
| `particula/gas/gas_data.py:54–80,82–147` | Ordered gas metadata, kg/m^3, constructor coercion and copy |
| `particula/gas/environment_data.py:22–43,45–83,94–129` | Environment authority and constructor copies; not a read-only validator |
| `particula/gpu/conversion.py` | Protected transfer boundary; no deletion authorization |
| `.opencode/guides/testing_guide.md`, `pyproject.toml` | Focused/full-suite split, markers, warning policy and wall-loss exclusion |
| `.opencode/plans/templates/maintenance/` | Canonical twelve-section format; embedded M23 examples used as formatting prior art |

Only E9 maintenance shells were found under the canonical maintenance section
directory during discovery; those are not completed implementation examples.

## Implementation ledgers to complete

- Contract row: field/helper, actual consumer, distribution convention, raw
  units, physical units, conversion equation, shape, order, mutation and test.
- Species row: gas index/name/flag, particle lane or explicit absence,
  environment lane, process-parameter index and mismatch rejection rule.
- Helper row: existing/reused/added, module/import, caller need, copy/view
  behavior, invalid-input behavior and adjacent test node.
- Replacement case: operation, held/candidate triple, admission result,
  identity guarantees and owning M2 acceptance test.
- Validation row: revision/date, command, exit status, counts/availability,
  report location and reviewer. At drafting time, all implementation outcomes
  were pending; P1 characterization evidence is recorded below.

## Drafting limitations

At drafting time, this run populated E9-M1 only. No production code or other
plans were edited.
The available editor is `apply_patch`, so full-file replacements are used
instead of the unavailable `write` tool. Returned canonical relative paths
were checked for the E9-M1 prefix and traversal, and files are accessed under
the exact worktree. No independent lstat/symlink audit tool is available.
No schema command was used. At drafting time, implementation tests, lint and
docs execution were future gates, not results of that planning-only run.

## P1 characterization ledger (draft; scientific review pending)

**Revision note (2026-09-28):** The scientific proposals in this ledger now
reflect the maintainer clarification below. Dated P1 characterization and P2
gate rechecks later in this document remain historical observations, not
validation of the revised contract.

This is a proposed physical contract, **not** an assertion that existing
processes implement it. P1 changes tests and this ledger only. In particular,
`ParticleData` does not store a distribution kind or a PDF radius grid.

| Owner/field | Raw storage, shape and units | Proposed physical use, order and ownership |
|---|---|---|
| `ParticleData.masses` | `(B,N,Sp)` kg per representative particle | Species order is the particle lane order; `total_mass` is `(B,N)` kg **per particle**, not population inventory. Mutable CPU arrays; `copy()` detaches. |
| `ParticleData.concentration`, discrete / PMF | `(B,N)` number per bin per m³, possibly fractional | Number density `sum_i c_i` [m⁻³]; species mass density `sum_i c_i m_i,s` [kg/m³]. No division by V or unit-weight-only assumption. |
| Same field, particle resolved | `(B,N)` represented counts N, normally unity for active slots | Number density `sum_i N_i/V` [m⁻³]; species mass density `sum_i N_i m_i,s/V` [kg/m³]. Fractional weights require process-specific support. Zero/inactive populations contribute zero mass. |
| Same field, continuous radius PDF | `(B,N)` `dn/dr` [m⁻⁴] on a declared radius grid [m] | Number density `∫(dn/dr)dr` [m⁻³]; species mass density `∫m_s(r)(dn/dr)dr` [kg/m³]. No slot sum, extra V division, or extrapolation. |
| `charge`, `density`, `volume` | `(B,N)` elementary-charge counts, `(Sp,)` kg/m³, `(B,)` m³ | Charge follows representative slot; density shares particle species order. Builder fixes V=1 m³ for PMF/PDF and requires positive finite V for resolved data. Explicit read-only post-mutation validation protects later interpretation; existing shape-only constructor does not check the physical domain. |
| `GasData` | ordered `name` `(Sg,)`, molar_mass `(Sg,)` kg/mol, partitioning `(Sg,)` bool, concentration `(B,Sg)` kg/m³ | Full gas order retained even for false mask lanes; gas extensive mass per species `C_g,s V` [kg]. Mutable arrays and names; `copy()` detaches. Duplicate names are currently tolerated. |
| `EnvironmentData` | temperature `(B,)` K, pressure `(B,)` Pa, saturation_ratio `(B,Sg)` dimensionless | Ratio lane order must follow **all** gas names, including nonpartitioning lanes; environment has no names or cross-container check today. `copy()` detaches. |

Only a particle-resolved physical density increment Δn [m⁻³] writes
ΔN=V Δn; PMF/PDF inputs and increments already have per-volume units and
must not be multiplied by V. Do not normalize an already-density input or
divide gas concentration again. For resolved N=8 at V=0.25/1/4 m³, number
densities are 32/8/2 m⁻³; per-particle species masses
`[0.5e-18,1.5e-18]` give densities `[16e-18,48e-18]`,
`[4e-18,12e-18]`, `[1e-18,3e-18]` kg/m³, respectively. Extensive
species mass remains `[4e-18,12e-18]` kg. A PMF bin value 8 at fixed
V=1 m³ instead represents 8 m⁻³ without normalization; fractional bins
contribute their own density weight × per-particle species mass.

**Clarified PDF grid rule (scientific choice recorded; implementation pending):**
radius in metres, finite, positive, strictly increasing, at least two nodes,
node count matching the PDF width; closed sampled interval with no
extrapolation. Use interval-wise trapezoids (`np.trapezoid(y, x=radius)`);
multiply species mass by PDF *before* quadrature. Neither diameter nor
log-radius may replace radius without the Jacobian. For
`r=[1e-9,2e-9,4e-9]` m and `dn/dr=[1e9,2e9,1e9]` m⁻⁴,
interval count densities are 1.5 and 3, total 4.5 m⁻³ at fixed V=1 m³;
constant per-particle `m_s=2e-18` kg gives 9e-18 kg/m³. A varying
per-species mass fixture must independently integrate endpoint products.
`sum(dn/dr)=4e9` is dimensionally wrong. Zero PDF on a valid grid gives
zero count/mass density; no PDF metadata or grid integration exists in P1.

### Proposed helper and provenance decisions (not implemented)

| Future seam / consumer | Direction, reuse and missing rule | Owner and tests |
|---|---|---|
| Distribution vocabulary and explicit metadata; `particles/representation_builders.py`, `particle_data.py:235-359` | Future constructor/builder and facade conversion must receive an explicit source distribution-kind tag and preserve it through native copy and explicit Warp conversion. `RadiiBasedMovingBin` supports both PDF and PMF: strategy class, shape and weights cannot distinguish them. Reject absent or ambiguous provenance instead of guessing a default or accepting a caller's unverified preference. Current conversions remain unchanged in P1. | M2 construction/conversion gate and P3/P4 access; test PDF vs PMF from the same strategy, missing provenance rejection and supported round-trips. |
| Resident checkpoint/restore; `execution/checkpoint.py` | P1 neither persists distribution metadata nor claims a checkpoint round-trip. Before tagged native data can be checkpointed, a future checkpoint schema and restore owner must preserve and validate the explicit kind with the canonical payload; inspection-only reconstruction or inference from strategy is insufficient. | M2 coordinates the later schema/restore owner with P3/P4 conversion; gate with tagged PDF/PMF checkpoint/restart round-trip and ambiguous/legacy payload rejection tests before claiming support. |
| Physical number/PDF accessor; `representation.py:566-592`, `coagulation_rate.py:160-172` | Resolved N/V; PMF sum directly; PDF radius trapezoid directly, distinct units and no mutation. Existing getter divides all types and total uses a bare sum. | P3/P4; independent resolved non-unit-volume and fixed-V=1 PDF/PMF arithmetic. |
| Population species inventory; `condensation_strategies.py:995-1018,2330-2358` | Sum per-species mass × PMF or resolved weight (divide by V only for resolved); PDF integrates mass × PDF. Return kg/m³ density and, if needed, density × physical V for kg inventory. Keep per-particle `total_mass` distinct; fresh computed arrays, no writable view. | P3/P4 then M3/M4 process corrections; compare each species and V. |
| Ordered alignment; `gas_data.py:82-147`, `environment_data.py:45-105`, `gpu/conversion.py:252-377,468-629` | Retain full names and all gas/ratio lanes, separately admit process pairs. `EnvironmentData` has no names: a structural gate can compare box count and ratio width with Sg, but cannot detect same-width ratio permutations. To certify chemistry, the process/binding boundary must receive authoritative expected ordered gas-name provenance tied to the ratio producer and compare it with actual gas names; without that provenance width-only admission cannot certify order. No current cross-container name check exists. CPU copies detach; `copy=True` Warp transfers detach, `copy=False` may alias on Warp CPU. Caller must pass ordered names on gas restore; without them placeholders are lossy. GPU-only vapor pressure is dropped from CPU inspection but canonical resident checkpoint bytes must keep it. | M2 structural seam `particula/aerosol_validation.py` for box/width, M3/M4 process binding for ordered provenance/map; test equal-width permutation rejection where provenance exists, mixed/all-false masks and unequal widths. |

### Correction queue (not assertions of physical correctness)

| Source read / affected case | V≠1 independent expectation and unresolved issue | Owner |
|---|---|---|
| `representation.py:578-592`, PDF | At fixed V=1 a PDF integral of 4.5 m⁻³ differs from the legacy bare sum; grid absent. The legacy V=.25/4 characterization is not a valid PDF contract fixture. | P3/P4 |
| `condensation/condensation_strategies.py:995-1018,2330-2358`, PMF/resolved | For resolved N=8, m_s=0.5e-18 kg, mass is 4e-18 kg and density/gas decrement 4e-18/V kg/m³; for PMF c=8 m⁻³ at V=1 density is 4e-18 kg/m³ without dividing again. Audit actual process support. | M3/M4 |
| `coagulation/coagulation_strategy/coagulation_strategy_abc.py:88-94,395-429,525-579`, `coagulation_rate.py:160-172` | Resolved N=8 at V=.25 implies 32 m⁻³ for rates; PMF/PDF are already density and PDF rates integrate over radius. Audit update units and capability separately. | M3/M4 |
| `wall_loss/wall_loss_strategies.py` | Removal of one resolved count at V=.25 changes density by 4 m⁻³; PMF/PDF density updates have no extra volume factor. Verify supported branches before repair. | M3/M4 |
| `gpu/kernels/condensation.py`, `dilution.py`, `nucleation.py` | Direct condensation uses slot weights in gas coupling; dilution scales concentration, nucleation demand uses J V dt. Verify separately which work buffers are raw and which density; transfer is not itself normalization. No GPU PDF execution claim. | M3/M4 (hypotheses to audit) |
| `gpu/kernels/condensation.py:623-627,652-733,936-949,967-973` | Direct condensation uses one `species_idx` for particle `masses[...,species_idx]`, density and activity/config arrays and gas concentration, molar mass and vapor pressure; finalized transfers reduce into gas at that same index. A permuted gas order silently couples the wrong chemistry; unequal Sp/Sg cannot be treated as an implicit map and may reject or misindex. Audit preflight dimensions and configuration assumptions against an independent expected ordered gas-name → particle-index map, including unmatched gas lanes and false masks, before any correction; P1 does not change execution. | M3/M4 direct GPU condensation correction; add mapped/permuted and unequal-width tests with independent expected species transfers before claiming support. |
| `gpu/kernels/communication.py`, `execution/resident_communication.py`, `gpu/kernels/exhaustion.py` | Communication amount uses C_g V; physical box-volume change conserves inventory by scaling densities, while representative-volume scaling changes statistical weight, not physical box size. Audit exact dispatch before altering either. | M3/M4 (hypotheses to audit) |

Field-read audit (not a physics endorsement): `gpu/kernels/condensation.py`
reads particle concentration and gas concentration in its transfer/finalization
path (`:916-941`, `:1409`); whether each transfer work buffer has raw-mass or
gas-density units needs M3/M4 review. The proposal loop at `:623-733` indexes
particle and gas chemistry with the same `species_idx`; the bounded transfer
at `:967-973` and gas reduction at `:936-949` preserve that index, not an
explicit ordered map. M3/M4 must audit dimensional preflight and sidecar
ordering against independently named species. `gpu/kernels/dilution.py:455-521`
validates particle and gas concentrations at independent widths and passes both
to the exponential update; it does not infer chemistry from equal widths.
`gpu/kernels/nucleation.py:473-510,2108-2122` reads volume, particle weights,
and gas concentration for its planning/commit boundary; the `J V dt` claim
remains a proposed normalization check, not a conclusion from transfer tests.
`gpu/kernels/communication.py:549-720,1186` reads physical box volumes and
both concentrations and explicitly reconstructs gas density as amount/volume.
`execution/resident_communication.py:103-155` retains these fields in a
prepared binding; it is composition, not a second normalization rule.
Representative-volume scaling in `gpu/kernels/exhaustion.py` has a separate
policy purpose; no inference that it evolves physical chamber volume follows
from the identical field name. These audited reads identify correction sites,
not proven defects; M3/M4 must compare each process to independent V≠1 physics
before changing behavior.

### Ordered species and proposed admission matrix

Fixture: gas names `['water','inert','organic']`, molar masses and kg/m³
concentrations in that order, masks `[True,False,True]` and
`[False,False,False]`, environment ratios `[0.7,7.0,1.3]` (the middle
sentinel must survive), particle Sp=2 vs gas Sg=3. Process configuration
proposes ordered integer pairs `((0,1),(2,0))` and expected **full** ordered
gas names; particle-only and unmatched gas lanes remain present. No equal-width
chemical inference or mask compaction; stoichiometry is outside this contract.
Unique nonempty names, unique in-range integer indices on each side, mask=True
for each mapped gas lane, and one-to-one transfers are future requirements.

| Case | Future structural aggregate | Future process admission |
|---|---|---|
| Mixed mask, valid named map | Accept full Sg width | Accept mapped true lanes only |
| All-false, empty map | Accept all lanes | Accept no participating transfers; nonempty map rejects |
| Equal-width permuted chemistry | Width/box checks alone accept; ratio permutation is undetectable without provenance | Reject stale expected ordered gas names and ratio-producer order where authoritative provenance exists; otherwise order is unverified, not certified |
| Unequal Sp=2, Sg=3, unmatched lanes | Accept; no width equality constraint | Accept valid explicit pairs; preserve unmatched lanes |
| Duplicate/empty names | Reject (currently GasData only rejects empty *list*) | Reject before mutation |
| Duplicate or out-of-range/noninteger map indices | No map check at aggregate boundary | Reject before mutation |
| Map to false mask | Accept aggregate | Reject before mutation |
| Invalid box or ratio width | Reject before process mutation | Not reached |
| Invalid V, PDF grid or ambiguous kind | Builder rejection at creation; explicit post-mutation `ParticleData` validation and new interpreting-helper rejection | Processes reject unsupported distribution or invalid interpreted data before mutation |

Future structural `ValueError` covers shapes and box/ratio width; builder and
explicit representation validation own fixed PMF/PDF volume or positive finite
resolved volume. The structural aggregate cannot prove ordered gas/environment
chemistry. Future process
binding must compare expected ordered gas names and ratio-producer order when
authoritative provenance is available, rejecting stale names/map before
mutation; if unavailable, document unverified order rather than claiming
alignment. Process `ValueError` also covers malformed indices/names/mask and
unsupported capabilities (type errors for wrong container types). This table
is **not** a test of current enforcement: `GasData` permits duplicate names,
`EnvironmentData` has no gas names, and `ParticleData` checks shapes but not
physical volume. Neither a validator nor distribution metadata is added in P1.

### P1 validation and scientific gate

P1 status **at the initial validation recording**: PENDING complete scientific
sign-off; P2 blocked. See the later maintainer approval update below. The initial
draft lacked Kyle Gorkowski's approval on PDF grid/integration, provenance and
normalization. Later issue #1612 clarifications now choose the radius grid,
trapezoids, explicit kind, fixed PMF/PDF volume and resolved inverse-volume
normalization; the complete revised ledger and test evidence have not been
approved. Existing planning approval is not a P1 completion gate. No
executable physics changed.

Worktree `trees/f853734e`, branch `issue-1610-adw-f853734e`,
2026-09-28. Focused assertion checks with coverage disabled after formatting:
particle/gas/environment modules 132 passed; Warp conversion module 127
passed (Warp CPU baseline installed; optional CUDA evidence not claimed).
Untargeted repository-configured full-suite run by `adw-build-tests`:
6,922 passed, 20 skipped, 93% full-package coverage; reported as
`ADW_BUILD_TESTS_SUCCESS`. Focused Ruff checks and format checks on four
changed test files passed after formatting; repository-wide lint/mypy and
`.opencode/tools/run_linters.py` were not run in this build (deferred to
polish; do not mark them passed). No rendered docs changed, so strict MkDocs
is not a P1 build target. These results characterize current behavior only;
scientific review remains an independent, blocking gate.
At the build-phase recording, base revision `dcd6adc52` was the worktree
branch's pre-P1 commit and the P1 test and ledger edits were still uncommitted.
The focused checks and runner above were build-phase evidence, not a claim of
subsequent refine-phase or scientific validation. No actual scientific approval
had been supplied at that recording.

Refinement validation (2026-09-28, same worktree): focused CPU modules
132 passed and Warp conversion 127 passed, all without coverage. The
untargeted repository suite reported 6,922 passed, 20 skipped and 1 xfailed;
full-package coverage was 93% (repository policy passed). Repository-wide
Ruff check and mypy passed, and changed-test-file Ruff format-check passed.
Optional CUDA was not claimed as measured evidence. Scientific P1 approval
remains pending and P2 remains blocked despite passing characterization.

Commit reconciliation (2026-09-28): P1 characterization tests and the draft
ledger were committed at `5d8f36964` (issue #1610). The earlier uncommitted
description is retained as an at-time build observation, not current commit
state. The recorded build/refinement checks are not a fresh post-commit
validation claim. Kyle Gorkowski's scientific sign-off on PDF integration,
metadata/defaults and normalization is still pending; P1 is not approved or
shipped, and P2 remains blocked.

Review-fix validation (2026-09-28, same worktree): the new proposed PDF
varying-mass oracle passed with the particle module (55 passed), and the real
unequal-width fixture passed with the gas/environment modules (80 passed), both
focused and without coverage. Delegated untargeted full-package pytest reported
6,925 passed, 20 skipped and 93% coverage, with repository policy passing.
Strict MkDocs validation of the final ledger revision passed. Ruff,
formatting and mypy belong to subsequent Validate/Polish steps and are not
claimed here; `.opencode/tools/run_linters.py` was not run in the fix pass.
These are characterization checks, not scientific sign-off: P1 remains pending
and P2 remains blocked.

Build entry-gate recheck for issue #1611 (workflow `94cf733d`, 2026-09-28):
at worktree HEAD `6091a0d9e`, the P1 ledger still records Kyle Gorkowski's
scientific approval as pending. No identified revision approving inverse-volume
normalization, PDF grid/integration, or distribution and ratio-producer
provenance was available in the reviewed P1 phase/dependency/appendix records.
The P1 ledger also proposes positive finite volume as future structural
admission (lines 150–158), whereas E9 D3 assigns physical-domain checks to
process admission. This conflict needs an explicit maintainer decision; it is
not resolved by the prior epic planning approval or passing characterization
tests. P2 steps 2–5, their acceptance matrix, additional characterization
tests, and any P3/M2 authorization were not undertaken. P1 and P2 remain
unapproved; no P2 test or review result is claimed by this gate recheck.

## Maintainer clarification for P1 handoff (2026-09-28)

Kyle Gorkowski clarified the intended representation semantics during the
issue #1612 fix discussion. This section supersedes conflicting **proposals**
above and E9 D1 for future implementation; it does not rewrite historical
characterization outcomes or claim that current processes already comply.

| Representation | Stored concentration | Bulk number density | Particle species mass density |
|---|---|---|---|
| Discrete / PMF | Number per bin per m³ | Sum bin values | Sum bin value × per-particle species mass |
| Continuous radius PDF | `dn/dr` per m⁴ | Integrate PDF over radius | Integrate per-particle species mass × PDF over radius |
| Particle resolved | Represented counts in physical volume V | Sum counts / V | Sum count × per-particle species mass / V |

PMF and PDF have fixed `volume == 1 m³`; their concentration fields are
**already densities** and must not be divided by volume again. Resolved data
uses finite, strictly positive physical volume, including for empty populations.
`total_mass` remains per-particle mass, not population inventory. Extensive
mass, when needed, is mass density times the physical volume, not a mislabeled
per-particle result. Inverse normalization (density-to-storage) multiplies by
V only for particle-resolved counts; PMF/PDF density values remain on their
existing basis. The older PDF examples at V=.25 and V=4 are invalid under this
fixed-volume rule; non-unit-volume oracles belong to particle-resolved cases.

The PDF grid is supplied explicitly as radius in metres, with one node per PDF
value, at least two finite, strictly increasing, positive nodes. Trapezoidal
integration applies on the supplied interval (no extrapolation); multiply
per-particle species mass by PDF **before** integration. An empty PDF may retain
a valid grid with zero PDF values and zero particle mass so it can later gain
or lose particles on that grid. PDF slot summation is not integration. PMF and
particle-resolved bulk properties use weighted sums, not PDF quadrature.

The caller is responsible for ensuring that additions or changes to the object
match its declared representation. Do not infer PDF/PMF provenance from a
facade, strategy, array shape, or weights, and do not silently convert updates
between representations. Backward compatibility for untagged legacy data is
not required: future builders must create data with an explicit kind. Enforce
the representation-specific volume rule **when creating through the builder**;
the maintainer did not authorize a blanket new `ParticleData` constructor
check. Direct construction and later raw-array mutation require an explicit
read-only `ParticleData` representation-validation method. Callers should
invoke it after inputting/updating data, before using that data; new bulk
read-only helpers must invoke the same validation before interpreting it.
Validation checks declared kind, compatible volume, grid and array schema,
rejects invalid data without mutation, and must not silently repair or
reinterpret caller input. Direct writable NumPy-array mutation cannot be
intercepted automatically, so validation is a deliberate post-mutation gate;
process admission for paths using the new interpretation must call it too.
Likewise, broadening the currently restricted CPU nucleation runnable to PDF
or other representations is not approved by this clarification.

**P1 test reconciliation required:** Keep old assertions as legacy
characterization, not new contract evidence. Replace proposed V=.25/4 PDF
oracles with fixed-V=1, m⁻⁴ radius-PDF trapezoidal count and independently
hand-calculated per-species mass-density examples, including varying masses
and a valid grid with zero PDF/mass. Keep V=.25/1/4 number- and mass-density
oracles only for explicitly declared particle-resolved data; add PMF bin sums
at fixed V=1, including fractional and zero values. P1 must specify builder
rejection cases for missing kind, incorrect fixed PMF/PDF volume, invalid
resolved volume, and invalid PDF grid, plus post-mutation validation rejection
with unchanged arrays and metadata; P3 adds the runnable assertions alongside
the corresponding implementation, not expected-failing future tests in P1.
Test the method after direct construction/mutation and via new bulk helpers
when available; legacy facade arithmetic and raw Warp transfer tests are not
evidence of tagged support.

**Remaining gates:** Reconcile P1's normalization/alignment ledger and
independent test expectations with the clarified units; specify metadata/grid
copy and unsupported transfer handling, validate and obtain explicit P1 gate
approval; then complete and validate P2's acceptance/ownership matrix.
Builder creation, post-mutation validation and new read-only helper admission
are separate from E9 D3's process-specific physical checks. P3 issue #1612
remains blocked until predecessor gates pass. No code, tests, transfer,
resident, checkpoint, or documentation-render validation is claimed by this
decision record.

## P2 replacement and ownership gate (specification pending approval)

P2 may **start** only after the revised P1 ledger, independent scientific
fixtures and validation ownership are reconciled, checked at a recorded
revision, and explicitly approved by Kyle. The preceding issue #1610
characterization and issue #1612 maintainer clarifications alone are not
proof that this gate passed. P2 is a specification/characterization phase;
the future M2 track implements `Aerosol` property setters and `replace_data`.

Proposed acceptance matrix for maintainer review (not implemented or signed
off). In each case validate the complete candidate triple without coercing
held objects, publishing partial references, modifying arrays/names, or
uploading/rebinding GPU/resident resources. Do not infer species identity from
equal width; gas/environment lanes retain full ordered gas names and masks.

| Candidate operation | Required outcome and ownership |
|---|---|
| Assign the same held object through an individual property | Accept; getter still returns that exact object; no copy or rebuild. |
| Assign a compatible, distinct candidate through one setter | Accept; retain the supplied candidate by identity and keep the other two references unchanged. |
| Assign an individually incompatible candidate | Reject before publication; all old references/arrays and candidate arrays stay unchanged. |
| Change box/species layout by one setter while other two remain held | Reject if the resulting triple is incompatible; no partial publication. |
| Replace all three with a compatible new layout | Accept all three supplied objects by identity together; no hidden copy or GPU/resident rebinding. |
| Replace all three with one invalid volume, kind, grid, box/width or gas/name candidate | Reject the entire proposed triple; original identities and both old/candidate payloads remain unchanged. |
| Replace with mixed gas flags, all-false mask, or Sp≠Sg | Accept structurally compatible full gas/environment width and ordered names; mapping/physics compatibility stays at process admission. |

Before P2 approval, specify exactly which candidate checks call the
read-only `ParticleData` representation validator after mutable input, which
checks belong to aggregate structure (box/gas/environment width, unique
nonempty gas names), and which stay process-only (species mapping, physical
capability, single-box limit). Freeze field-level copy/view behavior and
validation order, including one-bad-box rejection, empty cases, same-object
assignment and candidate mutation after acceptance. Back P2's specification
with adjacent **current-behavior characterization** only, not expected-failing
future aggregate tests. Record actual checks, revision, reviewer and explicit
P2 matrix approval before permitting P3; none was claimed at this planning
recording. See the later maintainer approval update below.

## Issue #1612 provisional P3 implementation check (2026-09-28)

After the maintainer requested the corrected P3 implementation, worktree
`trees/8ef4d823` now contains an uncommitted slice: explicit kind/radius-grid
metadata and copy detachment, a read-only post-mutation validator, fresh CPU
slot/number/species density and inventory helpers, builder kind/units/volume
admission, and early rejection at untagged facade/Warp upload boundaries.
Existing untagged direct construction and Warp round trips remain a separate
legacy array-only path; tagged Warp download/restart and GPU PDF execution
are not implemented or claimed. No process physics or nucleation support was
broadened.

Independent CPU cases check non-unit-volume resolved counts, fixed-V=1 PMF,
nonuniform varying-mass PDF quadrature, empty PDF grid retention, post-mutation
and one-bad-box rejection, and pre-upload tagged exclusion. The untargeted
repository runner at the implementation revision before the final added
one-bad-box test reported 6,936 passed, 20 skipped and 93% coverage.
That final test subsequently passed as a focused assertion (12 passed), but
the focused wrapper still reported failure on the unrelated full-package
coverage threshold (34%); it cannot express the testing guide's `--no-cov`.
Repository-wide Ruff and mypy, targeted Ruff format-check, and strict MkDocs
validation passed before that final test addition; targeted format-check for
the final test also passed. This is **not** a post-commit or signed-off P3
closeout: at the time of this validation recording, P1/P2 approvals and final
review were still required before marking #1612 complete or shipping it. The
subsequent maintainer approval below resolves the sign-off decisions, not the
final implementation review.

### Maintainer approval update (2026-09-28)

Kyle Gorkowski explicitly approved the revised P1 scientific rules and P2
replacement/ownership specification in the issue #1612 conversation after
reviewing the corrected P3 implementation. This approval covers the fixed-V=1
already-density PMF/PDF basis, resolved count/V basis, explicit positive radius
grid with trapezoidal PDF integration, builder-time volume/kind checks,
post-mutation read-only validation, and the proposed P2 acceptance matrix
above. It supersedes the earlier *pending-approval* status for those decisions.
The dated historical P1 characterization checks are not retrospective tests
of the revised semantics, and P2's future aggregate setter/replace_data
implementation has not been delivered or validated. The P3 code and its
validation evidence remain subject to normal final review and shipping gates;
approval of a contract does not manufacture passing implementation evidence.
