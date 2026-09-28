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

This is a proposed physical contract, **not** an assertion that existing
processes implement it. P1 changes tests and this ledger only. In particular,
`ParticleData` does not store a distribution kind or a PDF radius grid.

| Owner/field | Raw storage, shape and units | Proposed physical use, order and ownership |
|---|---|---|
| `ParticleData.masses` | `(B,N,Sp)` kg per representative particle | Species order is the particle lane order; `total_mass` is `(B,N)` kg **per particle**, not population inventory. Mutable CPU arrays; `copy()` detaches. |
| `ParticleData.concentration`, discrete | `(B,N)` raw counts N (may be fractional weights) | Number density N/V [m⁻³]; species extensive mass `sum_i N_i m_i,s` [kg], density that sum/V [kg/m³]. No unit-weight-only algorithm is implied. |
| Same field, particle resolved | `(B,N)` raw counts N, normally unity for active slots | Same N/V and mass equations; algorithms that require unit weights must separately reject fractional weights. Fully zero mass/weight/charge slots are inactive; do not count them. |
| Same field, continuous PDF | `(B,N)` raw dN/dr [m⁻¹] at ascending radii in m | `integral (dN/dr) dr` gives raw number; `integral m_s(r) (dN/dr) dr` gives kg. Divide once by V for physical density. A slot sum is not an integral. |
| `charge`, `density`, `volume` | `(B,N)` elementary-charge counts, `(Sp,)` kg/m³, `(B,)` m³ | Charge follows representative slot (not population charge); density shares particle species order; positive finite volume is required for future physical access. Existing shape-only constructor does not check its sign. |
| `GasData` | ordered `name` `(Sg,)`, molar_mass `(Sg,)` kg/mol, partitioning `(Sg,)` bool, concentration `(B,Sg)` kg/m³ | Full gas order retained even for false mask lanes; gas extensive mass per species `C_g,s V` [kg]. Mutable arrays and names; `copy()` detaches. Duplicate names are currently tolerated. |
| `EnvironmentData` | temperature `(B,)` K, pressure `(B,)` Pa, saturation_ratio `(B,Sg)` dimensionless | Ratio lane order must follow **all** gas names, including nonpartitioning lanes; environment has no names or cross-container check today. `copy()` detaches. |

For a physical density increment Δn [m⁻³], write raw ΔN=V Δn;
for a physical PDF increment Δ(dn/dr) [m⁻⁴], write raw
Δ(dN/dr)=V Δ(dn/dr). Do not apply this conversion to an already raw
transfer or divide gas concentration again. For example, eight particles at
V=0.25/1/4 have densities 32/8/2 m⁻³; a mass vector
`[0.5e-18,1.5e-18]` contributes `[4e-18,12e-18]` kg extensive,
independent of V. Fractional-weight slots add their own weight × mass.

**Proposed PDF grid gate (requires Kyle Gorkowski approval):** radius in metres,
finite, strictly increasing, at least two distinct nodes, node count matching
the PDF width; proposal: positive first node and positive last node, closed
sampled interval with no extrapolation beyond endpoints. On nonuniform grids
use interval-wise linear trapezoids (candidate `np.trapezoid(y, x=radius)`,
as in existing coagulation integration); multiply species mass by PDF *before*
quadrature. Neither diameter nor log-radius may replace radius without the
Jacobian. Independent interval arithmetic for `r=[1e-9,2e-9,4e-9]` m,
`dN/dr=[1e9,2e9,1e9]` m⁻¹: interval counts 1.5 and 3, total 4.5;
physical densities at V=0.25/1/4 are 18/4.5/1.125 m⁻³. Constant
`m_s=2e-18` kg gives 9e-18 kg extensive, and
3.6e-17/9e-18/2.25e-18 kg/m³. `sum(dN/dr)=4e9` is dimensionally wrong.
No PDF metadata, grid validation or integration is implemented in P1.

### Proposed helper and provenance decisions (not implemented)

| Future seam / consumer | Direction, reuse and missing rule | Owner and tests |
|---|---|---|
| Distribution vocabulary and explicit metadata; `particles/representation_builders.py`, `particle_data.py:235-359` | Future constructor/builder and facade conversion must receive an explicit source distribution-kind tag and preserve it through native copy and explicit Warp conversion. `RadiiBasedMovingBin` supports both PDF and PMF: strategy class, shape and weights cannot distinguish them. Reject absent or ambiguous provenance instead of guessing a default or accepting a caller's unverified preference. Current conversions remain unchanged in P1. | M2 construction/conversion gate and P3/P4 access; test PDF vs PMF from the same strategy, missing provenance rejection and supported round-trips. |
| Resident checkpoint/restore; `execution/checkpoint.py` | P1 neither persists distribution metadata nor claims a checkpoint round-trip. Before tagged native data can be checkpointed, a future checkpoint schema and restore owner must preserve and validate the explicit kind with the canonical payload; inspection-only reconstruction or inference from strategy is insufficient. | M2 coordinates the later schema/restore owner with P3/P4 conversion; gate with tagged PDF/PMF checkpoint/restart round-trip and ambiguous/legacy payload rejection tests before claiming support. |
| Physical number/PDF accessor; `representation.py:566-592`, `coagulation_rate.py:160-172` | Read raw N or dN/dr and V; return N/V or integral/V, with distinct units and no mutation. Existing getter divides but total uses a bare sum. | P3/P4; compare against independent nonuniform-grid arithmetic. |
| Population species inventory; `condensation_strategies.py:995-1018,2330-2358` | Read slot masses and weights, return kg and kg/m³ (PDF quadrature), never conflate `ParticleData.total_mass` with inventory. Fresh computed arrays, no writable view. | P3/P4 then M3/M4 process corrections; compare each species and V. |
| Ordered alignment; `gas_data.py:82-147`, `environment_data.py:45-105`, `gpu/conversion.py:252-377,468-629` | Retain full names and all gas/ratio lanes, separately admit process pairs. `EnvironmentData` has no names: a structural gate can compare box count and ratio width with Sg, but cannot detect same-width ratio permutations. To certify chemistry, the process/binding boundary must receive authoritative expected ordered gas-name provenance tied to the ratio producer and compare it with actual gas names; without that provenance width-only admission cannot certify order. No current cross-container name check exists. CPU copies detach; `copy=True` Warp transfers detach, `copy=False` may alias on Warp CPU. Caller must pass ordered names on gas restore; without them placeholders are lossy. GPU-only vapor pressure is dropped from CPU inspection but canonical resident checkpoint bytes must keep it. | M2 structural seam `particula/aerosol_validation.py` for box/width, M3/M4 process binding for ordered provenance/map; test equal-width permutation rejection where provenance exists, mixed/all-false masks and unequal widths. |

### Correction queue (not assertions of physical correctness)

| Source read / affected case | V≠1 independent expectation and unresolved issue | Owner |
|---|---|---|
| `representation.py:578-592`, PDF | Raw integral 4.5 gives 18 at V=.25; current bare sum yields 1.6e10 instead. PDF grid absent. | P3/P4 |
| `condensation/condensation_strategies.py:995-1018,2330-2358`, discrete/resolved | An 8-weight slot's 0.5e-18 kg species is 4e-18 kg extensive regardless of V; gas decrement must be 4e-18/V kg/m³. Audit raw vs normalized reads before correction. | M3/M4 |
| `coagulation/coagulation_strategy/coagulation_strategy_abc.py:88-94,395-429,525-579`, `coagulation_rate.py:160-172` | At V=.25 a raw count 8 is 32 m⁻³ for rate evaluation; any extensive collision update must return to raw N units. Existing algorithms may have unit-weight limits. | M3/M4 |
| `wall_loss/wall_loss_strategies.py` | Removal of one raw count at V=.25 changes density by 4 m⁻³, not 1; verify actual distribution branches before repair. | M3/M4 |
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
| Invalid V, PDF grid, ambiguous kind or unsupported distribution | Future structural/metadata rejection where applicable | Reject capability or bounds before mutation |

Future structural `ValueError` covers shapes, box/ratio width and positive
finite volume; it cannot prove ordered gas/environment chemistry. Future process
binding must compare expected ordered gas names and ratio-producer order when
authoritative provenance is available, rejecting stale names/map before
mutation; if unavailable, document unverified order rather than claiming
alignment. Process `ValueError` also covers malformed indices/names/mask and
unsupported capabilities (type errors for wrong container types). This table
is **not** a test of current enforcement: `GasData` permits duplicate names,
`EnvironmentData` has no gas names, and `ParticleData` checks shapes but not
physical volume. Neither a validator nor distribution metadata is added in P1.

### P1 validation and scientific gate

P1 status: **PENDING scientific sign-off**; P2 blocked. Reviewer: Kyle
Gorkowski, approval on PDF grid/integration, provenance/default and inverse
volume normalization **not received**. Existing planning approval is not P1
scientific approval. No executable physics changed.

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
